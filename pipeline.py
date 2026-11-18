"""Post-training pipeline orchestration with stage-level failure isolation."""

from dataclasses import dataclass, field


@dataclass
class PipelineResult:
    stages: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def success(self):
        return all(value == "PASS" for value in self.stages.values()) and not self.warnings


def run_post_training_pipeline(
    *,
    experiment_manager,
    experiment_paths,
    args,
    device,
    summary,
    validation_results,
    test_results,
    best_epoch,
    stopped_epoch,
    training_time,
    test_time,
    early_stopping_reason,
    sample_factory,
    visualizer,
    report_generator,
):
    """Run visualization and reporting after metrics are safely committed.

    A failure in one post-training stage is recorded and does not erase metrics,
    checkpoints, or the completed benchmark result.
    """
    result = PipelineResult()

    experiment_manager.update_stage(experiment_paths, "POST_TRAINING", status="RUNNING")

    try:
        sample = sample_factory()
        if sample is None:
            raise RuntimeError("No visualization sample could be produced from the validation loader.")
        image, mask, output = sample
        visualizer.save_visualizations(
            experiment_paths,
            image,
            mask,
            output,
        )
        visualizer.plot_training_history(experiment_paths)
        visualizer.plot_test_metrics(experiment_paths, test_results)
        visualizer.update_model_comparison(
            experiment_paths.model_comparison_path,
            experiment_manager.benchmark_results_path,
        )
        required = [
            experiment_paths.overlay_path,
            experiment_paths.prediction_path,
            experiment_paths.ground_truth_path,
            experiment_paths.training_loss_plot_path,
            experiment_paths.validation_metrics_plot_path,
            experiment_paths.test_metrics_plot_path,
            experiment_paths.model_comparison_path,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("Visualization stage completed without required artifacts: " + "; ".join(missing))
        result.stages["visualization"] = "PASS"
        experiment_manager.update_stage(experiment_paths, "VISUALIZATION", status="COMPLETED")
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        result.stages["visualization"] = "FAILED"
        result.warnings.append(f"Visualization: {message}")
        experiment_manager.update_stage(experiment_paths, "VISUALIZATION", status="FAILED", error=message)

    try:
        report_generator.generate(
            experiment_paths,
            args,
            device,
            summary,
            validation_results,
            test_results,
            best_epoch,
            stopped_epoch,
            training_time,
            test_time,
            early_stopping_reason,
            benchmark_success=True,
            visualization_status=(
                "PASS" if result.stages.get("visualization") == "PASS" else "FAILED"
            ),
        )
        result.stages["report"] = "PASS"
        experiment_manager.update_stage(experiment_paths, "REPORT", status="COMPLETED")
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        result.stages["report"] = "FAILED"
        result.warnings.append(f"Report: {message}")
        experiment_manager.update_stage(experiment_paths, "REPORT", status="FAILED", error=message)

    if result.warnings:
        experiment_manager.update_stage(
            experiment_paths,
            "COMPLETED_WITH_WARNINGS",
            status="COMPLETED",
            error="; ".join(result.warnings),
        )
    else:
        experiment_manager.update_stage(experiment_paths, "COMPLETED", status="COMPLETED")

    return result
