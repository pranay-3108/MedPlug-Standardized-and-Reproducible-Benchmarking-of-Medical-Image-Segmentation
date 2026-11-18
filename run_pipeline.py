"""Run one isolated experiment without keeping the web UI open."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from core.plugin_loader import load_config_yaml
from experiment_manager import ExperimentManager


def _write_state(path: Path, run_id: str, status: str, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "status": status, "updated_at": time.time(), **extra}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _utf8_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _model_from_config(config_path: Path) -> str:
    data = load_config_yaml(config_path)
    return str((data.get("model") or {}).get("plugin", "unet"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a MedPlugBench experiment independently of app.py.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=False, default=None, help="Existing run ID; omit to create a new run.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = args.config.expanduser().resolve()
    manager = ExperimentManager()
    if not config_path.is_file():
        if args.run_id:
            try:
                paths = manager.get_paths(args.run_id)
                _write_state(paths.root / "process_state.json", args.run_id, "FAILED", stage="CONFIG", error=f"Config not found: {config_path}")
            except Exception:
                pass
        print(f"Config not found: {config_path}")
        return 2

    model_name = _model_from_config(config_path)
    if args.run_id:
        try:
            paths = manager.get_paths(args.run_id)
        except FileNotFoundError:
            paths = manager.create_experiment(
                model_name,
                checkpoint_name=f"{model_name}_best_model",
                resume=False,
                run_id=args.run_id,
            )
    else:
        paths = manager.create_experiment(
            model_name,
            checkpoint_name=f"{model_name}_best_model",
            resume=False,
        )

    run_id = paths.run_id
    state_path = paths.root / "process_state.json"
    logs_dir = paths.root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "pipeline.log"

    _write_state(
        state_path,
        run_id,
        "STARTING",
        stage="PREFLIGHT",
        pid=os.getpid(),
        config=str(config_path),
        command=f'"{sys.executable}" "{root / "run_pipeline.py"}" --config "{config_path}" --run-id "{run_id}"',
    )

    with log_path.open("a", encoding="utf-8") as log:
        def write(line: str) -> None:
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        write(f"=== MedPlugBench Pipeline: {run_id} ===")
        write(f"Config: {config_path}")
        write(f"Model: {model_name}")
        write("Stage: PREFLIGHT")

        preflight = subprocess.run(
            [sys.executable, str(root / "preflight.py"), "--config", str(config_path)],
            cwd=root,
            capture_output=True,
            text=True,
            env=_utf8_env(),
        )
        preflight_output = (preflight.stdout or "") + (preflight.stderr or "")
        log.write(preflight_output)
        log.flush()
        (logs_dir / "preflight.log").write_text(preflight_output, encoding="utf-8")
        if preflight.returncode != 0:
            message = f"Preflight failed with exit code {preflight.returncode}. Training was not started."
            write(message)
            manager.update_stage(paths, "PREFLIGHT", status="FAILED", error=message)
            _write_state(state_path, run_id, "FAILED", stage="PREFLIGHT", exit_code=preflight.returncode, error=message)
            print(f"Preflight log: {logs_dir / 'preflight.log'}")
            return preflight.returncode

        write("Preflight passed.")
        _write_state(state_path, run_id, "RUNNING", stage="TRAINING", pid=os.getpid(), config=str(config_path))
        write("Stage: TRAINING")
        command = [sys.executable, "-u", str(root / "train.py"), "--config", str(config_path), "--run-id", run_id]
        command_text = " ".join(f'"{part}"' for part in command)
        write("Command: " + command_text)
        (paths.root / "audit").mkdir(parents=True, exist_ok=True)
        (paths.root / "audit" / "run_command.txt").write_text(command_text + "\n", encoding="utf-8")
        child = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=_utf8_env(),
        )
        current_epoch = None
        if child.stdout is not None:
            for raw_line in child.stdout:
                line = raw_line.rstrip("\r\n")
                write(line)
                stripped = line.strip()
                if stripped.startswith("Epoch ") and "/" in stripped:
                    try:
                        current_epoch = int(stripped.split("/", 1)[0].split()[-1].rstrip(":"))
                        _write_state(
                            state_path,
                            run_id,
                            "RUNNING",
                            stage="TRAINING",
                            pid=child.pid,
                            pipeline_pid=os.getpid(),
                            config=str(config_path),
                            command=command_text,
                            epoch=current_epoch,
                        )
                    except (ValueError, IndexError):
                        pass
        code = child.wait()

    if code != 0:
        manager.update_stage(paths, "TRAINING", status="FAILED", error=f"Training process exited with code {code}")
        _write_state(state_path, run_id, "FAILED", stage="TRAINING", exit_code=code, epoch=current_epoch)
        print(f"PIPELINE FAILED: {run_id} (exit code {code})")
        print(f"Log: {log_path}")
        return code

    manifest = manager.load_manifest(paths)
    final_stage = manifest.get("stage") or "COMPLETED"
    final_status = manifest.get("status") or "COMPLETED"
    _write_state(state_path, run_id, final_status, stage=final_stage, exit_code=0)
    print(f"PIPELINE COMPLETED: {run_id}")
    print(f"Results: {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
