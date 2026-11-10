"""Run CartpoleBalance in a disposable Daytona sandbox without checkpoints."""

import argparse
import tomllib
from pathlib import Path

from daytona import Daytona
from daytona.v0 import CreateSandboxFromImageParams, Image, Resources
from daytona.v0.common.sandbox import GpuType
from dotenv import load_dotenv

from daytona_runner.train import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Use one H100 instead of CPU")
    parser.add_argument("--total-steps", type=int, default=Config().total_steps)
    args = parser.parse_args()
    if args.total_steps <= 0:
        parser.error("--total-steps must be positive")

    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    image = (
        Image.base("python:3.14-slim-bookworm")
        .run_commands("apt-get update && apt-get install -y --no-install-recommends git libgl1")
        .pip_install("uv==0.13.4")
        .workdir("/app")
        .add_local_file(root / "pyproject.toml", "/app/pyproject.toml")
        .add_local_file(root / "uv.lock", "/app/uv.lock")
        .run_commands("uv sync --locked --no-dev && uv cache clean")
        .add_local_dir(root / "daytona_runner", "/app/daytona_runner")
    )
    resources = Resources(cpu=4, memory=8, disk=10)
    if args.gpu:
        lock = tomllib.loads((root / "uv.lock").read_text())
        jax_version = next(p["version"] for p in lock["package"] if p["name"] == "jax")
        image = image.run_commands(
            f"uv pip install --python /app/.venv/bin/python 'jax[cuda12]=={jax_version}' "
            "&& uv cache clean"
        )
        resources = Resources(cpu=4, memory=8, disk=20, gpu=1, gpu_type=GpuType.H100)

    with Daytona() as daytona:
        sandbox = daytona.v0.create(
            CreateSandboxFromImageParams(
                image=image,
                resources=resources,
                auto_stop_interval=0,
                ttl_minutes=65,
                ephemeral=True,
            ),
            timeout=1200,
            on_snapshot_create_logs=lambda chunk: print(chunk, end="", flush=True),
        )
        try:
            print(f"Sandbox: {sandbox.id}; training started (up to 60 minutes).", flush=True)
            result = sandbox.process.exec(
                "/app/.venv/bin/python -u -m daytona_runner.train "
                f"--task CartpoleBalance --{'gpu' if args.gpu else 'cpu'} "
                f"--total-steps {args.total_steps} --no-checkpoints --output /app/runs",
                cwd="/app",
                timeout=3600,
            )
            print(result.result, flush=True)
            archive = sandbox.process.exec("tar -czf /tmp/results.tar.gz -C /app runs", timeout=60)
            if archive.exit_code == 0:
                output = root / "runs" / f"daytona-{sandbox.id}.tar.gz"
                output.parent.mkdir(parents=True, exist_ok=True)
                sandbox.fs.download_file("/tmp/results.tar.gz", str(output))
                print(f"Results: {output}")
            elif result.exit_code == 0:
                raise RuntimeError(f"Could not archive results: {archive.result}")
            return result.exit_code
        finally:
            sandbox.delete()


if __name__ == "__main__":
    raise SystemExit(main())
