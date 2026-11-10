from contextlib import contextmanager

from daytona import Daytona
from daytona.v0 import CreateSnapshotParams, Image, Resources
from daytona.v0.common.errors import DaytonaNotFoundError
from daytona.v0.common.sandbox import GpuType


@contextmanager
def daytona_sandbox(daytona: Daytona):
    sandbox = daytona.v0.create()

    try:
        yield sandbox
    finally:
        sandbox.stop()
        sandbox.delete()


def get_or_create_gpu_snapshot(daytona: Daytona, snapshot_name: str):
    try:
        snapshot = daytona.v0.snapshot.get(snapshot_name)
    except DaytonaNotFoundError:
        snapshot = daytona.v0.snapshot.create(
            CreateSnapshotParams(
                name=snapshot_name,
                image=Image.base("python:3.12").pip_install_from_pyproject(
                    "pyproject.toml",
                    optional_dependencies=[],  # optional dependencies parameter is not optional :)
                ),
                resources=Resources(cpu=1, memory=1, disk=1, gpu=1, gpu_type=GpuType.H100),
            ),
        )

    return snapshot


def main():
    # Create a Daytona instance
    daytona = Daytona()

    # Get or create a GPU snapshot
    snapshot_name = "my_gpu_snapshot"
    volume_name = "my_gpu_volume"

    snapshot = get_or_create_gpu_snapshot(daytona, snapshot_name)
    volume = daytona.v0.volume.get(volume_name, create=True)

    # with daytona_sandbox(daytona) as sandbox:
    #     response = sandbox.process.code_run(
    #         dedent(
    #             """
    #             print("Hello, World!")
    #             """
    #         )
    #     )
    #
    # print(response.result)


if __name__ == "__main__":
    main()
