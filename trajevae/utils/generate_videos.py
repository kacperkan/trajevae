import argparse
import shlex
import subprocess
from pathlib import Path

import tqdm
from functional import seq

OUTPUT_FOLDER = Path("outputs") / "videos"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_EXTENSION = ".mp4"


def generate_video(pattern: str, out_path: Path):
    command = shlex.split(
        (
            "ffmpeg -framerate 20 -i {} -c:v libx264 -pix_fmt yuv420p -y {}"
        ).format(pattern, out_path.as_posix())
    )
    subprocess.call(command)


def process_single_rendering_files_in_folder(folder: str):
    folder_path = Path(folder)
    output_folder = OUTPUT_FOLDER / folder_path.name
    output_folder.mkdir(parents=True, exist_ok=True)

    unique_patterns = list(
        seq(folder_path.rglob("*.png"))
        .map(
            lambda x: x.parent
            / (
                x.name.split("_")[0]
                + "_%04d_"
                + "_".join(x.name.split("_")[2:])
            )
        )
        .filter_not(lambda x: "deterministic" in x.as_posix())
        .filter_not(lambda x: x.name.startswith("trajectory"))
        .to_set()
    )
    for pattern in tqdm.tqdm(sorted(unique_patterns)):
        output_file_path = (
            output_folder / pattern.relative_to(folder_path)
        ).with_suffix(OUTPUT_EXTENSION)
        output_file_path = (
            output_file_path.parent
            / output_file_path.name.replace("_%04d", "")
        )
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        generate_video(pattern, output_file_path)


def process_several_rendering_files_in_folder(folder: str):
    folder_path = Path(folder)
    output_folder = OUTPUT_FOLDER / folder_path.name
    output_folder.mkdir(parents=True, exist_ok=True)

    unique_patterns = list(
        seq(folder_path.rglob("*.png"))
        .map(
            lambda x: x.parent
            / (
                "_".join(x.with_suffix("").name.split("_")[:-1])
                + "_%04d"
                + x.suffix
            )
        )
        .filter_not(lambda x: x.name.startswith("trajectory"))
        .to_set()
    )
    for pattern in tqdm.tqdm(sorted(unique_patterns)):
        output_file_path = (
            output_folder / pattern.relative_to(folder_path)
        ).with_suffix(OUTPUT_EXTENSION)
        output_file_path = (
            output_file_path.parent
            / output_file_path.name.replace("_%04d", "")
        )
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        generate_video(pattern, output_file_path)


def process_scene_files_in_folder(folder: str):
    folder_path = Path(folder)
    output_folder = OUTPUT_FOLDER / folder_path.name
    output_folder.mkdir(parents=True, exist_ok=True)

    unique_patterns = list(
        seq(folder_path.rglob("*.png"))
        .map(lambda x: x.parent / "%04d.png")
        .to_set()
    )
    for pattern in tqdm.tqdm(unique_patterns):
        output_file_path = (
            output_folder / pattern.relative_to(folder_path)
        ).with_suffix(OUTPUT_EXTENSION)
        output_file_path = (
            output_file_path.parent
            / output_file_path.name.replace("%04d", "scene")
        )
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        generate_video(pattern, output_file_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos from all the data under the given folder"
    )
    parser.add_argument("in_folder")
    parser.add_argument(
        "type",
        help="Where the rendered images come from",
        choices=["scene", "single", "several"],
    )

    args = parser.parse_args()

    if args.type == "scene":
        process_scene_files_in_folder(args.in_folder)
    elif args.type == "single":
        process_single_rendering_files_in_folder(args.in_folder)
    else:
        process_several_rendering_files_in_folder(args.in_folder)


if __name__ == "__main__":
    main()
