from pathlib import Path

from pydantic import BaseModel, FilePath


class Song(BaseModel):
    audio_path: FilePath
    annotation_path: FilePath
    name: str


def get_files(input_file: Path | str, base_path: Path | str) -> list[Path]:
    """Read a text file of filenames and return resolved paths."""
    base = Path(base_path)
    text = Path(input_file).read_text().strip()
    return [base / name for name in text.splitlines() if name]


def name_from_path(audio_path: Path | str) -> str:
    """Extract a song name from an audio file path.

    Expects the format 'Artist - Song Title.wav'; falls back to the file stem.
    """
    stem = Path(audio_path).stem
    if ' - ' in stem:
        return stem.split(' - ', 1)[1]
    return stem


def songs_from_files(audio_list: list[Path], label_list: list[Path]) -> list[Song]:
    """Build a list of Song models from matching audio and annotation paths."""
    return [
        Song(
            audio_path=audio,
            annotation_path=label,
            name=name_from_path(audio),
        )
        for audio, label in zip(audio_list, label_list)
    ]


def txt_to_csv(input_path: Path | str, output_path: Path | str) -> None:
    """Convert a tab-separated annotation TXT file to semicolon-separated CSV."""
    lines = Path(input_path).read_text().strip().splitlines()
    csv_lines = ['"Start";"End";"Label"']
    for line in lines:
        parts = line.split('\t')
        if len(parts) == 3:
            start, end, label = parts
            csv_lines.append(f'{start};{end};"{label}"')
    Path(output_path).write_text('\n'.join(csv_lines))


def txt_to_csv_aligned(input_path: Path | str, output_path: Path | str) -> None:
    """Convert annotation TXT to CSV, aligning each segment's start time to
    the previous segment's end time to ensure contiguous boundaries."""
    lines = Path(input_path).read_text().strip().splitlines()
    adjusted = []
    for i, line in enumerate(lines):
        parts = line.split('\t')
        parts[0] = '0.000000' if i == 0 else lines[i - 1].split('\t')[1]
        adjusted.append('\t'.join(parts))

    csv_lines = ['"Start";"End";"Label"']
    for line in adjusted:
        start, end, label = line.split('\t')
        csv_lines.append(f'{start};{end};"{label}"')
    Path(output_path).write_text('\n'.join(csv_lines))
