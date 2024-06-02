import os
from typing import List, Dict


def run_exact_line_deduplication(
    file_paths: List[os.PathLike], output_dir: os.PathLike
) -> None:
    line_counts: Dict[int, int] = {}
    for file_path in file_paths:
        with open(file_path, "r") as f:
            for line in f:
                line_hash = hash(line)
                line_counts[line_hash] = line_counts.get(line_hash, 0) + 1

    for file_path in file_paths:
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        with open(file_path, "r") as f, open(output_path, "w") as output_file:
            seen_lines = set()
            for line in f:
                line_hash = hash(line)
                if line_counts[line_hash] == 1 and line_hash not in seen_lines:
                    output_file.write(line)
                    seen_lines.add(line_hash)
