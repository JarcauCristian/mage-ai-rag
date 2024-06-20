from pathlib import Path


def load_transformer() -> list[str]:
    files = []
    for path in Path("testing_blocks/transformers").glob("*.py"):
        with open(path) as f:
            files.append(f.read())

    return files
