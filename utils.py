from pathlib import Path


def clean_files(directory: str) -> None:
    for path in Path(directory).rglob("*.py"):
        print(f"Cleaning {path.name}")
        with open(path, 'r') as f:
            res = ""
            lines = f.readlines()
            for line in lines:
                if "{%" in line or '{{' in line:
                    continue

                res += line

            with open(path, 'w') as w:
                w.write(res)
