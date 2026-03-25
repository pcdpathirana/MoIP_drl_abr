from __future__ import annotations

from pathlib import Path



def main() -> None:
    root = Path("data/public_sources")
    for sub in ["fcc_raw", "norway_raw", "pensieve_reference", "converted"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

    guide = root / "README.md"
    guide.write_text(
        """# Public Trace Workspace

"
        "Place downloaded public datasets here before conversion:
"
        "- `fcc_raw/`: FCC Measuring Broadband America raw release archives
"
        "- `norway_raw/`: Norway HSDPA mobile bandwidth logs
"
        "- `pensieve_reference/`: optional Pensieve repository or sample cooked traces

"
        "Then use the project scripts and docs in `docs/DATASET_GUIDE.md` to prepare training-ready CSV files.
"
        """,
        encoding="utf-8",
    )
    print(f"Prepared public-data workspace at {root.resolve()}")


if __name__ == "__main__":
    main()
