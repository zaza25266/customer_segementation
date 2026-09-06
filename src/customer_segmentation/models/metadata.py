
import json
from pathlib import Path


def save_model_metadata(
    metadata: dict,
    output_path: str | Path,
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(f"Metadata saved to: {output_path}")