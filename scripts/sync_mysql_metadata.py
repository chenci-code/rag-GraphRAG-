from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mysql_metadata import sync_graphrag_metadata_to_mysql


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync GraphRAG documents/text units metadata into MySQL."
    )
    parser.add_argument("--root", default=".", help="GraphRAG project root.")
    parser.add_argument(
        "--default-tenant-id",
        default="default",
        help="Tenant assigned when no override is provided.",
    )
    parser.add_argument(
        "--default-department-id",
        default="default",
        help="Department assigned when no override is provided.",
    )
    parser.add_argument(
        "--default-visibility",
        choices=["public", "department", "private"],
        default="department",
        help="Visibility assigned when no override is provided.",
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="Optional JSON file mapping documents/text units to departments.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    load_dotenv(root / ".env", override=False)
    stats = sync_graphrag_metadata_to_mysql(
        root=root,
        default_tenant_id=args.default_tenant_id,
        default_department_id=args.default_department_id,
        default_visibility=args.default_visibility,
        overrides_path=Path(args.overrides).resolve() if args.overrides else None,
    )
    print(stats)


if __name__ == "__main__":
    main()
