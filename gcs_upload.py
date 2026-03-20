#!/usr/bin/env python3
"""
gcs_upload.py — Upload a file to private GCS bucket and return a signed URL.

Usage:
    python3 gcs_upload.py upload <local_file> [--key-file <path>] [--bucket <name>] [--expiry <hours>]
    python3 gcs_upload.py delete <object_name> [--key-file <path>] [--bucket <name>]

The signed URL is printed to stdout (one line). All other output goes to stderr.
The GCS object name is always: runpod-audio/<basename>.

Requirements:
    pip install google-cloud-storage
"""
import argparse
import datetime
import os
import sys
from pathlib import Path


DEFAULT_BUCKET = "whisper-files-jh"
DEFAULT_EXPIRY_HOURS = 2
GCS_PREFIX = "runpod-audio"


def _load_clients(key_file: str):
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
    except ImportError:
        print(
            "ERROR: google-cloud-storage not installed.\n"
            "  Run: pip install google-cloud-storage",
            file=sys.stderr,
        )
        sys.exit(1)

    credentials = service_account.Credentials.from_service_account_file(
        key_file,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = storage.Client(credentials=credentials, project=credentials.project_id)
    return client, credentials


def cmd_upload(args) -> None:
    local_path = Path(args.file)
    if not local_path.exists():
        print(f"ERROR: File not found: {local_path}", file=sys.stderr)
        sys.exit(1)

    key_file = _resolve_key_file(args.key_file)
    client, credentials = _load_clients(key_file)

    object_name = f"{GCS_PREFIX}/{local_path.name}"
    size_mb = local_path.stat().st_size / 1024 / 1024
    print(
        f"Uploading {size_mb:.1f}MB → gs://{args.bucket}/{object_name} ...",
        file=sys.stderr,
    )

    bucket = client.bucket(args.bucket)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(local_path))
    print("Upload complete.", file=sys.stderr)

    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(hours=args.expiry),
        method="GET",
        credentials=credentials,
    )
    # Only the URL goes to stdout — so the shell script can capture it cleanly
    print(signed_url)
    print(f"Signed URL expires in {args.expiry}h.", file=sys.stderr)
    print(f"Object name for cleanup: {object_name}", file=sys.stderr)


def cmd_delete(args) -> None:
    key_file = _resolve_key_file(args.key_file)
    client, _ = _load_clients(key_file)

    bucket = client.bucket(args.bucket)
    blob = bucket.blob(args.object_name)
    blob.delete()
    print(f"Deleted gs://{args.bucket}/{args.object_name}", file=sys.stderr)


def _resolve_key_file(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env:
        return env
    # Default: key sits at workspace root (one dir above the worker repo)
    script_dir = Path(__file__).parent
    default = script_dir.parent / "athefact-jhgallas-bec6b8fb5abd.json"
    if default.exists():
        return str(default)
    print(
        "ERROR: GCS key file not found.\n"
        "  Pass --key-file <path> or set GOOGLE_APPLICATION_CREDENTIALS.\n"
        f"  Looked for: {default}",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # upload subcommand
    up = sub.add_parser("upload", help="Upload a file and return a signed URL")
    up.add_argument("file", help="Local file path to upload")
    up.add_argument("--key-file", default=None, help="Path to GCP service account JSON key")
    up.add_argument("--bucket", default=DEFAULT_BUCKET, help="GCS bucket name")
    up.add_argument("--expiry", type=int, default=DEFAULT_EXPIRY_HOURS, help="Signed URL expiry in hours (default: 2)")

    # delete subcommand
    dl = sub.add_parser("delete", help="Delete a GCS object")
    dl.add_argument("object_name", help="GCS object name, e.g. runpod-audio/file.opus")
    dl.add_argument("--key-file", default=None)
    dl.add_argument("--bucket", default=DEFAULT_BUCKET)

    args = parser.parse_args()
    if args.command == "upload":
        cmd_upload(args)
    elif args.command == "delete":
        cmd_delete(args)


if __name__ == "__main__":
    main()
