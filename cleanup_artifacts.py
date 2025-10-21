#!/usr/bin/env python3
"""
Auto-cleanup script for AI4Pain Feature Extraction V2
Removes old log and result files to prevent disk bloat

Default thresholds:
- logs/: Delete files older than 7 days OR if folder exceeds 50MB
- results/: Delete files older than 30 days OR if folder exceeds 500MB
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import shutil


def get_folder_size(folder_path):
    """Calculate total size of folder in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB


def get_file_age_days(filepath):
    """Get age of file in days"""
    mtime = os.path.getmtime(filepath)
    file_time = datetime.fromtimestamp(mtime)
    age = datetime.now() - file_time
    return age.days


def cleanup_folder(folder_path, max_age_days, max_size_mb, dry_run=False, preserve_gitkeep=True):
    """
    Clean up old files in a folder based on age and size thresholds

    Args:
        folder_path: Path to folder to clean
        max_age_days: Maximum age of files in days
        max_size_mb: Maximum total folder size in MB
        dry_run: If True, only show what would be deleted
        preserve_gitkeep: If True, never delete .gitkeep files

    Returns:
        Tuple of (files_deleted, space_freed_mb)
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        print(f"📁 Folder not found: {folder_path}")
        return 0, 0.0

    # Get all files sorted by modification time (oldest first)
    all_files = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = Path(dirpath) / filename

            # Skip .gitkeep files if preserve_gitkeep is True
            if preserve_gitkeep and filename == '.gitkeep':
                continue

            if filepath.exists():
                all_files.append(filepath)

    all_files.sort(key=lambda f: os.path.getmtime(f))

    # Track statistics
    files_to_delete = []
    space_to_free = 0.0
    current_size = get_folder_size(folder_path)

    # Phase 1: Delete files older than max_age_days
    for filepath in all_files:
        age_days = get_file_age_days(filepath)
        if age_days > max_age_days:
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            files_to_delete.append((filepath, age_days, file_size_mb, "age"))
            space_to_free += file_size_mb
            current_size -= file_size_mb

    # Phase 2: If still over size limit, delete oldest files until under limit
    remaining_files = [f for f in all_files if not any(f == fd[0] for fd in files_to_delete)]
    remaining_files.sort(key=lambda f: os.path.getmtime(f))

    for filepath in remaining_files:
        if current_size <= max_size_mb:
            break

        age_days = get_file_age_days(filepath)
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        files_to_delete.append((filepath, age_days, file_size_mb, "size"))
        space_to_free += file_size_mb
        current_size -= file_size_mb

    # Execute deletions
    if files_to_delete:
        print(f"\n📂 {folder_path.name}/ cleanup:")
        print(f"   Current size: {get_folder_size(folder_path):.2f} MB")
        print(f"   Files to delete: {len(files_to_delete)}")
        print(f"   Space to free: {space_to_free:.2f} MB")

        if dry_run:
            print("\n   DRY RUN - Would delete:")

        for filepath, age_days, file_size_mb, reason in files_to_delete:
            rel_path = filepath.relative_to(folder_path.parent)
            if dry_run:
                print(f"   - {rel_path} ({age_days}d old, {file_size_mb:.2f} MB, reason: {reason})")
            else:
                try:
                    filepath.unlink()
                    print(f"   ✓ Deleted: {rel_path} ({age_days}d old, {file_size_mb:.2f} MB, reason: {reason})")
                except Exception as e:
                    print(f"   ✗ Error deleting {rel_path}: {e}")

        if not dry_run:
            final_size = get_folder_size(folder_path)
            print(f"   Final size: {final_size:.2f} MB")
    else:
        print(f"\n📂 {folder_path.name}/ - No cleanup needed")
        print(f"   Current size: {get_folder_size(folder_path):.2f} MB")

    return len(files_to_delete), space_to_free


def main():
    parser = argparse.ArgumentParser(
        description="Auto-cleanup script for logs and results folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview what would be deleted)
  python cleanup_artifacts.py --dry-run

  # Run with default thresholds
  python cleanup_artifacts.py

  # Custom thresholds
  python cleanup_artifacts.py --logs-max-age 3 --logs-max-size 25

  # Clean only logs folder
  python cleanup_artifacts.py --folder logs

  # Force cleanup without confirmation
  python cleanup_artifacts.py --force
        """
    )

    parser.add_argument(
        '--folder',
        choices=['logs', 'results', 'both'],
        default='both',
        help='Which folder(s) to clean (default: both)'
    )

    parser.add_argument(
        '--logs-max-age',
        type=int,
        default=7,
        help='Maximum age of log files in days (default: 7)'
    )

    parser.add_argument(
        '--logs-max-size',
        type=float,
        default=50.0,
        help='Maximum total size of logs folder in MB (default: 50)'
    )

    parser.add_argument(
        '--results-max-age',
        type=int,
        default=30,
        help='Maximum age of result files in days (default: 30)'
    )

    parser.add_argument(
        '--results-max-size',
        type=float,
        default=500.0,
        help='Maximum total size of results folder in MB (default: 500)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt'
    )

    args = parser.parse_args()

    # Get project root directory
    script_dir = Path(__file__).parent
    logs_dir = script_dir / 'logs'
    results_dir = script_dir / 'results'

    print("=" * 60)
    print("AI4Pain Auto-Cleanup Script")
    print("=" * 60)

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be deleted")

    print(f"\nConfiguration:")
    if args.folder in ['logs', 'both']:
        print(f"  logs/: max age = {args.logs_max_age} days, max size = {args.logs_max_size} MB")
    if args.folder in ['results', 'both']:
        print(f"  results/: max age = {args.results_max_age} days, max size = {args.results_max_size} MB")

    # Confirmation prompt (unless force or dry-run)
    if not args.force and not args.dry_run:
        print("\n⚠️  This will permanently delete files!")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return

    # Run cleanup
    total_deleted = 0
    total_freed = 0.0

    if args.folder in ['logs', 'both']:
        deleted, freed = cleanup_folder(
            logs_dir,
            args.logs_max_age,
            args.logs_max_size,
            dry_run=args.dry_run
        )
        total_deleted += deleted
        total_freed += freed

    if args.folder in ['results', 'both']:
        deleted, freed = cleanup_folder(
            results_dir,
            args.results_max_age,
            args.results_max_size,
            dry_run=args.dry_run
        )
        total_deleted += deleted
        total_freed += freed

    # Summary
    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"DRY RUN Summary: Would delete {total_deleted} files, freeing {total_freed:.2f} MB")
        print("\nRun without --dry-run to actually delete files")
    else:
        print(f"Cleanup complete: Deleted {total_deleted} files, freed {total_freed:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
