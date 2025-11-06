#!/usr/bin/env python3
"""
Dataset verification and exploration script for PHEME dataset.

This script loads and validates the PHEME dataset, providing:
- Statistics about the dataset
- Verification of data integrity
- Sample data exploration
- Summary reports

Usage:
    python verify_dataset.py [--max-threads N] [--story STORY_NAME]
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import List

from src import load_dataset, find_all_threads, verify_thread, TweetThread


def print_separator(title: str = ""):
    """Print a separator line with optional title."""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}\n")
    else:
        print(f"{'=' * 80}")


def explore_dataset_structure(data_root: Path):
    """Explore and print the high-level structure of the dataset."""
    print_separator("Dataset Structure")

    data_root = Path(data_root)
    print(f"Dataset root: {data_root}\n")

    # Find all story directories
    story_dirs = [d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith('.')]

    print(f"Found {len(story_dirs)} story directories:\n")

    for story_dir in sorted(story_dirs):
        # Count threads in each category
        thread_counts = defaultdict(int)

        for category_dir in story_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('.'):
                threads = list(category_dir.glob("*/annotation.json"))
                thread_counts[category_dir.name] = len(threads)

        total_threads = sum(thread_counts.values())
        print(f"  {story_dir.name}: {total_threads} threads")

        for category, count in sorted(thread_counts.items()):
            print(f"    - {category}: {count}")

    print()


def analyze_threads(threads: List[TweetThread]):
    """Analyze and print statistics about loaded threads."""
    print_separator("Thread Statistics")

    print(f"Total threads loaded: {len(threads)}\n")

    if not threads:
        print("No threads to analyze.")
        return

    # Rumour type distribution
    rumour_types = Counter(t.annotation.is_rumour for t in threads)
    print("Rumour type distribution:")
    for rumour_type, count in rumour_types.most_common():
        percentage = (count / len(threads)) * 100
        print(f"  {rumour_type:15s}: {count:4d} ({percentage:5.1f}%)")

    # Misinformation and truth distribution
    print("\nMisinformation distribution:")
    misinformation_counts = Counter(t.annotation.misinformation for t in threads)
    for value, count in sorted(misinformation_counts.items()):
        percentage = (count / len(threads)) * 100
        label = "Misinformation" if value == 1 else "Not misinformation"
        print(f"  {label:20s}: {count:4d} ({percentage:5.1f}%)")

    print("\nTruth value distribution:")
    true_counts = Counter(t.annotation.true for t in threads)
    for value, count in sorted(true_counts.items()):
        percentage = (count / len(threads)) * 100
        label = "True" if value == 1 else "False"
        print(f"  {label:20s}: {count:4d} ({percentage:5.1f}%)")

    # Tweet and user statistics
    total_tweets = sum(len(t.get_all_tweets()) for t in threads)
    total_reactions = sum(len(t.reaction_tweets) for t in threads)
    total_users = len(set(user_id for t in threads for user_id in t.get_all_users().keys()))
    total_edges = sum(len(t.structure.get_reply_edges()) for t in threads)

    print(f"\nContent statistics:")
    print(f"  Total tweets:       {total_tweets:6d}")
    print(f"  Total reactions:    {total_reactions:6d}")
    print(f"  Total unique users: {total_users:6d}")
    print(f"  Total reply edges:  {total_edges:6d}")

    # Average statistics
    avg_tweets = total_tweets / len(threads)
    avg_reactions = total_reactions / len(threads)
    avg_edges = total_edges / len(threads)

    print(f"\nAverage per thread:")
    print(f"  Tweets:    {avg_tweets:6.1f}")
    print(f"  Reactions: {avg_reactions:6.1f}")
    print(f"  Edges:     {avg_edges:6.1f}")

    # Find threads with most activity
    threads_by_tweets = sorted(threads, key=lambda t: len(t.get_all_tweets()), reverse=True)
    threads_by_users = sorted(threads, key=lambda t: len(t.get_all_users()), reverse=True)

    print(f"\nTop 5 threads by tweet count:")
    for i, thread in enumerate(threads_by_tweets[:5], 1):
        num_tweets = len(thread.get_all_tweets())
        print(f"  {i}. {thread.thread_id}: {num_tweets} tweets")

    print(f"\nTop 5 threads by unique users:")
    for i, thread in enumerate(threads_by_users[:5], 1):
        num_users = len(thread.get_all_users())
        print(f"  {i}. {thread.thread_id}: {num_users} users")

    print()


def verify_threads(threads: List[TweetThread]):
    """Verify thread integrity and report issues."""
    print_separator("Thread Verification")

    issues_found = 0
    threads_with_warnings = 0

    for thread in threads:
        results = verify_thread(thread)

        if results["errors"]:
            issues_found += 1
            print(f"❌ Thread {thread.thread_id} has ERRORS:")
            for error in results["errors"]:
                print(f"   - {error}")

        if results["warnings"]:
            threads_with_warnings += 1
            if issues_found < 5:  # Only show first few
                print(f"⚠️  Thread {thread.thread_id} has warnings:")
                for warning in results["warnings"]:
                    print(f"   - {warning}")

    print(f"\nVerification summary:")
    print(f"  Threads with errors:   {issues_found}")
    print(f"  Threads with warnings: {threads_with_warnings}")
    print(f"  Clean threads:         {len(threads) - threads_with_warnings - issues_found}")

    if issues_found == 0 and threads_with_warnings == 0:
        print("\n✅ All threads verified successfully!")
    elif issues_found == 0:
        print("\n✅ No critical errors found (warnings are acceptable)")
    else:
        print("\n⚠️  Some threads have issues")

    print()


def show_sample_thread(threads: List[TweetThread], thread_id: str = None):
    """Display detailed information about a sample thread."""
    print_separator("Sample Thread Details")

    if not threads:
        print("No threads available.")
        return

    # Select thread to display
    if thread_id:
        thread = next((t for t in threads if t.thread_id == thread_id), None)
        if not thread:
            print(f"Thread {thread_id} not found. Showing first thread instead.")
            thread = threads[0]
    else:
        thread = threads[0]

    print(f"Thread ID: {thread.thread_id}\n")

    # Annotation
    print("Annotation:")
    print(f"  Rumour type:      {thread.annotation.is_rumour}")
    print(f"  Category:         {thread.annotation.category}")
    print(f"  Misinformation:   {thread.annotation.misinformation}")
    print(f"  True:             {thread.annotation.true}")
    print(f"  Is turnaround:    {thread.annotation.is_turnaround}")
    print(f"  Number of links:  {len(thread.annotation.links)}")

    if thread.annotation.links:
        print(f"\n  Links:")
        for i, link in enumerate(thread.annotation.links[:3], 1):
            print(f"    {i}. {link.mediatype} ({link.position})")
            print(f"       {link.link[:80]}...")

    # Source tweet
    print(f"\nSource Tweet:")
    print(f"  ID:           {thread.source_tweet.id}")
    print(f"  Text:         {thread.source_tweet.text[:100]}...")
    print(f"  User:         @{thread.source_tweet.user.screen_name}")
    print(f"  Retweets:     {thread.source_tweet.retweet_count}")
    print(f"  Favorites:    {thread.source_tweet.favorite_count}")
    print(f"  Created at:   {thread.source_tweet.created_at}")

    # Structure
    edges = thread.structure.get_reply_edges()
    all_tweets = thread.get_all_tweets()
    all_users = thread.get_all_users()

    print(f"\nThread Structure:")
    print(f"  Total tweets:      {len(all_tweets)}")
    print(f"  Reaction tweets:   {len(thread.reaction_tweets)}")
    print(f"  Unique users:      {len(all_users)}")
    print(f"  Reply edges:       {len(edges)}")

    if edges:
        print(f"\n  Sample reply edges (first 5):")
        for i, (parent, child) in enumerate(edges[:5], 1):
            parent_text = all_tweets.get(parent, None)
            child_text = all_tweets.get(child, None)
            parent_preview = parent_text.text[:30] if parent_text else "N/A"
            child_preview = child_text.text[:30] if child_text else "N/A"
            print(f"    {i}. {parent} → {child}")
            print(f"       \"{parent_preview}...\" → \"{child_preview}...\"")

    # User info
    print(f"\n  Top users by follower count:")
    sorted_users = sorted(all_users.values(), key=lambda u: u.followers_count, reverse=True)
    for i, user in enumerate(sorted_users[:5], 1):
        print(f"    {i}. @{user.screen_name:20s} - {user.followers_count:,} followers")

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify and explore the PHEME dataset"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/all-rnr-annotated-threads"),
        help="Path to the dataset root directory"
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="Maximum number of threads to load (default: all)"
    )
    parser.add_argument(
        "--story",
        type=str,
        default=None,
        help="Specific story to analyze (e.g., 'germanwings-crash-all-rnr-threads')"
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Specific thread ID to display in detail"
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip thread verification step"
    )

    args = parser.parse_args()

    # Determine dataset root
    data_root = args.data_root
    if args.story:
        data_root = data_root / args.story

    if not data_root.exists():
        print(f"Error: Dataset root not found: {data_root}")
        return 1

    # Explore structure
    explore_dataset_structure(args.data_root)

    # Load threads
    print_separator("Loading Dataset")
    print(f"Loading threads from: {data_root}")
    if args.max_threads:
        print(f"Limiting to: {args.max_threads} threads")
    print()

    threads = load_dataset(
        data_root,
        load_reactions=True,
        max_threads=args.max_threads,
        skip_errors=True
    )

    if not threads:
        print("No threads loaded. Exiting.")
        return 1

    # Analyze threads
    analyze_threads(threads)

    # Verify threads
    if not args.skip_verification:
        verify_threads(threads)

    # Show sample thread
    show_sample_thread(threads, thread_id=args.thread_id)

    print_separator()
    print("Dataset verification complete!")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
