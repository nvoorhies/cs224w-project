"""Test script to parse all tweets in the dataset and identify failures."""

from pathlib import Path
from src.parsers import load_tweet, ParseError
import json
from collections import defaultdict
import traceback

def test_all_tweets():
    """Attempt to parse all tweet JSON files in the dataset."""

    data_dir = Path('data')

    # Find all tweet JSON files (both source-tweets and reactions)
    tweet_files = []
    tweet_files.extend(list(data_dir.rglob('source-tweets/*.json')))
    tweet_files.extend(list(data_dir.rglob('reactions/*.json')))

    print(f'Found {len(tweet_files)} tweet JSON files in the dataset')
    print('='*70)
    print()

    successes = 0
    failures = 0
    failure_details = []
    error_types = defaultdict(int)

    for i, tweet_file in enumerate(tweet_files, 1):
        if i % 100 == 0:
            print(f'Progress: {i}/{len(tweet_files)} tweets processed... ({successes} ✓, {failures} ✗)')

        try:
            tweet = load_tweet(tweet_file)
            successes += 1
        except Exception as e:
            failures += 1
            error_type = type(e).__name__
            error_types[error_type] += 1

            # Store details for first few failures of each type
            if len([f for f in failure_details if f['error_type'] == error_type]) < 3:
                failure_details.append({
                    'file': str(tweet_file.relative_to(data_dir)),
                    'error_type': error_type,
                    'error_msg': str(e),
                    'traceback': traceback.format_exc()
                })

    # Print summary
    print()
    print('='*70)
    print('RESULTS SUMMARY')
    print('='*70)
    print(f'Total tweets: {len(tweet_files)}')
    print(f'Successfully parsed: {successes} ({100*successes/len(tweet_files):.2f}%)')
    print(f'Failed to parse: {failures} ({100*failures/len(tweet_files):.2f}%)')
    print()

    if failures > 0:
        print('ERROR TYPES:')
        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f'  {error_type}: {count} occurrences')
        print()

        print('='*70)
        print('FAILURE EXAMPLES (first 3 of each type):')
        print('='*70)
        for i, failure in enumerate(failure_details, 1):
            print(f'\n[{i}] File: {failure["file"]}')
            print(f'    Error Type: {failure["error_type"]}')
            print(f'    Error Message: {failure["error_msg"][:200]}')

            # Try to load the raw JSON to see what's in it
            try:
                tweet_path = data_dir / failure['file']
                with open(tweet_path) as f:
                    raw_data = json.load(f)

                # Show top-level keys
                print(f'    JSON keys: {list(raw_data.keys())[:10]}')

                # Check for any unusual fields
                if 'id' in raw_data:
                    print(f'    Tweet ID: {raw_data["id"]}')
                if 'text' in raw_data:
                    print(f'    Text: {raw_data["text"][:50]}...')
            except:
                pass

            print(f'    Full traceback available on request')

        if len(failure_details) > 0:
            print()
            print('='*70)
            print('DETAILED TRACEBACK FOR FIRST FAILURE:')
            print('='*70)
            print(failure_details[0]['traceback'])
    else:
        print('✓ ALL TWEETS PARSED SUCCESSFULLY!')

    return successes, failures, failure_details

if __name__ == '__main__':
    successes, failures, failure_details = test_all_tweets()

    if failures > 0:
        exit(1)
    else:
        exit(0)
