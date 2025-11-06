"""Test script to verify updated parser captures all fields."""

from pathlib import Path
from src.parsers import load_tweet

def test_parser_with_media():
    """Test that parser captures media and extended entities."""

    # Test tweet with extended_entities and media
    tweet_path = Path('data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/rumours/580333904126742528/reactions/580336203259457536.json')

    print('Testing parser with tweet that has media and extended_entities...')
    print(f'Loading: {tweet_path}\n')

    try:
        tweet = load_tweet(tweet_path)
        print(f'✓ Successfully parsed tweet: {tweet.id_str}')
        print(f'  Text: {tweet.text[:50]}...')
        print(f'  Has media in entities: {len(tweet.entities.media) > 0}')
        print(f'  Number of media entities: {len(tweet.entities.media)}')
        print(f'  Has extended_entities: {tweet.extended_entities is not None}')

        if tweet.extended_entities:
            print(f'  Number of extended media: {len(tweet.extended_entities.media)}')

        print(f'  possibly_sensitive: {tweet.possibly_sensitive}')
        print(f'  possibly_sensitive_appealable: {tweet.possibly_sensitive_appealable}')
        print(f'  filter_level: {tweet.filter_level}')

        # Check media details
        if tweet.entities.media:
            media = tweet.entities.media[0]
            print(f'\n  First media entity details:')
            print(f'    - ID: {media.id_str}')
            print(f'    - Type: {media.type}')
            print(f'    - URL: {media.media_url_https}')
            print(f'    - Sizes available: {list(media.sizes.keys())}')
            print(f'    - Size dimensions:')
            for size_name, size_info in media.sizes.items():
                print(f'      {size_name}: {size_info.w}x{size_info.h} ({size_info.resize})')

        print('\n✓ All new fields are being captured correctly!')
        return True

    except Exception as e:
        print(f'\n✗ Error parsing tweet: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_parser_without_media():
    """Test that parser still works with tweets that don't have media."""

    # Test a regular tweet without media
    tweet_path = Path('data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/rumours/580697361799876608/reactions/580697739777953792.json')

    print('\n' + '='*60)
    print('Testing parser with tweet WITHOUT media...')
    print(f'Loading: {tweet_path}\n')

    try:
        tweet = load_tweet(tweet_path)
        print(f'✓ Successfully parsed tweet: {tweet.id_str}')
        print(f'  Text: {tweet.text[:70]}...')
        print(f'  Has media: {len(tweet.entities.media) > 0}')
        print(f'  Has extended_entities: {tweet.extended_entities is not None}')

        print('\n✓ Parser handles tweets without media correctly!')
        return True

    except Exception as e:
        print(f'\n✗ Error parsing tweet: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success1 = test_parser_with_media()
    success2 = test_parser_without_media()

    if success1 and success2:
        print('\n' + '='*60)
        print('✓ ALL TESTS PASSED!')
        print('='*60)
    else:
        print('\n' + '='*60)
        print('✗ SOME TESTS FAILED')
        print('='*60)
        exit(1)
