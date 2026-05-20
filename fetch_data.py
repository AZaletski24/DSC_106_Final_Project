"""
ABS (Automated Ball-Strike) Data Fetcher
Pulls data from Baseball Savant for the 2026 MLB season.
Run this script to refresh the data before generating visualizations.
"""

import requests
import io
import pandas as pd
import time
import os

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://baseballsavant.mlb.com/'
}

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def fetch_abs_leaderboard(entity_type='batter', year=2026):
    """Fetch ABS challenge leaderboard from Baseball Savant."""
    url = (
        f'https://baseballsavant.mlb.com/leaderboard/abs-challenges'
        f'?year={year}&gameType=regular&level=mlb&type={entity_type}&csv=true'
    )
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df[df['level'] == 'MLB'].copy()


def fetch_statcast_by_date_range(start, end):
    """Fetch statcast pitch-by-pitch data for a date range."""
    url = (
        f'https://baseballsavant.mlb.com/statcast_search/csv'
        f'?all=true&game_date_gt={start}&game_date_lt={end}&type=details'
    )
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def fetch_full_season_statcast(year=2026):
    """Fetch the full season statcast data in monthly chunks."""
    ranges = [
        (f'{year}-03-27', f'{year}-03-31'),
        (f'{year}-04-01', f'{year}-04-30'),
        (f'{year}-05-01', f'{year}-05-18'),
    ]
    dfs = []
    for start, end in ranges:
        print(f'  Fetching {start} → {end}...')
        df = fetch_statcast_by_date_range(start, end)
        print(f'    Got {len(df):,} pitches')
        dfs.append(df)
        time.sleep(1)
    return pd.concat(dfs, ignore_index=True)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print('Fetching batter ABS leaderboard...')
    df_bat = fetch_abs_leaderboard('batter')
    df_bat.to_csv(f'{DATA_DIR}/abs_batter_leaderboard.csv', index=False)
    print(f'  Saved {len(df_bat)} batters')

    print('Fetching pitcher ABS leaderboard...')
    df_pit = fetch_abs_leaderboard('pitcher')
    df_pit.to_csv(f'{DATA_DIR}/abs_pitcher_leaderboard.csv', index=False)
    print(f'  Saved {len(df_pit)} pitchers')

    print('Fetching full season statcast pitch data...')
    df_pitches = fetch_full_season_statcast()
    df_pitches.to_csv(f'{DATA_DIR}/statcast_2026.csv', index=False)
    print(f'  Saved {len(df_pitches):,} total pitches')

    print('\nAll data saved to data/')


if __name__ == '__main__':
    main()
