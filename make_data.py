"""
Pre-processes raw CSVs into small JSON files the browser can load directly.
Outputs:
  data/zone_heatmap.json  — binned pitch density for the strike zone heatmap
  data/game_pitches.json  — 500 sampled pitches with physics for the calling game
  data/team_stats.json    — aggregated team challenge stats
"""

import pandas as pd
import numpy as np
import json, os, requests, io

DATA = os.path.join(os.path.dirname(__file__), 'data')
HEADERS = {'User-Agent': 'Mozilla/5.0'}


# ── helpers ──────────────────────────────────────────────────────────────────
def load_statcast():
    path = f'{DATA}/statcast_2026.csv'
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    print('Fetching statcast data…')
    ranges = [('2026-03-27','2026-03-31'),('2026-04-01','2026-04-30'),('2026-05-01','2026-05-18')]
    dfs = []
    for s, e in ranges:
        r = requests.get(
            f'https://baseballsavant.mlb.com/statcast_search/csv?all=true&game_date_gt={s}&game_date_lt={e}&type=details',
            headers=HEADERS, timeout=120)
        dfs.append(pd.read_csv(io.StringIO(r.text)))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(path, index=False)
    return df

def load_batter_lb():
    path = f'{DATA}/abs_batter_leaderboard.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    r = requests.get(
        'https://baseballsavant.mlb.com/leaderboard/abs-challenges?year=2026&gameType=regular&level=mlb&type=batter&csv=true',
        headers=HEADERS, timeout=30)
    df = pd.read_csv(io.StringIO(r.text))
    df = df[df['level']=='MLB'].copy()
    df.to_csv(path, index=False)
    return df


# ── ABS geometric call ───────────────────────────────────────────────────────
def abs_call(row):
    """True = strike by ABS zone geometry."""
    if pd.isna(row['plate_x']) or pd.isna(row['plate_z']):
        return np.nan
    return (abs(row['plate_x']) <= 0.708 and
            row['sz_bot'] <= row['plate_z'] <= row['sz_top'])


# ── 1. Zone heatmap data ─────────────────────────────────────────────────────
def make_zone_heatmap(df):
    called = df[df['description'].isin(['called_strike','ball'])].copy()
    called = called.dropna(subset=['plate_x','plate_z','sz_top','sz_bot','stand'])
    called = called[called['plate_x'].between(-2.5, 2.5) & called['plate_z'].between(0.2, 5.5)]

    called['is_strike_ump'] = called['description'] == 'called_strike'
    called['is_strike_abs'] = called.apply(abs_call, axis=1)
    called['ump_error']     = called['is_strike_ump'] != called['is_strike_abs']

    # 30×30 grid
    X_BINS = 30; Z_BINS = 30
    x_edges = np.linspace(-2.5, 2.5, X_BINS + 1)
    z_edges = np.linspace(0.2,  5.5, Z_BINS + 1)

    records = []
    for stand in ['R', 'L', 'both']:
        sub = called if stand == 'both' else called[called['stand'] == stand]
        if len(sub) == 0:
            continue
        for xi in range(X_BINS):
            for zi in range(Z_BINS):
                mask = (
                    (sub['plate_x'] >= x_edges[xi])  & (sub['plate_x'] < x_edges[xi+1]) &
                    (sub['plate_z'] >= z_edges[zi])  & (sub['plate_z'] < z_edges[zi+1])
                )
                cell = sub[mask]
                if len(cell) == 0:
                    continue
                records.append({
                    'stand':       stand,
                    'xi':          xi,
                    'zi':          zi,
                    'x_center':    round(float((x_edges[xi] + x_edges[xi+1]) / 2), 3),
                    'z_center':    round(float((z_edges[zi] + z_edges[zi+1]) / 2), 3),
                    'n_called_strike': int(cell['is_strike_ump'].sum()),
                    'n_ball':          int((~cell['is_strike_ump']).sum()),
                    'n_total':         int(len(cell)),
                    'error_rate':      round(float(cell['ump_error'].mean()), 4),
                    'strike_rate':     round(float(cell['is_strike_ump'].mean()), 4),
                })

    # Overall umpire accuracy stats
    valid = called.dropna(subset=['is_strike_abs'])
    ump_overall  = float((valid['is_strike_ump'] == valid['is_strike_abs']).mean())
    strikes_only = valid[valid['is_strike_ump']]
    balls_only   = valid[~valid['is_strike_ump']]
    ump_strike_acc = float((strikes_only['is_strike_abs'] == True).mean())  if len(strikes_only) else 0
    ump_ball_acc   = float((balls_only['is_strike_abs'] == False).mean())   if len(balls_only)   else 0

    out = {
        'x_edges': [round(v,3) for v in x_edges.tolist()],
        'z_edges': [round(v,3) for v in z_edges.tolist()],
        'cells': records,
        'umpire_stats': {
            'overall':  round(ump_overall, 4),
            'strike':   round(ump_strike_acc, 4),
            'ball':     round(ump_ball_acc, 4),
            'n_pitches': int(len(valid)),
        }
    }
    with open(f'{DATA}/zone_heatmap.json', 'w') as f:
        json.dump(out, f)
    print(f'  zone_heatmap.json  → {len(records)} cells, ump accuracy {ump_overall:.3f}')
    return ump_overall, ump_strike_acc, ump_ball_acc


# ── 2. Game pitches ───────────────────────────────────────────────────────────
def make_game_pitches(df, ump_overall, ump_strike_acc, ump_ball_acc):
    physics_cols = ['vx0','vy0','vz0','ax','ay','az',
                    'release_pos_x','release_pos_z','release_pos_y']
    needed = ['plate_x','plate_z','sz_top','sz_bot','stand',
              'pitch_type','pitch_name','release_speed',
              'description','player_name','pfx_x','pfx_z'] + physics_cols

    called = df[df['description'].isin(['called_strike','ball'])].copy()
    called = called.dropna(subset=needed)
    called = called[called['plate_x'].between(-2.5, 2.5) & called['plate_z'].between(0.3, 5.5)]
    called = called[called['release_speed'] > 60]  # exclude eephus junk
    called['abs_strike'] = called.apply(abs_call, axis=1)
    called = called.dropna(subset=['abs_strike'])

    # Signed distance to nearest zone boundary (negative = inside, positive = outside)
    def bdist(r):
        dx  = abs(r['plate_x']) - 0.708
        dz1 = r['sz_bot'] - r['plate_z']
        dz2 = r['plate_z'] - r['sz_top']
        return max(dx, dz1, dz2)

    called['bdist'] = called.apply(bdist, axis=1)
    # Edge = within 0.3 ft of boundary on either side
    called['is_edge'] = called['bdist'].abs() < 0.3

    abs_strikes = called[called['abs_strike'] == True]
    abs_balls   = called[called['abs_strike'] == False]

    edge_str  = abs_strikes[abs_strikes['is_edge']]
    clear_str = abs_strikes[~abs_strikes['is_edge']]
    edge_ball = abs_balls[abs_balls['is_edge']]
    clear_ball= abs_balls[~abs_balls['is_edge']]

    n = 125  # per bucket (edge-strike, clear-strike, edge-ball, clear-ball)
    sampled = pd.concat([
        edge_str.sample( n=min(n, len(edge_str)),   random_state=42),
        clear_str.sample(n=min(n, len(clear_str)),  random_state=42),
        edge_ball.sample(n=min(n, len(edge_ball)),  random_state=42),
        clear_ball.sample(n=min(n, len(clear_ball)),random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    pitches = []
    for _, row in sampled.iterrows():
        pitches.append({
            'id':          int(row.name),
            'pitch_name':  str(row.get('pitch_name','Unknown') or 'Unknown'),
            'speed':       round(float(row['release_speed']), 1),
            'plate_x':     round(float(row['plate_x']), 4),
            'plate_z':     round(float(row['plate_z']), 4),
            'sz_top':      round(float(row['sz_top']), 3),
            'sz_bot':      round(float(row['sz_bot']), 3),
            'abs_call':    'strike' if row['abs_strike'] else 'ball',
            'ump_call':    'strike' if row['description'] == 'called_strike' else 'ball',
            'stand':       str(row['stand']),
            'pfx_x':       round(float(row['pfx_x']), 3),
            'pfx_z':       round(float(row['pfx_z']), 3),
            # physics
            'vx0':         round(float(row['vx0']), 3),
            'vy0':         round(float(row['vy0']), 3),
            'vz0':         round(float(row['vz0']), 3),
            'ax':          round(float(row['ax']), 3),
            'ay':          round(float(row['ay']), 3),
            'az':          round(float(row['az']), 3),
            'rel_x':       round(float(row['release_pos_x']), 3),
            'rel_z':       round(float(row['release_pos_z']), 3),
            'rel_y':       round(float(row['release_pos_y']), 3),
            'dist_edge':   round(float(row['bdist']), 4),
        })

    out = {
        'pitches': pitches,
        'umpire_stats': {
            'overall': round(ump_overall, 4),
            'strike':  round(ump_strike_acc, 4),
            'ball':    round(ump_ball_acc, 4),
        }
    }
    with open(f'{DATA}/game_pitches.json', 'w') as f:
        json.dump(out, f)
    print(f'  game_pitches.json  → {len(pitches)} pitches '
          f'({sum(1 for p in pitches if p["abs_call"]=="strike")} strikes, '
          f'{sum(1 for p in pitches if p["abs_call"]=="ball")} balls)')


# ── 3. Team stats ─────────────────────────────────────────────────────────────
def make_team_stats(df_bat):
    mlb = df_bat[df_bat['level'] == 'MLB'] if 'level' in df_bat.columns else df_bat
    team = mlb.groupby('team_abbr').agg(
        n_challenges     =('n_challenges',      'sum'),
        n_overturns      =('n_overturns',       'sum'),
        k_flipped        =('n_strikeouts_flip',  'sum'),
        bb_flipped       =('n_walks_flip',       'sum'),
        n_challenges_vs  =('n_challenges_against','sum'),
        n_overturns_vs   =('n_overturns_against', 'sum'),
    ).reset_index()
    team['rate'] = (team['n_overturns'] / team['n_challenges'].replace(0, np.nan)).round(4)
    team['rate_vs'] = (team['n_overturns_vs'] / team['n_challenges_vs'].replace(0, np.nan)).round(4)
    team['net_flip'] = team['bb_flipped'] - team['k_flipped']
    team = team.sort_values('rate', ascending=False)
    records = team.where(pd.notna(team), None).to_dict(orient='records')
    with open(f'{DATA}/team_stats.json', 'w') as f:
        json.dump(records, f)
    print(f'  team_stats.json    → {len(records)} teams')


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(DATA, exist_ok=True)
    print('Loading raw data…')
    df_pit = load_statcast()
    df_bat = load_batter_lb()
    print(f'  {len(df_pit):,} pitches  |  {len(df_bat)} batters')

    print('\nBuilding zone_heatmap.json…')
    ump_ov, ump_str, ump_ball = make_zone_heatmap(df_pit)

    print('\nBuilding game_pitches.json…')
    make_game_pitches(df_pit, ump_ov, ump_str, ump_ball)

    print('\nBuilding team_stats.json…')
    make_team_stats(df_bat)

    print('\nDone — all data files in data/')
