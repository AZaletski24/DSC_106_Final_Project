"""
ABS Visualization Generator — 5 static figures for DSC 106 Final Project Proposal
All figures saved to figures/ directory as high-res PNGs.
"""

import requests
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── Style ─────────────────────────────────────────────────────────────────────
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FIG_DIR  = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

DARK_BG   = '#0f1117'
PANEL_BG  = '#1a1d26'
TEXT_COL  = '#e8eaf0'
ACCENT    = '#e63946'
ACCENT2   = '#457b9d'
GREEN     = '#2a9d8f'
YELLOW    = '#e9c46a'

plt.rcParams.update({
    'figure.facecolor':  DARK_BG,
    'axes.facecolor':    PANEL_BG,
    'axes.edgecolor':    '#333645',
    'axes.labelcolor':   TEXT_COL,
    'xtick.color':       TEXT_COL,
    'ytick.color':       TEXT_COL,
    'text.color':        TEXT_COL,
    'grid.color':        '#2a2d3a',
    'grid.linewidth':    0.6,
    'font.family':       'DejaVu Sans',
    'axes.titlecolor':   TEXT_COL,
    'axes.titlesize':    14,
    'axes.labelsize':    11,
})

def savefig(name):
    path = f'{FIG_DIR}/{name}'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f'  Saved → {path}')


# ── Load Data ──────────────────────────────────────────────────────────────────
def load_data():
    batter_path  = f'{DATA_DIR}/abs_batter_leaderboard.csv'
    pitcher_path = f'{DATA_DIR}/abs_pitcher_leaderboard.csv'
    pitch_path   = f'{DATA_DIR}/statcast_2026.csv'

    def fetch_leaderboard(entity_type):
        url = (
            f'https://baseballsavant.mlb.com/leaderboard/abs-challenges'
            f'?year=2026&gameType=regular&level=mlb&type={entity_type}&csv=true'
        )
        r = requests.get(url, headers=HEADERS, timeout=30)
        df = pd.read_csv(io.StringIO(r.text))
        return df[df['level'] == 'MLB'].copy()

    def fetch_pitches():
        ranges = [('2026-03-27','2026-03-31'),('2026-04-01','2026-04-30'),('2026-05-01','2026-05-18')]
        dfs = []
        for s, e in ranges:
            url = f'https://baseballsavant.mlb.com/statcast_search/csv?all=true&game_date_gt={s}&game_date_lt={e}&type=details'
            r = requests.get(url, headers=HEADERS, timeout=120)
            dfs.append(pd.read_csv(io.StringIO(r.text)))
        return pd.concat(dfs, ignore_index=True)

    if os.path.exists(batter_path):
        df_bat = pd.read_csv(batter_path)
    else:
        print('Fetching batter leaderboard...')
        df_bat = fetch_leaderboard('batter')
        df_bat.to_csv(batter_path, index=False)

    if os.path.exists(pitcher_path):
        df_pit = pd.read_csv(pitcher_path)
    else:
        print('Fetching pitcher leaderboard...')
        df_pit = fetch_leaderboard('pitcher')
        df_pit.to_csv(pitcher_path, index=False)

    if os.path.exists(pitch_path):
        df_pitches = pd.read_csv(pitch_path, low_memory=False)
    else:
        print('Fetching statcast pitches...')
        df_pitches = fetch_pitches()
        df_pitches.to_csv(pitch_path, index=False)

    return df_bat, df_pit, df_pitches


# ── Figure 1: Team Challenge Success Rate ─────────────────────────────────────
def fig_team_challenge_rate(df_bat):
    print('Building Figure 1: Team challenge success rates...')

    team = df_bat.groupby('team_abbr').agg(
        total_challenges=('n_challenges', 'sum'),
        total_overturns=('n_overturns', 'sum'),
        k_flipped=('n_strikeouts_flip', 'sum'),
        bb_flipped=('n_walks_flip', 'sum'),
    ).reset_index()
    team['rate'] = team['total_overturns'] / team['total_challenges']
    team = team[team['total_challenges'] >= 20].sort_values('rate', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 9))

    colors = [GREEN if r >= 0.50 else ACCENT if r < 0.40 else ACCENT2 for r in team['rate']]
    bars = ax.barh(team['team_abbr'], team['rate'], color=colors, height=0.7, edgecolor='none')

    # Add value labels
    for bar, val in zip(bars, team['rate']):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', ha='left', fontsize=8.5, color=TEXT_COL)

    # League avg line
    league_avg = team['total_overturns'].sum() / team['total_challenges'].sum()
    ax.axvline(league_avg, color=YELLOW, linewidth=1.5, linestyle='--', alpha=0.9)
    ax.text(league_avg + 0.003, len(team) - 0.5, f'MLB avg\n{league_avg:.1%}',
            color=YELLOW, fontsize=8.5, va='top')

    ax.set_xlabel('ABS Challenge Overturn Rate (higher = better decision-making)')
    ax.set_title('Which Teams Win the Most ABS Challenges?', fontsize=15, pad=12, fontweight='bold')
    ax.set_xlim(0, 0.80)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis='x', alpha=0.4)
    ax.set_facecolor(PANEL_BG)

    legend_els = [
        mpatches.Patch(color=GREEN,   label='≥ 50% overturn rate'),
        mpatches.Patch(color=ACCENT2, label='40–50% overturn rate'),
        mpatches.Patch(color=ACCENT,  label='< 40% overturn rate'),
    ]
    ax.legend(handles=legend_els, loc='lower right', fontsize=8.5,
              facecolor=PANEL_BG, edgecolor='#444', labelcolor=TEXT_COL)

    ax.text(0.99, -0.04, 'Source: Baseball Savant ABS Leaderboard, 2026 MLB Season (through May 18)',
            transform=ax.transAxes, ha='right', fontsize=7, color='#888', style='italic')

    plt.tight_layout()
    savefig('fig1_team_challenge_rate.png')


# ── Figure 2: Top/Bottom Batters by Challenge Value Added ─────────────────────
def fig_batter_value_added(df_bat):
    print('Building Figure 2: Batter challenge value added...')

    df = df_bat[df_bat['n_challenges'] >= 5].copy()
    df = df.sort_values('total_vs_expected', ascending=False)
    top10 = df.head(10)
    bot10 = df.tail(10)
    plot_df = pd.concat([top10, pd.DataFrame([{'entity_name': '', 'total_vs_expected': np.nan, 'team_abbr': ''}]), bot10])

    fig, ax = plt.subplots(figsize=(11, 8))

    # Lollipop chart
    y_pos = range(len(plot_df))
    for i, (_, row) in enumerate(plot_df.iterrows()):
        if pd.isna(row['total_vs_expected']):
            continue
        val = row['total_vs_expected']
        color = GREEN if val > 0 else ACCENT
        ax.plot([0, val], [i, i], color=color, linewidth=2, alpha=0.8, solid_capstyle='round')
        ax.scatter(val, i, color=color, s=80, zorder=5)
        label = f"{row['entity_name']} ({row['team_abbr']})"
        ha = 'left' if val > 0 else 'right'
        offset = 0.08 if val > 0 else -0.08
        ax.text(val + offset, i, label, va='center', ha=ha, fontsize=8.5, color=TEXT_COL)

    ax.axvline(0, color='#555', linewidth=1.0, linestyle='-')
    ax.set_yticks([])
    ax.set_xlabel('Net Challenge Value Added vs. Expected  (challenge equivalents)')
    ax.set_title('ABS Challenge Value Added by Batter\n(min. 5 challenges, 2026 MLB Season)',
                 fontsize=14, pad=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.4)

    ax.annotate('Better decision-\nmaking than expected →', xy=(4.5, 20), fontsize=8,
                color=GREEN, ha='center', style='italic')
    ax.annotate('← Worse decision-\nmaking than expected', xy=(-4, 2), fontsize=8,
                color=ACCENT, ha='center', style='italic')

    ax.text(0.99, -0.05, 'Source: Baseball Savant ABS Leaderboard, 2026 MLB Season (through May 18)',
            transform=ax.transAxes, ha='right', fontsize=7, color='#888', style='italic')

    plt.tight_layout()
    savefig('fig2_batter_value_added.png')


# ── Figure 3: Game-Outcome Impact — K/BB Flipped by Team ─────────────────────
def fig_outcome_impact(df_bat):
    print('Building Figure 3: K and BB flipped by team...')

    team = df_bat.groupby('team_abbr').agg(
        k_flipped=('n_strikeouts_flip', 'sum'),
        bb_flipped=('n_walks_flip', 'sum'),
    ).reset_index()
    team['net'] = team['bb_flipped'] - team['k_flipped']
    team = team.sort_values('net', ascending=False)

    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(team))
    w = 0.35

    bars_k  = ax.bar(x - w/2, team['k_flipped'],  w, label='Strikeouts avoided (ball→call)',  color=GREEN,  alpha=0.9, edgecolor='none')
    bars_bb = ax.bar(x + w/2, team['bb_flipped'], w, label='Strikeouts gained (strike→call)', color=ACCENT, alpha=0.9, edgecolor='none')

    # Net difference line
    ax2 = ax.twinx()
    ax2.plot(x, team['net'], color=YELLOW, linewidth=2, marker='o', markersize=5, label='Net gain (green - red)')
    ax2.axhline(0, color='#666', linewidth=0.8, linestyle='--')
    ax2.set_ylabel('Net Outcome Gain', color=YELLOW, fontsize=10)
    ax2.tick_params(axis='y', colors=YELLOW)
    ax2.set_facecolor('none')

    ax.set_xticks(x)
    ax.set_xticklabels(team['team_abbr'], rotation=45, ha='right', fontsize=8.5)
    ax.set_ylabel('Number of At-Bat Outcomes Flipped')
    ax.set_title('ABS Challenges: At-Bat Outcomes Flipped by Team\n(Strikeouts avoided vs. Strikeouts gained, 2026 Season)',
                 fontsize=13, pad=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.4)

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=8.5,
              facecolor=PANEL_BG, edgecolor='#444', labelcolor=TEXT_COL, loc='upper right')

    ax.text(0.99, -0.14, 'Source: Baseball Savant ABS Leaderboard, 2026 MLB Season (through May 18)',
            transform=ax.transAxes, ha='right', fontsize=7, color='#888', style='italic')

    plt.tight_layout()
    savefig('fig3_outcome_impact.png')


# ── Figure 4: Called Strike Zone Heatmap vs. ABS Zone ─────────────────────────
def fig_strike_zone_heatmap(df_pitches):
    print('Building Figure 4: Strike zone heatmap vs ABS calls...')

    # Filter to called pitches near the plate
    called = df_pitches[df_pitches['description'].isin(['called_strike', 'ball', 'automatic_ball', 'automatic_strike'])].copy()
    called = called.dropna(subset=['plate_x', 'plate_z'])
    called = called[(called['plate_x'].between(-2.5, 2.5)) & (called['plate_z'].between(0, 5))]

    called['is_strike'] = called['description'].isin(['called_strike', 'automatic_strike']).astype(int)
    called['call_type']  = called['description'].map({
        'called_strike': 'Umpire Strike',
        'ball':          'Umpire Ball',
        'automatic_ball':   'ABS Ball',
        'automatic_strike': 'ABS Strike',
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)

    # Standard zone boundaries (in feet)
    sz_left, sz_right = -0.708, 0.708
    sz_bot,  sz_top   = 1.5,   3.5

    for ax, (label, desc_vals, color) in zip(axes, [
        ('Umpire Called Strikes',   ['called_strike'],   ACCENT2),
        ('Umpire Called Balls\n(edge only ±3 in.)', ['ball'], ACCENT),
    ]):
        subset = called[called['description'].isin(desc_vals)].copy()
        if label.startswith('Umpire Called Balls'):
            # Only edge balls (within 3 in. of zone boundary)
            edge = (
                ((subset['plate_x'].abs().between(sz_left - 0.25, sz_right + 0.25)) |
                 (subset['plate_z'].between(sz_bot - 0.25, sz_bot + 0.25)) |
                 (subset['plate_z'].between(sz_top - 0.25, sz_top + 0.25)))
            )
            subset = subset[edge]

        h = ax.hexbin(
            subset['plate_x'], subset['plate_z'],
            gridsize=28, cmap='YlOrRd' if desc_vals == ['called_strike'] else 'Blues',
            mincnt=1, linewidths=0.1
        )
        plt.colorbar(h, ax=ax, label='Pitch count')

        # Strike zone box
        zone = mpatches.FancyBboxPatch(
            (sz_left, sz_bot), sz_right - sz_left, sz_top - sz_bot,
            linewidth=2, edgecolor=YELLOW, facecolor='none',
            boxstyle='square,pad=0', zorder=10
        )
        ax.add_patch(zone)

        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(0.5, 5.0)
        ax.set_xlabel('Horizontal Position (ft,  ← LHB side | RHB side →)')
        ax.set_ylabel('Vertical Position (ft)')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_facecolor(PANEL_BG)

    fig.suptitle('Where Umpires Call Strikes vs. Where Balls Get Called Near the Zone\n'
                 '(2026 MLB Season — ABS "yellow box" shows true strike zone)',
                 fontsize=13, fontweight='bold', y=1.01)

    fig.text(0.99, -0.02, 'Source: Baseball Savant Statcast, 2026 MLB Season (through May 18)',
             ha='right', fontsize=7, color='#888', style='italic')

    plt.tight_layout()
    savefig('fig4_strike_zone_heatmap.png')


# ── Figure 5: ABS Challenge Rate vs. Success Rate (Batter Scatter) ────────────
def fig_challenge_scatter(df_bat):
    print('Building Figure 5: Challenge frequency vs. success scatter...')

    df = df_bat[df_bat['n_challenges'] >= 4].copy()
    df['rate_overturns'] = df['rate_overturns'].clip(0, 1)

    # Annotate notable players
    top_n = df.nlargest(6, 'total_vs_expected')
    bot_n = df.nsmallest(6, 'total_vs_expected')
    highlight = pd.concat([top_n, bot_n])

    fig, ax = plt.subplots(figsize=(11, 7))

    sc = ax.scatter(
        df['n_challenges'], df['rate_overturns'],
        c=df['total_vs_expected'], cmap='RdYlGn',
        s=60, alpha=0.75, edgecolors='#333', linewidths=0.4,
        vmin=-5, vmax=5
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label('Net Challenge Value vs. Expected', color=TEXT_COL, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    # Annotate highlights
    for _, row in highlight.iterrows():
        ax.annotate(
            f"{row['entity_name'].split()[-1]} ({row['team_abbr']})",
            xy=(row['n_challenges'], row['rate_overturns']),
            xytext=(6, 3), textcoords='offset points',
            fontsize=7.5, color=TEXT_COL,
            arrowprops=dict(arrowstyle='->', color='#666', lw=0.8)
        )

    # League avg lines
    avg_rate = df['n_overturns'].sum() / df['n_challenges'].sum()
    avg_chal = df['n_challenges'].mean()
    ax.axhline(avg_rate, color=YELLOW, linewidth=1.2, linestyle='--', alpha=0.8)
    ax.axvline(avg_chal, color=YELLOW, linewidth=1.2, linestyle='--', alpha=0.8)
    ax.text(avg_chal + 0.2, 0.05, f'Avg challenges\n({avg_chal:.1f})', color=YELLOW, fontsize=8)
    ax.text(0.5, avg_rate + 0.02, f'Avg success rate ({avg_rate:.1%})', color=YELLOW, fontsize=8)

    ax.set_xlabel('Total ABS Challenges Attempted')
    ax.set_ylabel('Challenge Overturn Rate (success %)')
    ax.set_title('ABS Challenge Frequency vs. Success Rate by Batter\n'
                 '(Color = net value added vs. expected, min. 4 challenges, 2026 Season)',
                 fontsize=13, pad=12, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3)

    ax.text(0.99, -0.07, 'Source: Baseball Savant ABS Leaderboard, 2026 MLB Season (through May 18)',
            transform=ax.transAxes, ha='right', fontsize=7, color='#888', style='italic')

    plt.tight_layout()
    savefig('fig5_challenge_scatter.png')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading data...')
    df_bat, df_pit, df_pitches = load_data()
    print(f'  Batters: {len(df_bat)} | Pitches: {len(df_pitches):,}')
    print()

    fig_team_challenge_rate(df_bat)
    fig_batter_value_added(df_bat)
    fig_outcome_impact(df_bat)
    fig_strike_zone_heatmap(df_pitches)
    fig_challenge_scatter(df_bat)

    print('\nAll 5 figures saved to figures/')
