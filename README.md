# Challenging the Call: MLB's ABS Revolution
**DSC 106 Final Project — UC San Diego, Spring 2026**

An interactive explorable explanation of MLB's Automated Ball-Strike (ABS) Challenge System and how it is reshaping the 2026 season.

## Data Sources
- **ABS Leaderboard**: `baseballsavant.mlb.com/leaderboard/abs-challenges`
- **Statcast Pitches**: `baseballsavant.mlb.com/statcast_search`

## Scripts
| File | Purpose |
|------|---------|
| `fetch_data.py` | Download fresh data from Baseball Savant → `data/` |
| `make_figures.py` | Generate 5 static proposal figures → `figures/` |

## Quickstart
```bash
python3 fetch_data.py   # refresh data
python3 make_figures.py # regenerate figures
```

Open `index.html` in a browser or deploy via GitHub Pages.