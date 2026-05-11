# F1 Telemetrix

![Race Replay Preview](./resources/preview.png)

A Python desktop application that replays Formula 1 race and qualifying sessions using real telemetry data from the [FastF1](https://github.com/theOehrly/Fast-F1) API, rendered with an interactive [Arcade](https://api.arcade.academy/en/latest/) window.

## Base Project

The race replay visualization is built on top of the open source [F1 Race Replay](https://tomshaw.dev) project by [Tom Shaw](https://tomshaw.dev). It loads real F1 telemetry via FastF1 and renders an interactive Arcade window with driver positions on a live track, a leaderboard with tyre compounds, lap counter, weather, playback controls, and per-driver telemetry (speed, gear, DRS). Qualifying mode adds a segment leaderboard, DRS zone toggles, and telemetry charts over lap distance.

## What I Built On Top

### CAN Bus Security Simulation

Using the real telemetry frames from the replay as a data source, I built a layer that simulates the CAN bus traffic a Formula 1 car would generate and injects attacks into that traffic:

- **Injection attack** — forged speed/RPM values for a target driver
- **Fuzzing attack** — random-payload bursts across arbitration IDs

### Hybrid Intrusion Detection System (IDS)

A three-layer detection pipeline that runs against the simulated CAN traffic:

1. **Rule-Based IDS** — per-frame whitelist, payload-bounds, and frequency checks against the observed baseline
2. **Isolation Forest** (unsupervised) — streaming anomaly detector that trains on a baseline window then scores live frames
3. **Random Forest** (supervised) — post-simulation classifier over the labeled CAN log; supports cross-run evaluation where the model trains on N-1 runs and tests on a held-out run

Each evaluation run is self-contained and saved under `computed_data/can_security_ml/runs/`.

---

## Setup

```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

A `.fastf1-cache/` folder is created automatically on first run. Computed telemetry frames are cached under `computed_data/` so subsequent runs load instantly.

---

## Running

### Race replay (default)
```bash
python main.py --year 2025 --round 12
```

### Sprint race
Sprint sessions only exist on certain rounds — use `--list-sprints` to see them.
```bash
python main.py --year 2025 --round 13 --sprint
```

### Qualifying
```bash
python main.py --year 2025 --round 12 --qualifying
```

### Sprint qualifying
```bash
python main.py --year 2025 --round 13 --sprint-qualifying
```

### Force recompute telemetry cache
```bash
python main.py --year 2025 --round 12 --refresh-data
```

### Find round numbers
```bash
python main.py --year 2025 --list-rounds
python main.py --year 2025 --list-sprints
```

---

## Security Demo

### Basic security overlay (injection attack)
```bash
python main.py --year 2025 --round 12 --security-demo
```

### Fuzzing attack
```bash
python main.py --year 2025 --round 12 --security-demo --attack-fuzzing
```

### Target a specific driver
```bash
python main.py --year 2025 --round 12 --security-demo --attack-target VER
```

### Run the ML IDS evaluation (no replay window)
```bash
python main.py --year 2024 --round 1 --security-demo --ml-eval --ml-only
```

### Recommended honest evaluation (mid-race attack + cross-run RF)
```bash
python main.py --year 2024 --round 1 --security-demo --ml-eval --ml-only `
    --ml-attack-start 600 --ml-batch-runs 5 --ml-cross-run-eval
```

Each run outputs a directory under `computed_data/can_security_ml/runs/<timestamp>/` with a labeled CAN log CSV, a JSON metrics report, an IF alert log, and the trained RF model.

See `QUICKSTART_COMMANDS.txt` for the full list of `--ml-*` tuning flags.

---

## Controls (in-app)

| Action | Key / Button |
|---|---|
| Pause / Resume | `SPACE` |
| Rewind / Fast-forward | `←` / `→` |
| Cycle playback speed | `↑` / `↓` or speed button |
| Set speed directly | `1` – `4` |
| Switch Telemetry / Attacks tab | `TAB` (security mode) |

---

## Requirements

- Python 3.8+
- `fastf1`, `pandas`, `numpy`, `matplotlib`
- `arcade`, `pyglet`
- `scikit-learn`, `joblib`

---

## License

MIT — data sourced from publicly available APIs for educational and non-commercial use only. Formula 1 and related trademarks are the property of their respective owners.

---

CAN bus security layer and IDS built on top of [F1 Race Replay](https://tomshaw.dev) by Tom Shaw.
