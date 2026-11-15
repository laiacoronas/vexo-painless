# Vexo Movement Monitor — demo-safe patch

This replaces the **two files** you uploaded. Tested offline with Python's built-in `unittest` and synthetic sensor data; **real Vexo Bluetooth cannot be verified here**.

## Install in VS Code (Windows / PowerShell)

1. **Stop any running `movement_live.py` with Ctrl+C.** Close the original Vexo browser dashboard and other BLE clients.
2. In `C:\Users\lclai\Desktop\vexo-project\vexo-pain-monitor\vexo_movement_module`, back up your two original `.py` files.
3. Copy `movement_live.py`, `movement_engine.py` and optionally `test_movement.py` from this ZIP **into the same `vexo_movement_module` folder**, overwriting the originals.
4. Open a PowerShell terminal in that folder:

```powershell
cd C:\Users\lclai\Desktop\vexo-project\vexo-pain-monitor\vexo_movement_module
python -m unittest -v test_movement
```

## Live demo

Start with a clean baseline once:

```powershell
python .\movement_live.py --band 79 --reset-baseline
```

During calibration, sit still for ~30 seconds and then walk normally for ~30 seconds. Because the first window needs an initial step-counter reading, the first window may report `insufficient_data`. Four *valid* windows are needed **separately for still and walk**. Avoid moving the wrist unnaturally during calibration.

After that, restart **without** resetting the baseline:

```powershell
python .\movement_live.py --band 79
```

### Test without hardware (CLEARLY SYNTHETIC)

```powershell
python .\movement_live.py --synthetic --reset-baseline
```

This writes synthetic results with `source: "synthetic"`. If you use this fallback in your presentation, label it as **simulated data, not live wrist data**.

## Outputs

These paths remain the same as the old module (inside `vexo_movement_module/data/`):

- `movement_latest.json` — current window or `movement_status: "disconnected"`; retains all prior JSON keys and adds `connection_status`.
- `movement_windows.csv` — windows for UI and debugging; column names unchanged.
- `movement_baseline.json` — **only overwritten when a new valid baseline window is collected**.

Missing signals are `null`, not a misleading zero; `movement_deviation_index` stays `null` during calibration, missing data, and disconnection. The index is a **movement anomaly heuristic**, not a pain score or pain probability.

### Status guide

- `calibrating` — collecting baseline for this specific activity (still/walk/run).
- `typical`, `change`, `strong_change` — measured deviation from that activity's baseline.
- `insufficient_data` — missing readings. No score or baseline update.
- `disconnected` — reconnection in progress. No score. Retry with 2–15s backoff.

## Most important demo rule

**One Bluetooth owner only.** If your teammate's voice program also opens Vexo-HACK-79, the scripts will conflict. For simultaneous voice and motion, use one SDK connection subscribed to both streams and send results to the shared dashboard. `movement_live.py` is a standalone movement collector, not yet a combined audio+motion collector.

## Windows data file lock handling

`os.replace` can fail if a file is locked. This version uses uniquely named temporary files, retries Windows-lock failures, prints a warning, and keeps running. If warnings repeat, check that only one collector is running and that the target file is not held open continuously by another process. The dashboard should check the `timestamp` to distinguish stale data from live data.

## Tests included

Six unit tests cover valid/invalid samples, distinct baselines, reconnect step-counter behavior, Windows file-lock handling, JSON schema stability, and early-return disconnect detection. Real BLE performance is hardware-dependent and must be checked on your computer before the demo.
