# roblox-up-and-coming-forecast-01

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/Hooknfts/roblox-up-and-coming-forecast-01/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/Hooknfts/roblox-up-and-coming-forecast-01/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

Forecasts Roblox's Up-and-Coming chart top 10, a probability for each game at each place, for the 72 hours after the history ends — from an empirical rank-transition kernel fitted to the chart's own history, with momentum, an ageing-out hazard, and an unnamed reserve for games that have not appeared yet. See METHOD.md for the method and the backtested scores.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Run from the kit's directory (or any directory holding `input/history.ndjson` and `input/task.json`). The program is one self-contained Python file, standard library only, deterministic: two runs on the same history write the same `forecast.json` byte for byte. It reads `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (else `/output/forecast.json`).

---

<sub>Entered on [make.fun](https://make.fun/?ref=y9fbhjgt&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
