#!/usr/bin/env python3
"""PID controller for GDP-DeepSulk 12.28M CPU training.

Controls batch_size and num_threads by watching CPU Tctl and available RAM.
Designed for Ryzen 3 / low-RAM CPU-only training.
"""

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

CONFIG_PATH = Path("configs/tiny_lowram_fast.json")
LOG_PATH = Path("pid_control_train.log")

TRAIN_CMD = [
    "python", "-u", "train/pretrain.py",
    "--train-bin", "data/train.bin",
    "--valid-bin", "data/valid.bin",
    "--tokenizer", "tokenizer/gdp_deepsulk_tokenizer.json",
    "--out-dir", "checkpoints/tiny_lowram",
    "--config", str(CONFIG_PATH),
    "--resume",
]

# Low load -> high load. Level 19 is intentionally aggressive.
PROFILES = [
    {"batch_size": 2,  "num_threads": 2},
    {"batch_size": 4,  "num_threads": 2},
    {"batch_size": 6,  "num_threads": 2},
    {"batch_size": 8,  "num_threads": 2},
    {"batch_size": 8,  "num_threads": 3},
    {"batch_size": 8,  "num_threads": 4},
    {"batch_size": 10, "num_threads": 4},
    {"batch_size": 12, "num_threads": 4},
    {"batch_size": 16, "num_threads": 4},
    {"batch_size": 20, "num_threads": 4},
    {"batch_size": 24, "num_threads": 4},
    {"batch_size": 24, "num_threads": 5},
    {"batch_size": 28, "num_threads": 5},
    {"batch_size": 32, "num_threads": 5},
    {"batch_size": 32, "num_threads": 6},
    {"batch_size": 36, "num_threads": 6},
    {"batch_size": 36, "num_threads": 7},
    {"batch_size": 40, "num_threads": 7},
    {"batch_size": 40, "num_threads": 8},
    {"batch_size": 48, "num_threads": 8},
]

TEMP_TARGET = 68.0
TEMP_MAX = 75.0
TEMP_HARD_DOWN = 76.0

AVAIL_MIN = 0.5
AVAIL_SOFT = 1.0
AVAIL_UP = 1.3

# Fast response settings.
CHECK_SEC = 2
STARTUP_HOLD_SEC = 20
CHANGE_COOLDOWN_SEC = 10

# PID gains. Output maps to discrete profile up/down.
KP = 0.28
KI = 0.03
KD = 0.45

INTEGRAL_MIN = -60.0
INTEGRAL_MAX = 60.0
EMA_ALPHA = 0.35
UP_THRESHOLD = 0.5
DOWN_THRESHOLD = -1.0


def log(msg: str) -> None:
    line = f"[PID {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def get_available_gib() -> float | None:
    try:
        out = subprocess.check_output(["free", "-m"], text=True)
        for line in out.splitlines():
            if line.startswith("Mem:"):
                return float(line.split()[6]) / 1024.0
    except Exception:
        return None
    return None


def get_cpu_temp() -> float | None:
    try:
        out = subprocess.check_output(["sensors"], text=True, errors="ignore")
    except Exception:
        return None

    preferred = []
    alltemps = []
    for line in out.splitlines():
        low = line.lower()
        if "crit" in low or "high" in low:
            continue
        nums = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*°C", line)
        if not nums:
            continue
        vals = [float(x) for x in nums]
        alltemps.extend(vals)
        if any(k in line for k in ["Tctl", "Tdie", "Package id 0", "CPU"]):
            preferred.extend(vals)

    if preferred:
        return max(preferred)
    if alltemps:
        return max(alltemps)
    return None


def apply_profile(level: int) -> None:
    prof = PROFILES[level]
    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
    cfg["batch_size"] = prof["batch_size"]
    cfg["grad_accum_steps"] = 1
    cfg["num_threads"] = prof["num_threads"]
    cfg["interop_threads"] = 1
    cfg["eval_interval"] = 2000
    cfg["eval_iters"] = 1
    cfg["eval_train_loss"] = False
    cfg["log_interval"] = 100
    cfg["save_last_interval"] = 500
    cfg["device"] = "cpu"
    json.dump(cfg, open(CONFIG_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log(f"apply level={level} batch_size={prof['batch_size']} num_threads={prof['num_threads']}")


def start_train():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    f = LOG_PATH.open("a", encoding="utf-8", buffering=1)
    f.write("\n===== TRAIN START =====\n")
    p = subprocess.Popen(
        TRAIN_CMD,
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    return p, f


def stop_train(p, f) -> None:
    if p and p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            time.sleep(5)
        except Exception:
            pass
    if p and p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    try:
        f.close()
    except Exception:
        pass


def calc_pid_action(level: int, temp: float | None, avail: float | None, state: dict, dt_sec: float):
    if temp is None:
        return level, 0.0, "no_temp"

    dt_min = max(dt_sec / 60.0, 0.1)

    if state["temp_ema"] is None:
        state["temp_ema"] = temp
    else:
        state["temp_ema"] = EMA_ALPHA * temp + (1.0 - EMA_ALPHA) * state["temp_ema"]

    t = state["temp_ema"]
    error = TEMP_TARGET - t  # positive = cool, raise load

    state["integral"] += error * dt_min
    state["integral"] = clamp(state["integral"], INTEGRAL_MIN, INTEGRAL_MAX)

    if state["prev_error"] is None:
        d_error = 0.0
    else:
        d_error = (error - state["prev_error"]) / dt_min
    state["prev_error"] = error

    p_term = KP * error
    i_term = KI * state["integral"]
    d_term = KD * d_error
    output = p_term + i_term + d_term
    reason = "pid"

    if avail is not None:
        if avail < AVAIL_MIN:
            output -= 5.0
            reason = "ram_hard_down"
        elif avail < AVAIL_SOFT:
            output -= 1.5 * (AVAIL_SOFT - avail)
            reason = "ram_soft_down"
        elif avail < AVAIL_UP and output > 0:
            output = min(output, 0.5)
            reason = "ram_blocks_up"

    if temp >= TEMP_HARD_DOWN:
        output -= 5.0
        reason = "temp_hard_down"
    elif temp >= TEMP_MAX:
        output -= 3.0
        reason = "temp_down"

    if d_error < -2.0:
        output -= 0.8
        reason = "temp_rising_fast"

    next_level = level
    if output >= UP_THRESHOLD:
        next_level = min(level + 1, len(PROFILES) - 1)
    elif output <= DOWN_THRESHOLD:
        next_level = max(level - 1, 0)

    log(
        f"pid temp_raw={temp:.1f} temp_ema={t:.1f} "
        f"err={error:.2f} I={state['integral']:.2f} D={d_error:.2f} "
        f"P={p_term:.2f} Iterm={i_term:.2f} Dterm={d_term:.2f} "
        f"out={output:.2f} reason={reason}"
    )
    return next_level, output, reason


def main() -> None:
    level = 3
    state = {"integral": 0.0, "prev_error": None, "temp_ema": None}

    while True:
        apply_profile(level)
        p, f = start_train()
        start_time = time.time()
        last_change = time.time()
        last_check = time.time()

        while True:
            time.sleep(CHECK_SEC)
            now = time.time()
            dt_sec = now - last_check
            last_check = now
            temp = get_cpu_temp()
            avail = get_available_gib()
            alive = p.poll() is None
            runtime = now - start_time

            log(f"status alive={alive} temp={temp}C avail={avail}GiB level={level}")

            if not alive:
                code = p.returncode
                log(f"train exited code={code}")
                try:
                    f.close()
                except Exception:
                    pass
                if level > 0:
                    level -= 1
                    state["integral"] = min(state["integral"], 0.0)
                    log("process stopped -> level down")
                time.sleep(10)
                break

            if runtime < STARTUP_HOLD_SEC:
                continue
            if now - last_change < CHANGE_COOLDOWN_SEC:
                continue

            next_level, output, reason = calc_pid_action(level, temp, avail, state, dt_sec)
            if next_level != level:
                log(f"change level {level} -> {next_level} output={output:.2f} reason={reason}")
                stop_train(p, f)
                if next_level == 0 or next_level == len(PROFILES) - 1:
                    state["integral"] *= 0.5
                level = next_level
                break


if __name__ == "__main__":
    main()
