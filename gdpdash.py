#!/usr/bin/env python3
"""Interactive terminal dashboard for GDP-DeepSulk training."""

import curses
import re
import subprocess
import time
from pathlib import Path

BOOST_FILE = "/sys/devices/system/cpu/cpufreq/boost"
LOG_PATH = Path("pid_control_train.log")
CONFIG_PATH = Path("configs/tiny_lowram_fast.json")
TEMP_ALERT = 85.0
UPDATE_SEC = 1.0

last_alert = 0.0
status_msg = ""


def sh(cmd: str, timeout: int = 3) -> str:
    try:
        return subprocess.check_output(
            cmd,
            shell=True,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        ).strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()
    except Exception as e:
        return str(e)


def get_boost() -> str:
    try:
        v = Path(BOOST_FILE).read_text().strip()
        return "ON" if v == "1" else "OFF"
    except Exception:
        return "UNKNOWN"


def set_boost(v: str) -> str:
    value = "1" if v == "on" else "0"
    out = sh(f"echo {value} | sudo -n tee {BOOST_FILE} >/dev/null && echo OK || echo FAILED", timeout=5)
    if "OK" in out:
        return f"BOOST: {'ON' if value == '1' else 'OFF'}"
    return "BOOST change failed. Run sudo -v in another shell first."


def get_mem():
    out = sh("free -h")
    lines = out.splitlines()
    mem_lines = [x for x in lines if x.startswith("Mem:") or x.startswith("Swap:")]
    avail = sh("free -m | awk '/^Mem:/ {printf \"%.2f\", $7/1024}'")
    return mem_lines, avail


def get_temp():
    out = sh("sensors")
    tctl = None
    lines = []
    for line in out.splitlines():
        if re.search(r"Tctl|Tdie|Package id 0|edge|Composite|PPT|temp1", line):
            lines.append(line)
        if "Tctl:" in line:
            m = re.search(r"\+?([0-9]+(?:\.[0-9]+)?)\s*°C", line)
            if m:
                tctl = float(m.group(1))
    return tctl, lines[:10]


def get_cpu_mhz() -> str:
    p = Path("/sys/devices/system/cpu")
    vals = []
    for f in sorted(p.glob("cpu*/cpufreq/scaling_cur_freq")):
        try:
            vals.append(int(f.read_text().strip()) / 1000.0)
        except Exception:
            pass
    if not vals:
        out = sh("awk -F: '/cpu MHz/ {print $2}' /proc/cpuinfo")
        for x in out.splitlines():
            try:
                vals.append(float(x.strip()))
            except Exception:
                pass
    if not vals:
        return "UNKNOWN"
    return f"avg {sum(vals)/len(vals):.0f} / min {min(vals):.0f} / max {max(vals):.0f} MHz"


def get_config():
    if not CONFIG_PATH.exists():
        return ["config not found"]
    out = sh("grep -E '\"batch_size\"|\"num_threads\"|\"eval_interval\"|\"save_last_interval\"' configs/tiny_lowram_fast.json | sed 's/[\",]//g; s/^ *//'")
    return out.splitlines()


def get_process():
    out = sh("ps -eo pid,pcpu,pmem,etime,cmd | grep -E 'pid_control_train_12m.py|train/pretrain.py' | grep -v grep | head -n 6")
    return out.splitlines() if out else ["no training process"]


def get_counts():
    c = sh("pgrep -fc 'pid_control_train_12m.py'")
    t = sh("pgrep -fc 'train/pretrain.py'")
    return c, t


def get_log():
    if not LOG_PATH.exists():
        return ["no pid_control_train.log"]
    return sh("tail -n 8 pid_control_train.log").splitlines()


def draw_line(stdscr, y: int, text: str, attr: int = 0) -> None:
    h, w = stdscr.getmaxyx()
    if y < h:
        stdscr.addstr(y, 0, text[: w - 1].ljust(w - 1), attr)


def command_mode(stdscr) -> bool:
    global status_msg

    # Blocking input only during command mode. This fixes ESC -> command prompt timeout.
    stdscr.nodelay(False)
    stdscr.timeout(-1)
    curses.curs_set(1)
    curses.echo()

    h, w = stdscr.getmaxyx()
    stdscr.move(h - 1, 0)
    stdscr.clrtoeol()
    stdscr.addstr(h - 1, 0, ":")

    try:
        cmd = stdscr.getstr(h - 1, 1, w - 2).decode("utf-8", "ignore").strip()
    except Exception:
        cmd = ""

    curses.noecho()
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    if cmd in ["q", "quit", "!q", "!quit"]:
        return False

    if cmd.startswith("!boost"):
        parts = cmd.split()
        if len(parts) >= 2 and parts[1] in ["on", "off"]:
            status_msg = set_boost(parts[1])
        else:
            status_msg = f"BOOST: {get_boost()}"
        return True

    if cmd in ["!help", "help", "?"]:
        status_msg = "commands: !boost on | !boost off | !boost status | !q"
        return True

    if cmd:
        status_msg = f"unknown command: {cmd}"
    return True


def main(stdscr) -> None:
    global last_alert, status_msg

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)
    running = True

    while running:
        start = time.time()
        key = stdscr.getch()
        if key == 27:  # ESC
            running = command_mode(stdscr)
            continue
        if key in [ord("q"), ord("Q")]:
            break

        tctl, temp_lines = get_temp()
        mem_lines, avail = get_mem()
        mhz = get_cpu_mhz()
        boost = get_boost()
        cfg = get_config()
        proc = get_process()
        c_count, t_count = get_counts()
        logs = get_log()

        stdscr.erase()
        y = 0
        alert = tctl is not None and tctl >= TEMP_ALERT
        if alert:
            draw_line(stdscr, y, f" !!! CPU TEMP ALERT: Tctl {tctl:.1f}C >= {TEMP_ALERT:.0f}C !!! ", curses.A_REVERSE | curses.A_BOLD)
            now = time.time()
            if now - last_alert > 5:
                curses.beep()
                last_alert = now
        else:
            draw_line(stdscr, y, " GDP-DeepSulk CONTROL PANEL ", curses.A_REVERSE)
        y += 1
        draw_line(stdscr, y, time.strftime("%Y-%m-%d %H:%M:%S")); y += 2

        draw_line(stdscr, y, "===== CPU CLOCK / BOOST =====", curses.A_BOLD); y += 1
        draw_line(stdscr, y, f"Boost: {boost}"); y += 1
        draw_line(stdscr, y, f"CPU MHz: {mhz}"); y += 2

        draw_line(stdscr, y, "===== CONFIG =====", curses.A_BOLD); y += 1
        for line in cfg[:6]:
            draw_line(stdscr, y, line); y += 1
        y += 1

        draw_line(stdscr, y, "===== MEMORY =====", curses.A_BOLD); y += 1
        for line in mem_lines:
            draw_line(stdscr, y, line); y += 1
        draw_line(stdscr, y, f"Available GiB: {avail}"); y += 2

        draw_line(stdscr, y, "===== TEMP =====", curses.A_BOLD); y += 1
        for line in temp_lines:
            draw_line(stdscr, y, line); y += 1

        if tctl is not None:
            if tctl < 60:
                s = "TEMP STATUS: COOL"
            elif tctl < 70:
                s = "TEMP STATUS: GOOD"
            elif tctl < 75:
                s = "TEMP STATUS: WATCH"
            elif tctl < 85:
                s = "TEMP STATUS: HOT"
            else:
                s = "TEMP STATUS: ALERT"
            draw_line(stdscr, y, s); y += 1

        try:
            a = float(avail)
            if a < 0.5:
                rs = "RAM STATUS: DANGER"
            elif a < 1.0:
                rs = "RAM STATUS: WATCH"
            else:
                rs = "RAM STATUS: OK"
        except Exception:
            rs = "RAM STATUS: UNKNOWN"
        draw_line(stdscr, y, rs); y += 2

        draw_line(stdscr, y, "===== PROCESS COUNT =====", curses.A_BOLD); y += 1
        draw_line(stdscr, y, f"controller: {c_count}"); y += 1
        draw_line(stdscr, y, f"train:      {t_count}"); y += 2

        draw_line(stdscr, y, "===== PROCESS =====", curses.A_BOLD); y += 1
        for line in proc[:5]:
            draw_line(stdscr, y, line); y += 1
        y += 1

        draw_line(stdscr, y, "===== PID LOG =====", curses.A_BOLD); y += 1
        for line in logs:
            draw_line(stdscr, y, line); y += 1

        h, _ = stdscr.getmaxyx()
        footer = "ESC: command  |  !boost on/off/status  |  q: quit"
        if status_msg:
            footer += "  |  " + status_msg
        draw_line(stdscr, h - 1, footer, curses.A_REVERSE)
        stdscr.refresh()

        elapsed = time.time() - start
        if elapsed < UPDATE_SEC:
            time.sleep(UPDATE_SEC - elapsed)


if __name__ == "__main__":
    curses.wrapper(main)
