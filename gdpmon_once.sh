#!/usr/bin/env bash
cd "$HOME/GDP-DeepSulk" || exit

echo "===== GDP-DeepSulk Monitor ====="
date
echo

echo "===== CONFIG ====="
grep -E '"batch_size"|"num_threads"|"eval_interval"|"save_last_interval"' configs/tiny_lowram_fast.json 2>/dev/null \
  | sed 's/[",]//g; s/^ *//' || echo "config not found"

echo
echo "===== CPU CLOCK / BOOST ====="
BOOST_FILE="/sys/devices/system/cpu/cpufreq/boost"
if [ -f "$BOOST_FILE" ]; then
  b="$(cat "$BOOST_FILE")"
  if [ "$b" = "1" ]; then
    echo "Boost: ON"
  else
    echo "Boost: OFF"
  fi
else
  echo "Boost: UNKNOWN"
fi

if ls /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq >/dev/null 2>&1; then
  awk '
  {
    mhz=$1/1000
    sum+=mhz
    n++
    if(n==1 || mhz<min) min=mhz
    if(n==1 || mhz>max) max=mhz
  }
  END {
    if(n>0) printf "CPU MHz: avg %.0f / min %.0f / max %.0f\n", sum/n, min, max
  }' /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
else
  awk -F: '/cpu MHz/ {
    mhz=$2+0
    sum+=mhz
    n++
    if(n==1 || mhz<min) min=mhz
    if(n==1 || mhz>max) max=mhz
  }
  END {
    if(n>0) printf "CPU MHz: avg %.0f / min %.0f / max %.0f\n", sum/n, min, max
    else print "CPU MHz: UNKNOWN"
  }' /proc/cpuinfo
fi

echo
echo "===== MEMORY ====="
free -h | awk 'NR==1 || /^Mem:/ || /^Swap:/'

avail="$(free -m | awk '/^Mem:/ {printf "%.2f", $7/1024}')"
echo "Available GiB: $avail"

echo
echo "===== TEMP ====="
sensors 2>/dev/null | grep -E "Tctl|Tdie|Package id 0|edge|Composite|PPT|temp1" | head -n 10

temp="$(sensors 2>/dev/null | awk '/Tctl:/ {gsub(/[^0-9.]/,"",$2); print $2; exit}')"

echo
if [ -n "$temp" ]; then
  awk -v t="$temp" 'BEGIN {
    if (t < 60) print "TEMP STATUS: COOL";
    else if (t < 70) print "TEMP STATUS: GOOD";
    else if (t < 75) print "TEMP STATUS: WATCH";
    else print "TEMP STATUS: HOT";
  }'
else
  echo "TEMP STATUS: UNKNOWN"
fi

awk -v a="$avail" 'BEGIN {
  if (a < 0.5) print "RAM STATUS: DANGER";
  else if (a < 1.0) print "RAM STATUS: WATCH";
  else print "RAM STATUS: OK";
}'

echo
echo "===== PROCESS ====="
ps -eo pid,pcpu,pmem,etime,cmd \
  | grep -E "pid_control_train_12m.py|train/pretrain.py" \
  | grep -v grep \
  | head -n 6 \
  || echo "no training process"

echo
echo "===== COUNT ====="
echo "controller: $(pgrep -fc 'pid_control_train_12m.py')"
echo "train:      $(pgrep -fc 'train/pretrain.py')"

echo
echo "===== PID LOG ====="
tail -n 8 pid_control_train.log 2>/dev/null || echo "no pid_control_train.log"

echo
echo "===== BOOST CONTROL ====="
echo "./gdpboost on      # boost ON"
echo "./gdpboost off     # boost OFF"
echo "./gdpboost status  # boost status"
