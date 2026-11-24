#!/bin/bash
# 跑两段考核视频，输出 README 里那张对比表。
# 用法：bash scripts/regression.sh [video1路径] [video2路径]
set -eu
cd "$(dirname "$0")/.." || exit 1

V1="${1:-assets/video1.avi}"
V2="${2:-assets/video2.avi}"
BIN=build/aim2027

if [ ! -x "$BIN" ]; then
  echo "找不到 $BIN，先编译："
  echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
  exit 1
fi
for f in "$V1" "$V2"; do
  if [ ! -f "$f" ]; then
    echo "缺视频文件：$f"
    exit 1
  fi
done

row() {
    local label="$1" video="$2" out
    shift 2
    if ! out=$("$BIN" "$video" --no-show --quiet "$@" 2>&1); then
        printf '%s\n' "$out" >&2
        return 1
    fi
    printf '%s\n' "$out" | awk -v label="$label" '
        /几何门通过帧/ { geometry = $2 " " $3 }
        /跟踪器接受帧/ { accepted = $2 " " $3 }
        /框显示总帧/ { shown = $2 " " $3 }
        /框消失次数/ { blink = $2 }
        /最长连续无框/ { gap = $2 }
        /上半屏几何通过帧/ { upper = $2 }
        /耗时 平均/ { cost = $3 " / " $5 }
        END {
            if (geometry == "" || shown == "") exit 1
            if (accepted == "") accepted = "—"
            printf "| %-16s | %-14s | %-14s | %-14s | %-5s | %-5s | %-5s | %-11s |\n",
                   label, geometry, accepted, shown, blink, gap, upper, cost
        }'
}

echo "===== aim2027 回归 ====="
echo
printf '| 配置 | 几何门通过 | 跟踪器接受 | 框显示 | 闪烁 | 最长空 | 上半屏几何通过 | 耗时ms(平均/中位) |\n'
printf '|---|---|---|---|---|---|---|---|\n'
row "video1 有跟踪" "$V1"
row "video1 无跟踪" "$V1" "--no-tracker"
row "video2 有跟踪" "$V2"
row "video2 无跟踪" "$V2" "--no-tracker"
echo
echo "闪烁 = 框从有到无的次数（越少越稳）；最长空 = 最长连续无框帧数；"
echo "上半屏几何通过 = 几何通过且选中候选中心在图像上半部的帧数。"
