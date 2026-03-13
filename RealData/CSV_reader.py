import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_path = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/RealData/CSV"

# 예) Data_30.csv, Data_45.csv ... 만
# csv_paths = sorted(glob.glob(os.path.join(base_path, "Data_*.csv")))
# 필요하면 특정 2개만 고르기:
csv_paths = [os.path.join(base_path,"Data_0.csv"), os.path.join(base_path,"Data_90.csv")]

def load_and_clean(csv_path):
    # 인코딩: cp949 우선
    try:
        df = pd.read_csv(csv_path, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 첫 컬럼이 Unnamed면 버림(인코딩/엑셀 잔재)
    if len(df.columns) >= 1 and str(df.columns[0]).startswith("Unnamed"):
        df = df.iloc[:, 1:].copy()

    # 앞 4개 컬럼만 사용하고 이름 강제
    df = df.iloc[:, :4].copy()
    df.columns = ["NO", "Time", "Value", "Pressure"]

    # 요약행(???果: ...) 제거: NO가 숫자가 아닌 행 제거
    df["NO"] = pd.to_numeric(df["NO"], errors="coerce")
    df = df[df["NO"].notna()].copy()

    # Value: "0 N" -> 0, "377 N" -> 377
    v = (df["Value"].astype(str)
         .str.replace(",", "", regex=False)
         .str.extract(r"([-+]?\d*\.?\d+)", expand=False))
    df["Value_N"] = pd.to_numeric(v, errors="coerce").fillna(0.0)

    # NO는 원래 내림차순일 수 있으니 오름차순 정렬
    df = df.sort_values("NO").reset_index(drop=True)

    # 초기 불안정 샘플 제거 (앞 200개 행)
    df = df.iloc[200:].reset_index(drop=True)
    return df

def find_first_periodic_peak(x, y):
    """
    '최댓값'이 아니라, 0이 반복되다가 시작되는 이후 구간에서
    (1) 국소 최대(local maxima)
    (2) 상위 분위수 이상(큰 peak)
    (3) 너무 가까운 peak는 제거(min distance)
    한 뒤, '첫 번째' peak를 위상 기준으로 사용.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    # 0이 끝나는 onset 이후만 보자
    onset = np.argmax(y > 0) if np.any(y > 0) else 0
    y2 = y[onset:]
    x2 = x[onset:]

    if len(y2) < 3:
        return None

    # 큰 값 임계치: 95퍼센타일(데이터에 맞게 90~99로 조절 가능)
    thr = np.quantile(y2, 0.95)

    # 국소 최대 찾기
    locmax = np.where((y2[1:-1] > y2[:-2]) & (y2[1:-1] >= y2[2:]) & (y2[1:-1] >= thr))[0] + 1
    if len(locmax) == 0:
        # fallback: 임계치만 적용해서 첫 지점
        idx = np.argmax(y2 >= thr)
        return x2[idx]

    # 너무 가까운 peak 제거 (샘플 기준 간격; 필요시 10~50으로 조절)
    min_dist = max(5, int(0.02 * len(y2)))
    kept = []
    last = -10**9
    for idx in locmax:
        if idx - last >= min_dist:
            kept.append(idx)
            last = idx

    return x2[kept[0]]  # 첫 peak의 x(NO)

# --- 데이터 로드
series = []
for p in csv_paths:
    df = load_and_clean(p)
    label = os.path.basename(p).replace(".csv", "")
    series.append((label, df["NO"].to_numpy(), df["Value_N"].to_numpy()))

if len(series) < 2:
    raise RuntimeError("2개 이상 CSV를 지정/발견해야 정렬 비교가 가능합니다.")

# --- 각 파일의 '첫 periodic peak' 위치를 찾아서 정렬(shift) 계산
peak_x = {}
degree = []
for label, x, y in series:
    px = find_first_periodic_peak(x, y)
    peak_x[label] = px
    print(f"{label}: first periodic peak at NO = {px}")
    degree.append(label)
# 기준(reference) = 첫 번째 파일
ref_label = series[0][0]
ref_peak = peak_x[ref_label]
if ref_peak is None:
    raise RuntimeError(f"기준 파일 {ref_label}에서 peak를 못 찾았습니다. thr(0.95)나 min_dist를 조정하세요.")

# --- 플롯: (왼쪽) 원본, (오른쪽) peak 정렬 후
plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

colors = plt.cm.tab10(np.linspace(0, 1, len(series)))

for i, (label, x, y) in enumerate(series):
    c = colors[i]

    # 원본
    ax0.plot(x, y, color=c, linewidth=1.8, label=label)

    # 정렬: peak가 ref_peak에 오도록 x를 shift
    if peak_x[label] is None:
        x_aligned = x  # peak 못 찾으면 그대로
    else:
        shift = ref_peak - peak_x[label]
        x_aligned = x + shift

    ax1.plot(x_aligned, y, color=c, linewidth=1.8, label=label)

# 표시선(기준 peak 위치)
ax0.axvline(ref_peak, color="k", linestyle="--", linewidth=1.0)
ax1.axvline(ref_peak, color="k", linestyle="--", linewidth=1.0)

ax0.set_title("Unaligned (NO asc)")
ax0.set_xlabel("NO (ascending)")
ax0.set_ylabel("Value (N)")
ax0.grid(True, alpha=0.25)
ax0.legend()

ax1.set_title(f"Phase-aligned to {ref_label} peak")
ax1.set_xlabel("Aligned NO")
ax1.grid(True, alpha=0.25)
ax1.legend()

plt.tight_layout()
plt.savefig(os.path.join(base_path, f"multi_phase_aligned_{degree[0], degree[1]}.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: multi_phase_aligned.png")
