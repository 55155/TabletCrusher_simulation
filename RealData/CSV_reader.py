import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 경로: 이 스크립트 위치 기준으로 CSV 폴더를 찾음 (OS 무관) ──
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSV")

# Data_0.csv, Data_30.csv, ... Data_90.csv 등 모든 각도 자동 수집
# 파일명에서 숫자를 추출해 오름차순 정렬
def _angle_key(p):
    m = re.search(r"Data_(-?\d+)", os.path.basename(p))
    return int(m.group(1)) if m else 9999

csv_paths = sorted(glob.glob(os.path.join(base_path, "Data_*.csv")), key=_angle_key)

# 0°를 기준(reference)으로 사용
REF_NAME = "Data_0"

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
for label, x, y in series:
    px = find_first_periodic_peak(x, y)
    peak_x[label] = px
    print(f"{label}: first periodic peak at NO = {px}")

# 기준(reference) = Data_90; 없으면 마지막 파일
ref_label = REF_NAME if REF_NAME in peak_x else series[-1][0]
ref_peak  = peak_x[ref_label]
if ref_peak is None:
    raise RuntimeError(f"기준 파일 {ref_label}에서 peak를 못 찾았습니다.")
print(f"\n기준(reference): {ref_label}, peak NO = {ref_peak}")

# ── 헬퍼: deg_str 추출 (Data_90_2 → "90deg_2" 로 구분) ──
def deg_label(label):
    m = re.search(r"Data_(-?\d+)(_\d+)?", label)
    if m:
        base   = f"{m.group(1)}deg"
        suffix = m.group(2) or ""
        return base + suffix
    return label

# ── 90° 데이터 꺼내기 ──
ref_entry = next((s for s in series if s[0] == ref_label), None)
if ref_entry is None:
    raise RuntimeError(f"{ref_label} 데이터를 series에서 찾을 수 없습니다.")
_, ref_x, ref_y = ref_entry

# ── 90° 정렬용 x 계산 (ref 자신은 shift=0) ──
def aligned_x(label, x):
    if peak_x[label] is not None:
        return x + (ref_peak - peak_x[label])
    return x

ref_x_al = aligned_x(ref_label, ref_x)

# ── 비교 대상: 90° 제외한 나머지 ──
others = [(lbl, x, y) for lbl, x, y in series if lbl != ref_label]

plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
saved = []

for lbl, x, y in others:
    tag      = deg_label(lbl)        # e.g. "0deg"
    ref_tag  = deg_label(ref_label)  # "90deg"
    x_al     = aligned_x(lbl, x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ax_raw, ax_al = axes

    # ── (1) 원본 ──
    ax_raw.plot(ref_x, ref_y, color="steelblue",  lw=2.2, label=f"{ref_tag} [REF]")
    ax_raw.plot(x,     y,     color="tomato",      lw=1.8, ls="--", label=tag)
    ax_raw.axvline(ref_peak, color="k", ls=":", lw=1.0, label="90deg peak")
    ax_raw.set_title(f"Raw  |  {ref_tag} vs {tag}", fontsize=12)
    ax_raw.set_xlabel("NO (sample index)")
    ax_raw.set_ylabel("Force (N)")
    ax_raw.legend(fontsize=9)
    ax_raw.grid(True, alpha=0.25)

    # 최대값 텍스트
    ax_raw.text(0.98, 0.97,
                f"{ref_tag} peak: {ref_y.max():.0f} N\n{tag} peak: {y.max():.0f} N",
                transform=ax_raw.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # ── (2) 위상 정렬 ──
    ax_al.plot(ref_x_al, ref_y, color="steelblue",  lw=2.2, label=f"{ref_tag} [REF]")
    ax_al.plot(x_al,     y,     color="tomato",      lw=1.8, ls="--", label=tag)
    ax_al.axvline(ref_peak, color="k", ls=":", lw=1.0, label="ref peak")
    ax_al.set_title(f"Phase-aligned to {ref_tag}  |  {ref_tag} vs {tag}", fontsize=12)
    ax_al.set_xlabel("Aligned NO")
    ax_al.legend(fontsize=9)
    ax_al.grid(True, alpha=0.25)

    plt.tight_layout()
    out_png = os.path.join(base_path, f"compare_0vs{tag}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_png)
    print(f"  Saved: {out_png}  |  {ref_tag} peak={ref_y.max():.0f}N  {tag} peak={y.max():.0f}N")

print(f"\nTotal {len(saved)} plots saved to {base_path}/")
