import os
import pickle
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import seaborn as sns

# OS 및 Docker(Linux) 환경 호환 한글 폰트 자동 감지 설정
def get_korean_font():
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for candidate in ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'AppleGothic', 'DejaVu Sans']:
        if candidate in available_fonts:
            return candidate
    return 'sans-serif'

KOR_FONT = get_korean_font()
plt.rc('font', family=KOR_FONT)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 110
sns.set_theme(style='whitegrid', font=KOR_FONT)
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="감귤 품종·거래단위별 모델별 일주일간 가격·거래량 예측 대시보드",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_CONFIG = {
    'TFT': {'label': 'GPU TFT (Temporal Fusion Transformer)', 'short': 'TFT', 'color': '#2A9D8F', 'marker': 'o'},
    'XGBoost': {'label': 'GPU XGBoost (Gradient Boosting)', 'short': 'XGBoost', 'color': '#E76F51', 'marker': 's'},
    'LightGBM': {'label': 'LightGBM (Leaf-wise GBDT)', 'short': 'LightGBM', 'color': '#457B9D', 'marker': '^'},
    'RandomForest': {'label': 'Random Forest (Quantile Forest)', 'short': 'RandomForest', 'color': '#8E44AD', 'marker': 'D'}
}
WEEKDAY_KOR = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}

def calc_accuracy_metrics(df_eval, m_name, prc_mult=1.0):
    sub = df_eval[df_eval['actual_vol_ton'].notnull() & df_eval['actual_prc_kg'].notnull()].copy()
    if len(sub) == 0:
        return None
    y_v = sub['actual_vol_ton'].values.astype(float)
    p_v = sub[f'{m_name}_vol_p50'].values.astype(float)
    y_p = sub['actual_prc_kg'].values.astype(float) * prc_mult
    p_p = sub[f'{m_name}_prc_p50'].values.astype(float) * prc_mult

    # 거래량 지표
    v_mae = float(np.mean(np.abs(y_v - p_v)))
    v_rmse = float(np.sqrt(np.mean((y_v - p_v) ** 2)))
    v_mape = float(np.mean(np.abs((y_v - p_v) / np.maximum(y_v, 0.5))) * 100.0)
    v_wmape = float(np.sum(np.abs(y_v - p_v)) / max(np.sum(y_v), 1e-6) * 100.0)
    v_acc = float(np.clip(100.0 - v_wmape, 0.0, 100.0))
    ss_res_v = float(np.sum((y_v - p_v) ** 2))
    ss_tot_v = float(np.sum((y_v - np.mean(y_v)) ** 2)) + 1e-8
    v_r2 = float(max(0.0, 1.0 - (ss_res_v / ss_tot_v))) if len(y_v) > 2 else np.nan

    # 가격 지표
    p_mae = float(np.mean(np.abs(y_p - p_p)))
    p_rmse = float(np.sqrt(np.mean((y_p - p_p) ** 2)))
    p_mape = float(np.mean(np.abs((y_p - p_p) / np.maximum(y_p, 100.0))) * 100.0)
    p_acc = float(np.clip(100.0 - p_mape, 0.0, 100.0))
    ss_res_p = float(np.sum((y_p - p_p) ** 2))
    ss_tot_p = float(np.sum((y_p - np.mean(y_p)) ** 2)) + 1e-8
    p_r2 = float(max(0.0, 1.0 - (ss_res_p / ss_tot_p))) if len(y_p) > 2 else np.nan

    return {
        'n_days': len(sub),
        'v_mae': v_mae, 'v_rmse': v_rmse, 'v_mape': v_mape, 'v_wmape': v_wmape, 'v_acc': v_acc, 'v_r2': v_r2,
        'p_mae': p_mae, 'p_rmse': p_rmse, 'p_mape': p_mape, 'p_acc': p_acc, 'p_r2': p_r2
    }

@st.cache_data(show_spinner="4개 모델 예측 데이터셋(2021-01-02 ~ 2026-12-31) 및 모델별 정확도를 불러오는 중입니다...")
def load_prediction_data():
    pkl_path = Path("data") / "streamlit_predictions.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"예측 데이터 파일이 존재하지 않습니다: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    df = data['all_preds_df'].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2021-01-02') & (df['date'] <= '2026-12-31')].sort_values(['group_id', 'date']).reset_index(drop=True)

    # 전체 47개 규격 종합 모델별 정확도 사전 산출
    overall_acc = {}
    for m_name in MODEL_CONFIG:
        overall_acc[m_name] = calc_accuracy_metrics(df, m_name, prc_mult=1.0)

    return (
        df,
        data['top5_units'],
        data['variety_order'],
        data['offseason_summary_df'],
        data['active_season_pairs'],
        set(pd.to_datetime(list(data['korean_holidays']))),
        data['wk_full_grid'],
        overall_acc
    )

try:
    all_preds_df, top5_units, variety_order, offseason_summary_df, active_season_pairs, korean_holidays, wk_full_grid, overall_acc_dict = load_prediction_data()
except Exception as e:
    st.error(f"데이터 로드 오류: {e}")
    st.stop()

# 상단 타이틀 헤더
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #1D3557 0%, #2A9D8F 100%); padding: 18px 24px; border-radius: 12px; color: white; margin-bottom: 16px;">
        <h2 style="margin: 0; color: white; font-weight: 800;">🍊 감귤 품종 · 거래단위별 일주일간 예상 가격 · 거래량 및 모델별 정확도 비교 대시보드</h2>
        <p style="margin: 6px 0 0 0; font-size: 14.5px; opacity: 0.94;">
            전체 기간(<b>2021-01-02 ~ 2026-12-31</b>) 중 선택한 <b>품종 / 거래단위</b>와 <b>기준 날짜(중앙)</b>를 중심으로 <b>일주일간(전후 ±3일)</b>의 <b>4개 AI 모델(TFT · XGBoost · LightGBM · Random Forest)</b>별 예상 가격·거래량과 <b>모델별 예측 정확도(Accuracy · R² · MAE · MAPE)</b>를 함께 비교합니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# 사이드바 컨트롤
with st.sidebar:
    st.header("🔍 예측 조회 조건 설정")

    avail_varieties = [v for v in variety_order if v in top5_units['variety'].unique()]
    selected_variety = st.selectbox(
        "1️⃣ 품종 선택",
        options=avail_varieties,
        index=0
    )

    sub_units = top5_units[top5_units['variety'] == selected_variety].sort_values('품종내_순위')
    unit_options = sub_units['box_unit'].tolist()
    unit_labels = {
        r['box_unit']: f"{r['box_unit']} (품종내 {r['품종내_건수비중(%)']:.1f}% · {int(r['품종내_순위'])}위)"
        for _, r in sub_units.iterrows()
    }
    selected_unit = st.selectbox(
        "2️⃣ 거래단위 선택",
        options=unit_options,
        format_func=lambda u: unit_labels.get(u, u),
        index=0
    )

    selected_gid = f"{selected_variety}_{selected_unit}"
    gid_df = all_preds_df[all_preds_df['group_id'] == selected_gid].sort_values('date').reset_index(drop=True)
    unit_weight_kg = float(gid_df['unit_weight_kg'].iloc[0]) if len(gid_df) > 0 else 5.0

    off_row = offseason_summary_df[offseason_summary_df['품종명'] == selected_variety].iloc[0]
    actual_sub = gid_df[gid_df['actual_vol_ton'].notnull()]
    last_actual_dt = actual_sub['date'].max().date() if len(actual_sub) > 0 else date(2026, 11, 11)

    st.info(
        f"**[{selected_variety}] 출하·데이터 범위 안내**\n"
        f"- **전체 조회 기간**: `2021-01-02 ~ 2026-12-31`\n"
        f"- **실제 거래 데이터 보유**: `~ {last_actual_dt.strftime('%Y-%m-%d')}` ({len(actual_sub):,}영업일)\n"
        f"- **출하기(예측 수행)**: {off_row['출하기(예측 수행 기간)']}\n"
        f"- **비수기(예측 제외)**: {off_row['계절적 비수기(예측 제외 기간)']}\n"
        f"- **단위 중량**: `{unit_weight_kg:g} kg`"
    )

    min_dt = date(2021, 1, 2)
    max_dt = date(2026, 12, 31)

    avail_dates_list = gid_df['date'].dt.date.tolist()
    default_target = last_actual_dt if last_actual_dt in avail_dates_list else (avail_dates_list[-1] if avail_dates_list else date(2026, 11, 11))

    selected_date = st.date_input(
        "3️⃣ 기준 날짜 선택 (2021-01-02 ~ 2026-12-31, 일주일 구간의 정중앙)",
        value=default_target,
        min_value=min_dt,
        max_value=max_dt
    )

    window_mode = st.radio(
        "4️⃣ 일주일(7일) 구간 구성 방식",
        options=[
            "달력 기준 7일 (선택일 중앙: D-3일 ~ D+3일)",
            "영업일 기준 7일 (선택일 중앙: 전후 ±3영업일)"
        ],
        index=0
    )

    price_unit_mode = st.radio(
        "5️⃣ 가격 그래프 표시 단위",
        options=["kg당 단가 (원/kg)", f"상자당 가격 ({selected_unit} 기준, 원)"],
        index=0
    )

    selected_models = st.multiselect(
        "6️⃣ 비교할 예측 모델 선택",
        options=list(MODEL_CONFIG.keys()),
        default=list(MODEL_CONFIG.keys()),
        format_func=lambda m: MODEL_CONFIG[m]['short']
    )
    if not selected_models:
        selected_models = list(MODEL_CONFIG.keys())

    show_actual = st.checkbox("실제 가격 · 실제 거래량 (데이터 존재 시) 함께 표시", value=True)
    show_hist_21_25 = st.checkbox("예년(2021~2025년) 동시점 가격·거래량 함께 표시", value=True)
    show_quantile_band = st.checkbox("P10 ~ P90 불확실성 예측 구간(음영/오차바) 표시", value=True)

# 선택된 날짜(selected_date)를 중앙으로 하는 7일 데이터 구성 함수
def build_7day_window_df(gid_df, selected_variety, selected_gid, target_dt, mode_str):
    target_ts = pd.Timestamp(target_dt)
    gid_indexed = gid_df.set_index('date')
    wk_hist_sub = wk_full_grid[wk_full_grid['group_id'] == selected_gid].set_index('week_idx')

    if "달력 기준" in mode_str:
        window_dates = [target_ts + pd.Timedelta(days=offset) for offset in range(-3, 4)]
    else:
        all_valid_dates = gid_df['date'].sort_values().tolist()
        if len(all_valid_dates) <= 7:
            window_dates = all_valid_dates
        else:
            diffs = [abs((d - target_ts).days) for d in all_valid_dates]
            center_idx = int(np.argmin(diffs))
            center_idx = max(3, min(len(all_valid_dates) - 4, center_idx))
            window_dates = all_valid_dates[center_idx - 3 : center_idx + 4]

    rows = []
    for d_ts in window_dates:
        dow = d_ts.dayofweek
        wk_num = min(52, (d_ts.dayofyear - 1) // 7 + 1)
        is_sun_or_hol = (dow == 6) or (d_ts in korean_holidays)
        is_active_wk = (selected_variety, wk_num) in active_season_pairs

        h_vol = float(wk_hist_sub.loc[wk_num, 'hist_wk_vol_ton']) if wk_num in wk_hist_sub.index else np.nan
        h_prc = float(wk_hist_sub.loc[wk_num, 'hist_wk_prc_kg']) if wk_num in wk_hist_sub.index else np.nan

        if d_ts in gid_indexed.index:
            r = gid_indexed.loc[d_ts]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            row_dict = r.to_dict()
            row_dict['date'] = d_ts
            has_act = pd.notnull(r.get('actual_vol_ton', np.nan))
            row_dict['day_status'] = '정상 영업일(실제+예측)' if has_act else '정상 영업일(예측)'
        else:
            if is_sun_or_hol:
                status_str = '휴장일(일요일/공휴일)'
            elif not is_active_wk:
                status_str = f'계절적 비수기({wk_num}주차 예측제외)'
            else:
                status_str = '개장일(인접 영업일 보간)'

            row_dict = {
                'date': d_ts,
                'year': d_ts.year,
                'month': d_ts.month,
                'week_idx': wk_num,
                'holiday_period': '휴장/비수기' if (is_sun_or_hol or not is_active_wk) else '평시',
                'group_id': selected_gid,
                'variety': selected_variety,
                'box_unit': selected_unit,
                'unit_weight_kg': unit_weight_kg,
                'hist_vol_21_25': 0.0 if is_sun_or_hol else h_vol,
                'hist_prc_21_25': np.nan if is_sun_or_hol else h_prc,
                'actual_vol_ton': np.nan,
                'actual_prc_kg': np.nan,
                'is_future': d_ts > pd.Timestamp('2026-11-11'),
                'day_status': status_str
            }
            if not is_sun_or_hol and is_active_wk and len(gid_df) > 0:
                nearest_idx = (gid_df['date'] - d_ts).abs().idxmin()
                nr = gid_df.loc[nearest_idx]
                for m_name in MODEL_CONFIG:
                    for suf in ['vol_p10', 'vol_p50', 'vol_p90', 'prc_p10', 'prc_p50', 'prc_p90']:
                        row_dict[f'{m_name}_{suf}'] = float(nr[f'{m_name}_{suf}'])
            else:
                for m_name in MODEL_CONFIG:
                    for suf in ['vol_p10', 'vol_p50', 'vol_p90']:
                        row_dict[f'{m_name}_{suf}'] = 0.0 if is_sun_or_hol else np.nan
                    for suf in ['prc_p10', 'prc_p50', 'prc_p90']:
                        row_dict[f'{m_name}_{suf}'] = np.nan

        row_dict['요일'] = WEEKDAY_KOR[dow]
        row_dict['date_label'] = f"{d_ts.strftime('%m/%d')}({WEEKDAY_KOR[dow]})"
        row_dict['is_center'] = (d_ts == target_ts)
        rows.append(row_dict)

    w_df = pd.DataFrame(rows)
    if not w_df['is_center'].any() and len(w_df) > 0:
        w_df.loc[len(w_df) // 2, 'is_center'] = True
    return w_df

win_df = build_7day_window_df(gid_df, selected_variety, selected_gid, selected_date, window_mode)
center_idx = int(win_df.index[win_df['is_center']][0])
center_row = win_df.loc[center_idx]
center_date_str = center_row['date'].strftime('%Y-%m-%d') + f" ({center_row['요일']})"

prc_mult = unit_weight_kg if "상자당" in price_unit_mode else 1.0
prc_unit_str = "원/상자" if "상자당" in price_unit_mode else "원/kg"
has_any_actual = show_actual and (win_df['actual_vol_ton'].notnull().any() or win_df['actual_prc_kg'].notnull().any())

# 선택 품종·단위 및 현재 7일 구간의 모델별 정확도 계산
group_acc_dict = {}
window_acc_dict = {}
for m_name in MODEL_CONFIG:
    group_acc_dict[m_name] = calc_accuracy_metrics(gid_df, m_name, prc_mult=prc_mult)
    window_acc_dict[m_name] = calc_accuracy_metrics(win_df, m_name, prc_mult=prc_mult)

# 상태 알림 배너
if "비수기" in str(center_row.get('day_status', '')):
    st.warning(
        f"⚠️ 선택한 날짜 **{center_date_str}** ({int(center_row['week_idx'])}주차)는 **{selected_variety}** 품종의 **계절적 비수기(거래 미미 예측 제외 구간)**입니다. "
        f"사이드바의 출하기 기간({off_row['출하기(예측 수행 기간)']}) 내 날짜를 선택하시면 정상적인 일주일 예측 그래프를 확인하실 수 있습니다."
    )
elif "휴장일" in str(center_row.get('day_status', '')):
    st.info(
        f"ℹ️ 선택한 중앙 날짜 **{center_date_str}**은 **도매시장 경매 휴장일(일요일/법정공휴일)**입니다. "
        f"해당 날짜를 중간으로 전후 일주일간의 개장 영업일 실제·예상 가격 및 거래량을 함께 출력합니다."
    )

# 1. 상단 KPI 요약 카드 (실제값 + 예년값 + 4개 모델별 예상값 및 정확도 요약)
st.subheader(f"📌 [{selected_variety} - {selected_unit}] 기준일({center_date_str}) 및 일주일간({win_df['date'].min().strftime('%Y-%m-%d')} ~ {win_df['date'].max().strftime('%Y-%m-%d')}) 요약")

n_extra_kpis = (1 if has_any_actual else 0) + 1
kpi_cols = st.columns(len(selected_models) + n_extra_kpis)
col_cursor = 0

h_vol_c = center_row['hist_vol_21_25'] if pd.notnull(center_row['hist_vol_21_25']) and center_row['hist_vol_21_25'] > 0 else win_df['hist_vol_21_25'].replace(0, np.nan).mean()
h_prc_c = (center_row['hist_prc_21_25'] if pd.notnull(center_row['hist_prc_21_25']) else win_df['hist_prc_21_25'].mean()) * prc_mult

act_vol_c = center_row['actual_vol_ton'] if pd.notnull(center_row.get('actual_vol_ton')) else win_df['actual_vol_ton'].mean()
act_prc_c = (center_row['actual_prc_kg'] if pd.notnull(center_row.get('actual_prc_kg')) else win_df['actual_prc_kg'].mean()) * prc_mult

if has_any_actual:
    with kpi_cols[col_cursor]:
        act_lbl = "기준일 실제" if pd.notnull(center_row.get('actual_prc_kg')) else "기간내 실제평균"
        st.metric(
            label=f"🎯 실제 경매 결과 ({act_lbl})",
            value=f"{act_prc_c:,.0f} {prc_unit_str}" if pd.notnull(act_prc_c) else "-",
            delta=f"실제 거래량: {act_vol_c:,.1f} 톤/일" if pd.notnull(act_vol_c) else "-",
            delta_color="off"
        )
        act_days_cnt = int(win_df['actual_vol_ton'].notnull().sum())
        st.caption(f"조회 7일 중 실제 거래일: {act_days_cnt}일")
    col_cursor += 1

with kpi_cols[col_cursor]:
    st.metric(
        label="📊 예년(2021~2025) 동시점 평균",
        value=f"{h_prc_c:,.0f} {prc_unit_str}" if pd.notnull(h_prc_c) else "-",
        delta=f"예년 물량: {h_vol_c:,.1f} 톤/일" if pd.notnull(h_vol_c) else "-",
        delta_color="off"
    )
    st.caption(f"해당 시점({int(center_row['week_idx'])}주차) 5개년 기준")
col_cursor += 1

for m_name in selected_models:
    m_cfg = MODEL_CONFIG[m_name]
    g_acc = group_acc_dict.get(m_name)
    with kpi_cols[col_cursor]:
        c_vol = center_row.get(f'{m_name}_vol_p50', np.nan)
        c_prc = center_row.get(f'{m_name}_prc_p50', np.nan)
        label_suffix = "기준일"
        if pd.isnull(c_prc) or c_vol == 0.0:
            c_vol = win_df[f'{m_name}_vol_p50'].replace(0, np.nan).mean()
            c_prc = win_df[f'{m_name}_prc_p50'].mean()
            label_suffix = "일주일 평균"
        c_prc_disp = c_prc * prc_mult if pd.notnull(c_prc) else np.nan
        if has_any_actual and pd.notnull(act_prc_c) and act_prc_c > 0 and pd.notnull(c_prc_disp):
            diff_str = f"{(c_prc_disp / act_prc_c - 1.0)*100.0:+.1f}% vs 실제"
        elif pd.notnull(c_prc_disp) and pd.notnull(h_prc_c) and h_prc_c > 0:
            diff_str = f"{(c_prc_disp / h_prc_c - 1.0)*100.0:+.1f}% vs 예년"
        else:
            diff_str = "-"
        st.metric(
            label=f"🤖 {m_cfg['short']} 예상 ({label_suffix})",
            value=f"{c_prc_disp:,.0f} {prc_unit_str}" if pd.notnull(c_prc_disp) else "-",
            delta=f"예상 물량: {c_vol:,.1f} 톤 ({diff_str})" if pd.notnull(c_vol) else "-"
        )
        if g_acc:
            st.caption(f"✅ 정확도: 가격 **{g_acc['p_acc']:.1f}%** · 물량 **{g_acc['v_acc']:.1f}%** (R²={g_acc['v_r2']:.3f})")
    col_cursor += 1

st.divider()

# 2. 메인 비교 그래프: 선택 날짜(중앙) 기준 일주일간 모델별 예상 거래량 & 예상 가격 (실제값 + 예년값 포함)
st.subheader(f"📈 1. 선택 날짜({center_date_str}) 중심 일주일간 실제 vs 예년(21~25) vs 모델별 예상 거래량 · 가격 비교 그래프")

fig_comp, (ax_vol, ax_prc) = plt.subplots(1, 2, figsize=(18.5, 6.2))
x_idx = np.arange(len(win_df))
x_labels = [
    f"★ {r['date_label']}\n[선택일]" if r['is_center'] else (f"{r['date_label']}\n(휴장)" if "휴장" in str(r['day_status']) else r['date_label'])
    for _, r in win_df.iterrows()
]

for ax in (ax_vol, ax_prc):
    ax.axvspan(center_idx - 0.42, center_idx + 0.42, color='#FFD166', alpha=0.30, zorder=0, label='선택 기준일(중앙)')
    ax.axvline(center_idx, color='#D62828', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)

# (좌) 일주일간 거래량 비교 그래프
n_bars = len(selected_models) + (1 if show_hist_21_25 else 0) + (1 if has_any_actual else 0)
bar_w = min(0.15, 0.78 / max(1, n_bars))
offsets = np.linspace(-bar_w * (n_bars - 1) / 2, bar_w * (n_bars - 1) / 2, n_bars)
b_pos = 0

if show_hist_21_25:
    ax_vol.bar(
        x_idx + offsets[b_pos],
        win_df['hist_vol_21_25'].fillna(0.0),
        width=bar_w, color='#8ECAE6', edgecolor='#219EBC', alpha=0.85,
        label='예년(2021~2025) 동시점 물량(톤)'
    )
    b_pos += 1

if has_any_actual:
    act_v_arr = win_df['actual_vol_ton'].fillna(0.0).values
    ax_vol.bar(
        x_idx + offsets[b_pos],
        act_v_arr,
        width=bar_w, color='#1D3557', edgecolor='black', linewidth=0.8, alpha=0.92,
        label='실제 거래량(톤)'
    )
    ax_vol.plot(
        x_idx + offsets[b_pos],
        win_df['actual_vol_ton'].values,
        color='#1D3557', marker='*', linewidth=2.0, markersize=8, zorder=5
    )
    if act_v_arr[center_idx] > 0:
        ax_vol.text(
            center_idx + offsets[b_pos],
            act_v_arr[center_idx] * 1.04,
            f"실제 {act_v_arr[center_idx]:,.1f}톤",
            ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#1D3557'
        )
    b_pos += 1

for m_name in selected_models:
    m_cfg = MODEL_CONFIG[m_name]
    g_acc = group_acc_dict.get(m_name)
    acc_tag = f" (정확도 {g_acc['v_acc']:.1f}%)" if g_acc else ""
    v_vals = win_df[f'{m_name}_vol_p50'].fillna(0.0).values
    ax_vol.bar(
        x_idx + offsets[b_pos],
        v_vals,
        width=bar_w, color=m_cfg['color'], alpha=0.90,
        label=f"{m_cfg['short']} 예상 물량{acc_tag}"
    )
    ax_vol.plot(
        x_idx + offsets[b_pos],
        np.where(v_vals > 0, v_vals, np.nan),
        color=m_cfg['color'], marker=m_cfg['marker'], linewidth=1.5, markersize=5
    )
    if v_vals[center_idx] > 0:
        ax_vol.text(
            center_idx + offsets[b_pos],
            v_vals[center_idx] * 1.02,
            f"{v_vals[center_idx]:,.1f}톤",
            ha='center', va='bottom', fontsize=8, fontweight='bold', color=m_cfg['color']
        )
    b_pos += 1

ax_vol.set_title(f"[{selected_variety} - {selected_unit}] 선택일({center_row['date_label']}) 중심 일주일간 실제 · 예년 · 모델별 예상 거래량 (톤/일)", fontsize=12, fontweight='bold', pad=10)
ax_vol.set_xticks(x_idx)
ax_vol.set_xticklabels(x_labels, fontsize=9.5, fontweight='bold')
ax_vol.set_ylabel("거래량 (톤/일)", fontsize=10, fontweight='bold')
ax_vol.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.1f}'))
max_vol_candidates = [win_df[f'{m}_vol_p90'].max() for m in selected_models] + [win_df['hist_vol_21_25'].max(), win_df['actual_vol_ton'].max(), 1.0]
max_vol_y = np.nanmax([v for v in max_vol_candidates if pd.notnull(v)])
if np.isfinite(max_vol_y) and max_vol_y > 0:
    ax_vol.set_ylim(0, max_vol_y * 1.28)
ax_vol.legend(loc='upper right', fontsize=8.0, ncol=2)

# (우) 일주일간 가격 비교 그래프
if show_hist_21_25:
    ax_prc.plot(
        x_idx,
        win_df['hist_prc_21_25'] * prc_mult,
        color='#6C757D', linestyle=':', linewidth=2.0, marker='D', markersize=5.8,
        label=f'예년(2021~2025) 동시점 가격({prc_unit_str})'
    )

if has_any_actual:
    act_p_vals = win_df['actual_prc_kg'] * prc_mult
    ax_prc.plot(
        x_idx,
        act_p_vals,
        color='#1D3557', linestyle='-', linewidth=2.8, marker='*', markersize=9.0, zorder=6,
        label=f'실제 가격({prc_unit_str})'
    )
    if pd.notnull(act_p_vals.iloc[center_idx]):
        ax_prc.annotate(
            f"실제: {act_p_vals.iloc[center_idx]:,.0f}원",
            xy=(center_idx, act_p_vals.iloc[center_idx]),
            xytext=(0, -15), textcoords="offset points",
            ha='center', fontsize=8.8, fontweight='bold', color='#1D3557'
        )

for m_name in selected_models:
    m_cfg = MODEL_CONFIG[m_name]
    g_acc = group_acc_dict.get(m_name)
    acc_tag = f" (정확도 {g_acc['p_acc']:.1f}%)" if g_acc else ""
    p50_vals = win_df[f'{m_name}_prc_p50'] * prc_mult
    p10_vals = win_df[f'{m_name}_prc_p10'] * prc_mult
    p90_vals = win_df[f'{m_name}_prc_p90'] * prc_mult
    ax_prc.plot(
        x_idx, p50_vals,
        color=m_cfg['color'], linestyle='--', linewidth=2.2, marker=m_cfg['marker'], markersize=6.5,
        label=f"{m_cfg['short']} 예상 가격{acc_tag}"
    )
    if show_quantile_band:
        ax_prc.fill_between(x_idx, p10_vals, p90_vals, color=m_cfg['color'], alpha=0.11)
    if pd.notnull(p50_vals.iloc[center_idx]):
        ax_prc.annotate(
            f"{m_cfg['short']}: {p50_vals.iloc[center_idx]:,.0f}원",
            xy=(center_idx, p50_vals.iloc[center_idx]),
            xytext=(0, 8), textcoords="offset points",
            ha='center', fontsize=8.3, fontweight='bold', color=m_cfg['color']
        )

ax_prc.set_title(f"[{selected_variety} - {selected_unit}] 선택일({center_row['date_label']}) 중심 일주일간 실제 · 예년 · 모델별 예상 가격 ({prc_unit_str})", fontsize=12, fontweight='bold', pad=10)
ax_prc.set_xticks(x_idx)
ax_prc.set_xticklabels(x_labels, fontsize=9.5, fontweight='bold')
ax_prc.set_ylabel(f"가격 ({prc_unit_str})", fontsize=10, fontweight='bold')
ax_prc.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax_prc.legend(loc='upper right', fontsize=8.0, ncol=2)

plt.tight_layout()
st.pyplot(fig_comp)
plt.close(fig_comp)

# 3. 4개 모델 개별(2x2) 일주일간 예상 거래량(막대) + 예상 가격(선) + 모델별 정확도 표시 상세 그래프
st.subheader(f"🔍 2. 각각의 모델별(TFT · XGBoost · LightGBM · Random Forest) 일주일간 실제 vs 예상 거래량 · 가격 및 정확도 상세 그래프")

ncols_m = 2
nrows_m = int(np.ceil(len(selected_models) / ncols_m))
fig_each, axes_each = plt.subplots(nrows_m, ncols_m, figsize=(18.5, nrows_m * 5.1))
axes_each = np.array(axes_each).reshape(-1)

for idx_m, m_name in enumerate(selected_models):
    ax1 = axes_each[idx_m]
    m_cfg = MODEL_CONFIG[m_name]
    g_acc = group_acc_dict.get(m_name)
    w_acc = window_acc_dict.get(m_name)

    v_p10 = win_df[f'{m_name}_vol_p10'].fillna(0.0).values
    v_p50 = win_df[f'{m_name}_vol_p50'].fillna(0.0).values
    v_p90 = win_df[f'{m_name}_vol_p90'].fillna(0.0).values

    p_p10 = (win_df[f'{m_name}_prc_p10'] * prc_mult).values
    p_p50 = (win_df[f'{m_name}_prc_p50'] * prc_mult).values
    p_p90 = (win_df[f'{m_name}_prc_p90'] * prc_mult).values

    act_v_sub = win_df['actual_vol_ton'].fillna(0.0).values
    act_p_sub = (win_df['actual_prc_kg'] * prc_mult).values

    ax1.axvspan(center_idx - 0.42, center_idx + 0.42, color='#FFD166', alpha=0.30, zorder=0, label='선택 기준일(중앙)')
    ax1.axvline(center_idx, color='#D62828', linestyle='--', linewidth=1.4, zorder=1)

    if show_hist_21_25 and has_any_actual:
        w_b = 0.24
        ax1.bar(x_idx - w_b, win_df['hist_vol_21_25'].fillna(0.0), width=w_b, color='#8ECAE6', edgecolor='#219EBC', alpha=0.82, label='예년(21~25) 거래량(톤)')
        ax1.bar(x_idx, act_v_sub, width=w_b, color='#1D3557', alpha=0.90, label='실제 거래량(톤)')
        ax1.bar(x_idx + w_b, v_p50, width=w_b, color=m_cfg['color'], alpha=0.90, label=f"{m_cfg['short']} 예상 거래량(톤)")
        pred_bar_x = x_idx + w_b
        act_bar_x = x_idx
    elif has_any_actual:
        w_b = 0.32
        ax1.bar(x_idx - w_b/2, act_v_sub, width=w_b, color='#1D3557', alpha=0.90, label='실제 거래량(톤)')
        ax1.bar(x_idx + w_b/2, v_p50, width=w_b, color=m_cfg['color'], alpha=0.90, label=f"{m_cfg['short']} 예상 거래량(톤)")
        pred_bar_x = x_idx + w_b/2
        act_bar_x = x_idx - w_b/2
    elif show_hist_21_25:
        w_b = 0.32
        ax1.bar(x_idx - w_b/2, win_df['hist_vol_21_25'].fillna(0.0), width=w_b, color='#8ECAE6', edgecolor='#219EBC', alpha=0.82, label='예년(21~25) 거래량(톤)')
        ax1.bar(x_idx + w_b/2, v_p50, width=w_b, color=m_cfg['color'], alpha=0.90, label=f"{m_cfg['short']} 예상 거래량(톤)")
        pred_bar_x = x_idx + w_b/2
        act_bar_x = None
    else:
        ax1.bar(x_idx, v_p50, width=0.45, color=m_cfg['color'], alpha=0.90, label=f"{m_cfg['short']} 예상 거래량(톤)")
        pred_bar_x = x_idx
        act_bar_x = None

    if show_quantile_band:
        valid_q = v_p50 > 0
        if valid_q.any():
            yerr_low = np.maximum(0.0, v_p50[valid_q] - v_p10[valid_q])
            yerr_high = np.maximum(0.0, v_p90[valid_q] - v_p50[valid_q])
            ax1.errorbar(pred_bar_x[valid_q], v_p50[valid_q], yerr=[yerr_low, yerr_high], fmt='none', ecolor='#264653', elinewidth=1.3, capsize=3.0, label='예상물량 P10~P90')

    for xi, bv in zip(pred_bar_x, v_p50):
        if bv > 0:
            ax1.text(xi, bv * 1.02, f"예상\n{bv:,.1f}t", ha='center', va='bottom', fontsize=7.3, fontweight='bold', color=m_cfg['color'])

    if has_any_actual and act_bar_x is not None:
        for xi, av in zip(act_bar_x, act_v_sub):
            if av > 0:
                ax1.text(xi, av * 1.02, f"실제\n{av:,.1f}t", ha='center', va='bottom', fontsize=7.3, fontweight='bold', color='#1D3557')

    acc_title_str = ""
    if g_acc:
        acc_title_str = f"\n[규격 정확도 — 가격: {g_acc['p_acc']:.1f}% (MAPE {g_acc['p_mape']:.1f}%) | 거래량: {g_acc['v_acc']:.1f}% (R²={g_acc['v_r2']:.3f})]"
        if w_acc and w_acc['n_days'] >= 1:
            acc_title_str += f" · [조회 7일 정확도 — 가격: {w_acc['p_acc']:.1f}% | 물량: {w_acc['v_acc']:.1f}%]"

    ax1.set_title(f"{idx_m + 1}. {m_cfg['label']}{acc_title_str}", fontsize=10.5, fontweight='bold', pad=10)
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_labels, fontsize=8.8, fontweight='bold')
    ax1.set_ylabel("거래량 (톤/일)", color=m_cfg['color'], fontsize=9.5, fontweight='bold')
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.1f}'))
    max_v_local = np.nanmax([
        np.nanmax(v_p90) if np.isfinite(v_p90).any() else 1.0,
        win_df['hist_vol_21_25'].max() if show_hist_21_25 else 1.0,
        np.nanmax(act_v_sub) if has_any_actual else 1.0,
        1.0
    ])
    ax1.set_ylim(0, max_v_local * 1.38)
    ax1.grid(axis='y', linestyle='--', alpha=0.35)
    ax1.grid(axis='x', visible=False)

    ax2 = ax1.twinx()
    if show_hist_21_25:
        ax2.plot(x_idx, win_df['hist_prc_21_25'] * prc_mult, color='#6C757D', linestyle=':', marker='D', markersize=4.8, linewidth=1.6, label=f'예년(21~25) 가격({prc_unit_str})')
    if has_any_actual:
        ax2.plot(x_idx, act_p_sub, color='#1D3557', linestyle='-', marker='*', markersize=8.0, linewidth=2.3, zorder=5, label=f'실제 가격({prc_unit_str})')
        for xi, apv in enumerate(act_p_sub):
            if np.isfinite(apv):
                ax2.text(xi, apv * 0.965, f"실제 {apv:,.0f}원", ha='center', va='top', fontsize=7.5, fontweight='bold', color='#1D3557')

    ax2.plot(x_idx, p_p50, color='#D62828', linestyle='--', marker='s', markersize=5.8, linewidth=2.2, zorder=6, label=f"{m_cfg['short']} 예상 가격({prc_unit_str})")
    if show_quantile_band:
        ax2.fill_between(x_idx, p_p10, p_p90, color='#D62828', alpha=0.13, label='예상가격 P10~P90')

    for xi, pv in enumerate(p_p50):
        if np.isfinite(pv):
            ax2.text(xi, pv * 1.025, f"예상 {pv:,.0f}원", ha='center', va='bottom', fontsize=7.8, fontweight='bold', color='#D62828')

    ax2.set_ylabel(f"가격 ({prc_unit_str})", color='#D62828', fontsize=9.5, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#D62828')
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax2.grid(False)

    all_p_vals = np.concatenate([
        p_p10[np.isfinite(p_p10)],
        p_p90[np.isfinite(p_p90)],
        act_p_sub[np.isfinite(act_p_sub)] if has_any_actual else np.array([]),
        (win_df['hist_prc_21_25'] * prc_mult).dropna().values if show_hist_21_25 else np.array([])
    ])
    if len(all_p_vals) > 0:
        ax2.set_ylim(max(0, np.min(all_p_vals) * 0.72), np.max(all_p_vals) * 1.25)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=7.2, ncol=2)

for j in range(len(selected_models), len(axes_each)):
    axes_each[j].set_visible(False)

plt.tight_layout()
st.pyplot(fig_each)
plt.close(fig_each)

st.divider()

# 4. 모델별 가격 및 거래량 예측 정확도 비교 섹션 (그래프 + 상세 요약표)
st.subheader(f"🎯 3. 모델별 가격 및 거래량 예측 정확도(Accuracy · R² · MAE · MAPE) 비교 분석")

acc_rows = []
for m_name in selected_models:
    m_cfg = MODEL_CONFIG[m_name]
    g_acc = group_acc_dict.get(m_name)
    w_acc = window_acc_dict.get(m_name)
    o_acc = overall_acc_dict.get(m_name)

    if g_acc and o_acc:
        acc_rows.append({
            '모델명': m_cfg['short'],
            '모델 상세': m_cfg['label'],
            f'[{selected_variety}-{selected_unit}] 가격 정확도(%)': f"{g_acc['p_acc']:.2f}%",
            f'[{selected_variety}-{selected_unit}] 가격 MAE({prc_unit_str})': f"{g_acc['p_mae']:,.0f} 원",
            f'[{selected_variety}-{selected_unit}] 가격 MAPE(%)': f"{g_acc['p_mape']:.2f}%",
            f'[{selected_variety}-{selected_unit}] 거래량 정확도(%)': f"{g_acc['v_acc']:.2f}%",
            f'[{selected_variety}-{selected_unit}] 거래량 R²': f"{g_acc['v_r2']:.4f}" if pd.notnull(g_acc['v_r2']) else "-",
            f'[{selected_variety}-{selected_unit}] 거래량 MAE(톤/일)': f"{g_acc['v_mae']:,.2f} 톤",
            '조회 7일 가격 정확도(%)': f"{w_acc['p_acc']:.2f}% (MAE {w_acc['p_mae']:,.0f}원)" if (w_acc and w_acc['n_days'] >= 1) else "미래구간(실제없음)",
            '조회 7일 거래량 정확도(%)': f"{w_acc['v_acc']:.2f}% (MAE {w_acc['v_mae']:,.2f}톤)" if (w_acc and w_acc['n_days'] >= 1) else "미래구간(실제없음)",
            '전체(47개 규격) 가격 정확도(%)': f"{o_acc['p_acc']:.2f}% (R²={o_acc['p_r2']:.4f})",
            '전체(47개 규격) 거래량 R² / 정확도': f"R²={o_acc['v_r2']:.4f} ({o_acc['v_acc']:.2f}%)",
            '_g_p_acc': g_acc['p_acc'],
            '_g_v_acc': g_acc['v_acc'],
            '_o_p_acc': o_acc['p_acc'],
            '_o_v_r2_pct': o_acc['v_r2'] * 100.0,
            '_w_p_acc': w_acc['p_acc'] if (w_acc and w_acc['n_days'] >= 1) else np.nan,
            '_w_v_acc': w_acc['v_acc'] if (w_acc and w_acc['n_days'] >= 1) else np.nan,
            '_color': m_cfg['color']
        })

acc_df = pd.DataFrame(acc_rows)

if len(acc_df) > 0:
    fig_acc, (ax_ap, ax_av) = plt.subplots(1, 2, figsize=(18, 4.8))
    xm = np.arange(len(acc_df))
    has_win_acc = acc_df['_w_p_acc'].notnull().any()
    bw = 0.25 if has_win_acc else 0.34

    # (좌) 모델별 가격 예측 정확도(%) 비교
    if has_win_acc:
        b1 = ax_ap.bar(xm - bw, acc_df['_g_p_acc'], width=bw, color='#2A9D8F', alpha=0.90, label=f'선택 규격({selected_variety}-{selected_unit}) 가격 정확도(%)')
        b2 = ax_ap.bar(xm, acc_df['_w_p_acc'].fillna(0.0), width=bw, color='#E76F51', alpha=0.90, label='현재 조회 7일 구간 가격 정확도(%)')
        b3 = ax_ap.bar(xm + bw, acc_df['_o_p_acc'], width=bw, color='#457B9D', alpha=0.85, label='전체 47개 규격 평균 가격 정확도(%)')
        for bars_grp in (b1, b2, b3):
            for b in bars_grp:
                h = b.get_height()
                if h > 0:
                    ax_ap.text(b.get_x() + b.get_width()/2, h + 0.4, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    else:
        b1 = ax_ap.bar(xm - bw/2, acc_df['_g_p_acc'], width=bw, color='#2A9D8F', alpha=0.90, label=f'선택 규격({selected_variety}-{selected_unit}) 가격 정확도(%)')
        b3 = ax_ap.bar(xm + bw/2, acc_df['_o_p_acc'], width=bw, color='#457B9D', alpha=0.85, label='전체 47개 규격 평균 가격 정확도(%)')
        for bars_grp in (b1, b3):
            for b in bars_grp:
                h = b.get_height()
                if h > 0:
                    ax_ap.text(b.get_x() + b.get_width()/2, h + 0.4, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax_ap.set_title(f"1. 모델별 가격 예측 정확도 (100 - MAPE, %) 비교", fontsize=12, fontweight='bold', pad=10)
    ax_ap.set_xticks(xm)
    ax_ap.set_xticklabels(acc_df['모델명'], fontsize=10, fontweight='bold')
    ax_ap.set_ylabel("가격 예측 정확도 (%)", fontsize=10, fontweight='bold')
    ax_ap.set_ylim(50, 105)
    ax_ap.legend(loc='lower right', fontsize=8.5)

    # (우) 모델별 거래량 예측 정확도(%) 및 결정계수(R²) 비교
    if has_win_acc:
        bv1 = ax_av.bar(xm - bw, acc_df['_g_v_acc'], width=bw, color='#2A9D8F', alpha=0.90, label=f'선택 규격({selected_variety}-{selected_unit}) 거래량 정확도(100-WMAPE, %)')
        bv2 = ax_av.bar(xm, acc_df['_w_v_acc'].fillna(0.0), width=bw, color='#E76F51', alpha=0.90, label='현재 조회 7일 구간 거래량 정확도(%)')
        bv3 = ax_av.bar(xm + bw, acc_df['_o_v_r2_pct'], width=bw, color='#8E44AD', alpha=0.85, label='전체 47개 규격 거래량 설명력(R²×100, %)')
        for bars_grp in (bv1, bv2, bv3):
            for b in bars_grp:
                h = b.get_height()
                if h > 0:
                    ax_av.text(b.get_x() + b.get_width()/2, h + 0.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    else:
        bv1 = ax_av.bar(xm - bw/2, acc_df['_g_v_acc'], width=bw, color='#2A9D8F', alpha=0.90, label=f'선택 규격({selected_variety}-{selected_unit}) 거래량 정확도(100-WMAPE, %)')
        bv3 = ax_av.bar(xm + bw/2, acc_df['_o_v_r2_pct'], width=bw, color='#8E44AD', alpha=0.85, label='전체 47개 규격 거래량 설명력(R²×100, %)')
        for bars_grp in (bv1, bv3):
            for b in bars_grp:
                h = b.get_height()
                if h > 0:
                    ax_av.text(b.get_x() + b.get_width()/2, h + 0.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax_av.set_title(f"2. 모델별 거래량 예측 정확도 (100 - WMAPE % 및 결정계수 R²) 비교", fontsize=12, fontweight='bold', pad=10)
    ax_av.set_xticks(xm)
    ax_av.set_xticklabels(acc_df['모델명'], fontsize=10, fontweight='bold')
    ax_av.set_ylabel("거래량 정확도 / 결정계수 (%)", fontsize=10, fontweight='bold')
    ax_av.set_ylim(40, 106)
    ax_av.legend(loc='lower right', fontsize=8.5)

    plt.tight_layout()
    st.pyplot(fig_acc)
    plt.close(fig_acc)

    disp_acc_cols = [c for c in acc_df.columns if not c.startswith('_')]
    st.dataframe(acc_df[disp_acc_cols], use_container_width=True, hide_index=True)

st.divider()

# 5. 일주일간 날짜별 · 모델별 상세 예측 데이터 테이블 출력 (실제 가격·거래량 포함)
st.subheader(f"📋 4. [{selected_variety} - {selected_unit}] 선택일({center_date_str}) 중심 일주일간 실제 · 예년 · 모델별 예상 가격 및 거래량 상세 비교표")

table_rows = []
for _, r in win_df.iterrows():
    row_item = {
        '기준구분': '★ 선택 기준일(중앙)' if r['is_center'] else '-',
        '날짜': r['date'].strftime('%Y-%m-%d'),
        '요일': r['요일'],
        '영업상태': r['day_status'],
        '명절구간': r['holiday_period'],
        '실제 거래량': f"{r['actual_vol_ton']:,.1f} 톤" if pd.notnull(r.get('actual_vol_ton')) else "-",
        f'실제 가격({prc_unit_str})': f"{r['actual_prc_kg']*prc_mult:,.0f} 원" if pd.notnull(r.get('actual_prc_kg')) else "-",
        '예년(21~25) 물량': f"{r['hist_vol_21_25']:,.1f} 톤" if pd.notnull(r['hist_vol_21_25']) and r['hist_vol_21_25'] > 0 else "-",
        f'예년(21~25) 가격({prc_unit_str})': f"{r['hist_prc_21_25']*prc_mult:,.0f} 원" if pd.notnull(r['hist_prc_21_25']) else "-"
    }

    for m_name in selected_models:
        m_short = MODEL_CONFIG[m_name]['short']
        vp50 = r.get(f'{m_name}_vol_p50', np.nan)
        pp50 = r.get(f'{m_name}_prc_p50', np.nan)
        pp10 = r.get(f'{m_name}_prc_p10', np.nan)
        pp90 = r.get(f'{m_name}_prc_p90', np.nan)
        row_item[f'{m_short} 예상물량'] = f"{vp50:,.1f} 톤" if pd.notnull(vp50) and vp50 > 0 else "휴장/제외"
        row_item[f'{m_short} 예상가격({prc_unit_str})'] = f"{pp50*prc_mult:,.0f} 원 ({pp10*prc_mult:,.0f}~{pp90*prc_mult:,.0f})" if pd.notnull(pp50) else "휴장/제외"

    table_rows.append(row_item)

table_df = pd.DataFrame(table_rows)
st.dataframe(table_df, use_container_width=True, hide_index=True)

csv_bytes = table_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
st.download_button(
    label="📥 현재 조회된 일주일간 실제·예년·모델별 예측 결과 CSV 다운로드",
    data=csv_bytes,
    file_name=f"citrus_7day_forecast_{selected_gid}_{selected_date.strftime('%Y%m%d')}.csv",
    mime="text/csv"
)
