"""AI 모델 학습 실험 비교 대시보드 (Streamlit Community Cloud 배포용).

GitHub 저장소에 app.py, requirements.txt, ml_experiments.csv를 같은 폴더에 올리고
Streamlit Community Cloud에서 Main file path를 app.py로 지정해 배포합니다.
로컬 실행: python -m streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

HERE = Path(__file__).parent

st.set_page_config(page_title="AI 모델 실험 비교 대시보드", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(HERE / "ml_experiments.csv")
    df["label"] = df["run_id"] + " · " + df["model"] + " (" + df["optimizer"] + ")"
    df["loss_gap"] = (df["val_loss"] - df["train_loss"]).round(4)
    return df


def min_max_scale(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """0~100 범위로 정규화합니다. 값이 모두 같으면 100점을 부여합니다."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(100.0, index=series.index)
    scaled = (series - mn) / (mx - mn) * 100
    return scaled if higher_is_better else (100.0 - scaled)


# ---------------------------------------------------------------------------
# 1. 헤더 및 데이터 로드
# ---------------------------------------------------------------------------
st.title("🧪 AI 모델 실험 비교 & 최적 모델 추천 대시보드")
st.caption(
    "이미지 분류 모델 8개 실험 로그(ml_experiments.csv) · "
    "학습 곡선(과적합 진단), 추론 속도 vs 정확도 트레이드오프, 가중치 기반 모델 추천"
)

try:
    data = load_data()
except FileNotFoundError:
    st.error("app.py와 같은 폴더에 ml_experiments.csv 파일을 올려 주세요.")
    st.stop()

# ---------------------------------------------------------------------------
# 2. 사이드바 필터 & 추천 가중치 설정
# ---------------------------------------------------------------------------
st.sidebar.header("실험 필터")
all_models = sorted(data["model"].unique().tolist())
selected_models = st.sidebar.multiselect("모델 아키텍처", all_models, default=all_models)

all_opts = sorted(data["optimizer"].unique().tolist())
selected_opts = st.sidebar.multiselect("옵티마이저", all_opts, default=all_opts)

max_latency = st.sidebar.slider(
    "허용 최대 추론 시간 (ms)",
    min_value=float(data["latency_ms"].min()),
    max_value=float(data["latency_ms"].max()),
    value=float(data["latency_ms"].max()),
    step=0.5,
)

epoch_range = st.sidebar.slider("에포크 범위", 1, int(data["epoch"].max()), (1, int(data["epoch"].max())))

st.sidebar.divider()
st.sidebar.header("최적 모델 추천 가중치")
acc_weight = st.sidebar.slider(
    "정확도 비중 (%) — 나머지는 추론 속도 비중",
    min_value=0,
    max_value=100,
    value=70,
    step=10,
)
speed_weight = 100 - acc_weight
st.sidebar.caption(f"현재 기준: 정확도 **{acc_weight}%** · 추론 속도 **{speed_weight}%**")

# 필터 적용
view = data.loc[
    data["model"].isin(selected_models)
    & data["optimizer"].isin(selected_opts)
    & (data["latency_ms"] <= max_latency)
    & data["epoch"].between(epoch_range[0], epoch_range[1])
].copy()

if view.empty:
    st.warning("조건에 맞는 실험 데이터가 없습니다. 필터 범위를 넓혀 보세요.")
    st.stop()

# 각 실험별 최종(또는 최고) 요약 테이블 생성
last_epoch = (
    view.sort_values("epoch")
    .groupby("run_id", as_index=False)
    .last()
)
best_acc = view.groupby("run_id", as_index=False)["val_acc"].max().rename(columns={"val_acc": "best_val_acc"})
summary = last_epoch.merge(best_acc, on="run_id")

# 종합 점수 계산 (정확도 + 추론 속도 가중 합산)
summary["acc_score"] = min_max_scale(summary["best_val_acc"], higher_is_better=True)
summary["speed_score"] = min_max_scale(summary["latency_ms"], higher_is_better=False)
summary["total_score"] = (
    summary["acc_score"] * (acc_weight / 100.0) + summary["speed_score"] * (speed_weight / 100.0)
).round(1)
summary["과적합상태"] = summary["loss_gap"].apply(lambda g: "⚠️ 주의" if g >= 0.20 else "✅ 양호")
summary = summary.sort_values("total_score", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 3. 핵심 KPI 카드
# ---------------------------------------------------------------------------
top_row = summary.iloc[0]
best_acc_row = summary.loc[summary["best_val_acc"].idxmax()]
fastest_row = summary.loc[summary["latency_ms"].idxmin()]
overfit_cnt = int((summary["loss_gap"] >= 0.20).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("추천 최적 모델 (종합 1위)", top_row["model"], f"{top_row['run_id']} · {top_row['total_score']}점")
c2.metric("최고 검증 정확도", f"{best_acc_row['best_val_acc']:.2f} %", best_acc_row["model"])
c3.metric("최소 추론 지연시간", f"{fastest_row['latency_ms']:.1f} ms", fastest_row["model"])
c4.metric("과적합 주의 실험 수", f"{overfit_cnt} / {len(summary)} 개", "Val−Train Loss ≥ 0.20", delta_color="off")

# ---------------------------------------------------------------------------
# 4. 탭 기반 시각화
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📈 학습 곡선 & 과적합 진단", "⚡ 정확도 vs 추론 속도", "📋 실험 요약 & 로그 다운로드"]
)

# === Tab 1: 학습 곡선 ===
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        acc_fig = px.line(
            view,
            x="epoch",
            y="val_acc",
            color="label",
            markers=True,
            title="에포크별 검증 정확도(Val Accuracy) 추이",
            labels={"epoch": "Epoch", "val_acc": "Val Accuracy (%)", "label": "실험"},
        )
        acc_fig.update_xaxes(dtick=1)
        st.plotly_chart(acc_fig, width="stretch")

    with col2:
        loss_fig = px.line(
            view,
            x="epoch",
            y="val_loss",
            color="label",
            markers=True,
            title="에포크별 검증 손실(Val Loss) 추이",
            labels={"epoch": "Epoch", "val_loss": "Val Loss", "label": "실험"},
        )
        loss_fig.update_xaxes(dtick=1)
        st.plotly_chart(loss_fig, width="stretch")

    # 개별 실험 선택 시 Train vs Val Loss 겹쳐 보기
    st.subheader("단일 실험 과적합 정밀 진단 (Train vs Val Loss)")
    target_label = st.selectbox("진단할 실험 선택", summary["label"].tolist())
    single = view[view["label"] == target_label]
    melted = single.melt(
        id_vars="epoch",
        value_vars=["train_loss", "val_loss"],
        var_name="구분",
        value_name="Loss",
    )
    melted["구분"] = melted["구분"].map({"train_loss": "Train Loss", "val_loss": "Val Loss"})
    diag_fig = px.line(
        melted,
        x="epoch",
        y="Loss",
        color="구분",
        markers=True,
        color_discrete_map={"Train Loss": "#1f77b4", "Val Loss": "#d62728"},
        title=f"{target_label} — 학습/검증 손실 격차 확인",
        labels={"epoch": "Epoch", "Loss": "Loss"},
    )
    diag_fig.update_xaxes(dtick=1)
    st.plotly_chart(diag_fig, width="stretch")

# === Tab 2: 성능 vs 효율 트레이드오프 ===
with tab2:
    col3, col4 = st.columns(2)

    with col3:
        scatter_fig = px.scatter(
            summary,
            x="latency_ms",
            y="best_val_acc",
            size="params_m",
            color="model",
            text="run_id",
            hover_data={
                "optimizer": True,
                "params_m": ":.1f",
                "val_f1": ":.2f",
                "loss_gap": ":.3f",
                "total_score": ":.1f",
            },
            title="추론 시간 vs 최고 정확도 (좌상단일수록 고효율 · 원 크기: 파라미터 수)",
            labels={
                "latency_ms": "추론 시간 (ms, 낮을수록 빠름)",
                "best_val_acc": "최고 검증 정확도 (%)",
                "params_m": "파라미터 (M)",
                "model": "모델",
            },
        )
        scatter_fig.update_traces(textposition="top center")
        st.plotly_chart(scatter_fig, width="stretch")

    with col4:
        rank_fig = px.bar(
            summary.sort_values("total_score", ascending=True),
            x="total_score",
            y="label",
            color="model",
            orientation="h",
            text="total_score",
            title=f"종합 추천 점수 랭킹 (정확도 {acc_weight}% · 속도 {speed_weight}%)",
            labels={"total_score": "종합 점수 (0~100)", "label": "실험", "model": "모델"},
        )
        rank_fig.update_traces(texttemplate="%{text:.1f}점", textposition="outside")
        rank_fig.update_xaxes(range=[0, 115])
        st.plotly_chart(rank_fig, width="stretch")

# === Tab 3: 실험 요약표 & 다운로드 ===
with tab3:
    st.subheader("실험별 최종 요약표 (종합 점수순)")
    display_summary = summary[
        [
            "run_id", "model", "optimizer", "lr", "batch_size",
            "best_val_acc", "val_f1", "latency_ms", "params_m",
            "loss_gap", "과적합상태", "total_score",
        ]
    ].rename(
        columns={
            "run_id": "실험ID",
            "model": "모델",
            "optimizer": "옵티마이저",
            "lr": "학습률",
            "batch_size": "배치크기",
            "best_val_acc": "최고정확도(%)",
            "val_f1": "최종F1(%)",
            "latency_ms": "추론시간(ms)",
            "params_m": "파라미터(M)",
            "loss_gap": "손실격차(Val-Train)",
            "total_score": "종합점수",
        }
    )
    st.dataframe(display_summary, hide_index=True, width="stretch")

    st.download_button(
        "요약 결과 CSV 다운로드",
        data=display_summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="ml_experiment_summary.csv",
        mime="text/csv",
    )

    with st.expander("원본 에포크별 로그 보기"):
        st.dataframe(view.drop(columns=["label"]), hide_index=True, width="stretch")
