"""QuantEvoLoop Streamlit Dashboard — real-time evolution monitoring.

Run: streamlit run src/quantevoloop/dashboard/app.py -- --workspace ./evo_workspace
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    print("streamlit not installed. Run: pip install streamlit plotly pandas")
    sys.exit(1)

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    pd = None
    px = None
    go = None


def load_json(path: Path) -> dict | list | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def main():
    st.set_page_config(
        page_title="QuantEvoLoop Dashboard",
        page_icon="🧬",
        layout="wide",
    )

    # Parse workspace path
    workspace = Path("./evo_workspace")
    if "--workspace" in sys.argv:
        idx = sys.argv.index("--workspace")
        if idx + 1 < len(sys.argv):
            workspace = Path(sys.argv[idx + 1])

    st.title("🧬 QuantEvoLoop Dashboard")

    if not workspace.exists():
        st.warning(f"Workspace not found: {workspace}")
        st.info("Start the evolution first: `quantevoloop run --max-gens 20 --lanes 3`")
        return

    # --- Sidebar: workspace selector ---
    st.sidebar.header("⚙️ Workspace")
    st.sidebar.text(str(workspace))

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
    if auto_refresh:
        import time
        time.sleep(0.1)  # small delay to avoid rapid reruns
        st.rerun()

    # --- Row 1: State overview ---
    state = load_json(workspace / "state.json")
    if state:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Generation", state.get("generation", 0))
        col2.metric("Status", state.get("status", "unknown"))
        col3.metric("Cost (USD)", f"${state.get('total_cost_usd', 0):.2f}")
        col4.metric("Promotions", state.get("promotions", 0))

    # --- Row 2: Champion metrics ---
    metrics = load_json(workspace / "champion" / "metrics.json")
    if metrics:
        st.header("👑 Champion")
        col1, col2, col3, col4 = st.columns(4)
        train = metrics.get("train", {})
        test = metrics.get("test", {})
        col1.metric("Train Sharpe", f"{train.get('sharpe', 0):.3f}")
        col2.metric("Test Sharpe", f"{test.get('sharpe', 0):.3f}")
        col3.metric("Test CAGR", f"{test.get('cagr', 0):.1%}")
        col4.metric("Test MaxDD", f"{test.get('max_drawdown_account', 0):.1%}")

    # --- Row 3: Generation index ---
    gen_index_file = workspace / "generations" / "gen_index.jsonl"
    if gen_index_file.exists() and pd is not None:
        st.header("📊 Generation History")
        records = []
        for line in gen_index_file.read_text().strip().split("\n"):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if records:
            df = pd.DataFrame(records)

            # Score per generation
            if "score" in df.columns and "gen" in df.columns:
                fig = px.scatter(
                    df, x="gen", y="score", color="status",
                    title="Score per Generation",
                    hover_data=["lane", "hypothesis_tag", "duration_s"],
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Cost accumulation
            if "cost_usd" in df.columns:
                df["cum_cost"] = df["cost_usd"].cumsum()
                fig2 = px.line(
                    df, x="gen", y="cum_cost",
                    title="Cumulative Cost (USD)",
                )
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)

            # Status distribution
            col1, col2 = st.columns(2)
            if "status" in df.columns:
                status_counts = df["status"].value_counts()
                fig3 = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Verdict Distribution",
                )
                col1.plotly_chart(fig3, use_container_width=True)

            if "hypothesis_tag" in df.columns:
                tag_counts = df["hypothesis_tag"].value_counts()
                fig4 = px.bar(
                    x=tag_counts.index, y=tag_counts.values,
                    title="Mutation Type Distribution",
                )
                col2.plotly_chart(fig4, use_container_width=True)

    # --- Row 4: UCB Bandit stats ---
    # Try to load from latest checkpoint
    checkpoints = sorted(workspace.glob("checkpoint_gen*.json"))
    if checkpoints:
        latest_ckpt = load_json(checkpoints[-1])
        if latest_ckpt and "bandit" in latest_ckpt:
            st.header("🎰 UCB Bandit (Mutation Selection)")
            bandit = latest_ckpt["bandit"]
            if isinstance(bandit, dict):
                bandit_df = pd.DataFrame([
                    {"mutation_type": k, **v}
                    for k, v in bandit.items()
                ]) if bandit else None

                if bandit_df is not None and not bandit_df.empty:
                    st.dataframe(bandit_df, use_container_width=True)

    # --- Row 5: Recent generations ---
    gen_dirs = sorted((workspace / "generations").glob("gen_*"), reverse=True)
    if gen_dirs:
        st.header("🔬 Recent Generations")
        for gen_dir in gen_dirs[:5]:
            with st.expander(f"📁 {gen_dir.name}"):
                decision_files = list(gen_dir.rglob("decision.md"))
                for df_path in decision_files:
                    st.markdown(df_path.read_text()[:500])

    st.sidebar.markdown("---")
    st.sidebar.caption("QuantEvoLoop Dashboard • Auto-refreshing")


if __name__ == "__main__":
    main()
