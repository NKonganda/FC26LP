import streamlit as st
import pandas as pd
import pulp

st.set_page_config(page_title="FC26 Squad Optimizer", layout="wide")

# ── Constants (from backend/main.py) ──────────────────────────────────────────
ATTR_COLS = [
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance",
    "attacking_finishing", "attacking_volleys", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_crossing",
    "power_shot_power", "power_long_shots", "power_jumping",
    "power_stamina", "power_strength",
    "skill_dribbling", "skill_ball_control", "skill_long_passing",
    "mentality_positioning", "mentality_composure", "mentality_vision",
    "mentality_interceptions", "mentality_aggression",
    "defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle",
    "physic",
    "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_positioning", "goalkeeping_reflexes", "goalkeeping_speed",
]

POSITION_MAP = {
    "GK": "GKP",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "CM": "MID", "CAM": "MID", "CDM": "MID", "LM": "MID", "RM": "MID",
    "ST": "FWD", "CF": "FWD", "LW": "FWD", "RW": "FWD", "LF": "FWD", "RF": "FWD",
}

SCORE_COL = {
    "GKP": "goalkeeper_score",
    "DEF": "defender_score",
    "MID": "midfielder_score",
    "FWD": "forward_score",
}

POSITION_COLORS = {
    "GKP": "#f5c518",
    "DEF": "#4caf50",
    "MID": "#2196f3",
    "FWD": "#f44336",
}


# ── Scoring helpers (from backend/main.py) ────────────────────────────────────
def _avg(row, cols):
    return sum(row[c] for c in cols) / len(cols)


def forward_score(row):
    finishing = _avg(row, ["attacking_finishing", "attacking_volleys",
                           "power_shot_power", "power_long_shots", "mentality_positioning"])
    pace      = _avg(row, ["movement_acceleration", "movement_sprint_speed"])
    dribbling = _avg(row, ["skill_dribbling", "skill_ball_control", "movement_agility"])
    physical  = _avg(row, ["attacking_heading_accuracy", "power_jumping", "power_strength"])
    mental    = _avg(row, ["mentality_composure", "movement_reactions"])
    return round(finishing * 0.30 + pace * 0.20 + dribbling * 0.20
                 + physical * 0.15 + mental * 0.15, 1)


def midfielder_score(row):
    passing    = _avg(row, ["attacking_short_passing", "skill_long_passing",
                            "mentality_vision", "attacking_crossing"])
    dribbling  = _avg(row, ["skill_dribbling", "skill_ball_control", "movement_agility"])
    physical   = _avg(row, ["power_stamina", "physic"])
    defending  = _avg(row, ["mentality_interceptions", "defending_marking_awareness"])
    pace_react = _avg(row, ["movement_acceleration", "movement_reactions"])
    return round(passing * 0.30 + dribbling * 0.20 + physical * 0.15
                 + defending * 0.15 + pace_react * 0.20, 1)


def defender_score(row):
    defending = _avg(row, ["defending_marking_awareness", "defending_standing_tackle",
                           "defending_sliding_tackle", "mentality_interceptions"])
    physical  = _avg(row, ["power_strength", "power_jumping",
                           "attacking_heading_accuracy", "physic"])
    mental    = _avg(row, ["mentality_aggression", "mentality_composure", "movement_reactions"])
    pace      = _avg(row, ["movement_acceleration", "movement_sprint_speed"])
    return round(defending * 0.40 + physical * 0.25 + mental * 0.20 + pace * 0.15, 1)


def goalkeeper_score(row):
    shot_stop = _avg(row, ["goalkeeping_diving", "goalkeeping_reflexes", "goalkeeping_positioning"])
    distribut = _avg(row, ["goalkeeping_handling", "goalkeeping_kicking"])
    mobility  = row["goalkeeping_speed"]
    return round(shot_stop * 0.60 + distribut * 0.30 + mobility * 0.10, 1)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_players():
    df = pd.read_csv("FC26_20250921.csv", low_memory=False)
    df[ATTR_COLS] = df[ATTR_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["value_eur"] = pd.to_numeric(df["value_eur"], errors="coerce").fillna(0)
    df["club_name"]        = df["club_name"].astype(str).str.strip()
    df["nationality_name"] = df["nationality_name"].astype(str).str.strip()
    # astype(str) converts NaN → "nan"; restore proper NaN so dropna() still works
    df["club_name"]        = df["club_name"].replace("nan", pd.NA)
    df["nationality_name"] = df["nationality_name"].replace("nan", pd.NA)

    df["forward_score"]    = df.apply(forward_score, axis=1)
    df["midfielder_score"] = df.apply(midfielder_score, axis=1)
    df["defender_score"]   = df.apply(defender_score, axis=1)
    df["goalkeeper_score"] = df.apply(goalkeeper_score, axis=1)

    df["category"] = df["player_positions"].apply(
        lambda p: POSITION_MAP.get(str(p).split(",")[0].strip(), "MID")
    )
    df["rating"] = df.apply(lambda r: r[SCORE_COL[r["category"]]], axis=1)

    opt_df = (
        df.sort_values("rating", ascending=False)
        .drop_duplicates("player_id")
        .reset_index(drop=True)
    )

    names     = list(opt_df["short_name"])
    ratings   = list(opt_df["rating"])
    prices    = list(opt_df["value_eur"])
    positions = list(opt_df["category"])
    clubs     = list(opt_df["club_name"].fillna(""))
    pos_raw   = list(opt_df["player_positions"])
    n         = len(opt_df)

    return opt_df, names, ratings, prices, positions, clubs, pos_raw, n


# ── Optimization (from backend/main.py) ───────────────────────────────────────
def optimize(budget, n_def, n_mid, n_fwd, opt_df, names, ratings, prices, positions, clubs, pos_raw, n, max_per_club=3):
    model = pulp.LpProblem("FC26_Squad_Optimizer", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)]

    model += pulp.lpSum(x[i] * ratings[i] for i in range(n))
    model += pulp.lpSum(x[i] * prices[i] for i in range(n)) <= budget

    model += pulp.lpSum(x[i] for i in range(n) if positions[i] == "GKP") == 1
    model += pulp.lpSum(x[i] for i in range(n) if positions[i] == "DEF") == n_def
    model += pulp.lpSum(x[i] for i in range(n) if positions[i] == "MID") == n_mid
    model += pulp.lpSum(x[i] for i in range(n) if positions[i] == "FWD") == n_fwd
    model += pulp.lpSum(x) == 11

    for club in opt_df["club_name"].dropna().unique():
        model += pulp.lpSum(x[i] for i in range(n) if clubs[i] == club) <= max_per_club

    model.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        return None, f"Solver returned: {status}"

    squad = [
        {
            "name":      names[i],
            "club":      clubs[i],
            "positions": pos_raw[i],
            "category":  positions[i],
            "rating":    ratings[i],
            "value_eur": prices[i],
        }
        for i in range(n)
        if x[i].value() == 1
    ]

    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    squad.sort(key=lambda p: (order[p["category"]], -p["rating"]))

    return {
        "status":       status,
        "total_cost":   sum(p["value_eur"] for p in squad),
        "total_rating": round(sum(p["rating"] for p in squad), 1),
        "budget":       budget,
        "squad":        squad,
    }, None


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center'>FC26 Squad Optimizer</h1>"
    "<p style='text-align:center;color:gray'>Build the highest-rated 11 within your budget</p>",
    unsafe_allow_html=True,
)

opt_df, names, ratings, prices, positions, clubs, pos_raw, n = load_players()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    budget_m = st.slider("Budget (€M)", min_value=100, max_value=1000, value=500, step=10)

    st.subheader("Formation")
    prev_def = int(st.session_state.get("n_def", 4))
    prev_mid = int(st.session_state.get("n_mid", 3))
    prev_fwd = int(st.session_state.get("n_fwd", 3))

    def_max = max(2, 10 - prev_mid - prev_fwd)
    mid_max = max(2, 10 - prev_def - prev_fwd)
    fwd_max = max(2, 10 - prev_def - prev_mid)

    n_def = st.number_input("DEF", min_value=2, max_value=def_max, value=min(prev_def, def_max), step=1, key="n_def")
    n_mid = st.number_input("MID", min_value=2, max_value=mid_max, value=min(prev_mid, mid_max), step=1, key="n_mid")
    n_fwd = st.number_input("FWD", min_value=2, max_value=fwd_max, value=min(prev_fwd, fwd_max), step=1, key="n_fwd")

    outfield_total = n_def + n_mid + n_fwd
    is_valid = outfield_total == 10

    st.markdown(
        f"**Formation total:** 1 GK + {n_def} DEF + {n_mid} MID + {n_fwd} FWD = "
        f"**{1 + outfield_total}** {'(valid)' if is_valid else '(invalid)'}"
    )
    if not is_valid:
        st.warning(f"DEF + MID + FWD must equal 10 (currently {outfield_total})")

    st.subheader("Filters")
    all_nations = sorted(opt_df["nationality_name"].dropna().unique())
    sel_nations = st.multiselect("Nationality", options=all_nations, placeholder="All nationalities")

    all_clubs = sorted(opt_df["club_name"].dropna().unique())
    sel_clubs = st.multiselect("Club", options=all_clubs, placeholder="All clubs")

    optimize_clicked = st.button("Optimize Squad", disabled=not is_valid, type="primary", use_container_width=True)

# ── Run optimizer ─────────────────────────────────────────────────────────────
if optimize_clicked:
    filtered_df = opt_df.copy()
    if sel_nations:
        filtered_df = filtered_df[filtered_df["nationality_name"].isin(sel_nations)]
    if sel_clubs:
        filtered_df = filtered_df[filtered_df["club_name"].isin(sel_clubs)]

    f_names     = list(filtered_df["short_name"])
    f_ratings   = list(filtered_df["rating"])
    f_prices    = list(filtered_df["value_eur"])
    f_positions = list(filtered_df["category"])
    f_clubs     = list(filtered_df["club_name"].fillna(""))
    f_pos_raw   = list(filtered_df["player_positions"])
    f_n         = len(filtered_df)

    with st.spinner("Optimizing…"):
        result, err = optimize(
            budget_m * 1_000_000,
            int(n_def), int(n_mid), int(n_fwd),
            filtered_df, f_names, f_ratings, f_prices, f_positions, f_clubs, f_pos_raw, f_n,
            max_per_club=11 if sel_clubs else 3,
        )
    if err:
        st.error(f"Solver error: {err}")
        st.session_state.pop("result", None)
    else:
        st.session_state["result"] = result

# ── Display results ───────────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]

    for cat in ["GKP", "DEF", "MID", "FWD"]:
        group = [p for p in result["squad"] if p["category"] == cat]
        if not group:
            continue

        color = POSITION_COLORS[cat]
        st.markdown(
            f"<h3 style='color:{color};border-left:4px solid {color};padding-left:8px'>"
            f"{cat} &nbsp;·&nbsp; {len(group)} player{'s' if len(group) != 1 else ''}</h3>",
            unsafe_allow_html=True,
        )

        rows = [
            {
                "Name":    p["name"],
                "Club":    p["club"],
                "Rating":  p["rating"],
                "Value":   f"€{p['value_eur'] / 1e6:.1f}M",
            }
            for p in group
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rating", f"{result['total_rating']:.1f}")
    with col2:
        used_m  = result["total_cost"] / 1e6
        total_m = result["budget"] / 1e6
        st.metric("Budget Used", f"€{used_m:.1f}M / €{total_m:.0f}M")

    budget_pct = min(result["total_cost"] / result["budget"], 1.0)
    st.progress(budget_pct)
