---
marp: true
theme: default
paginate: true
math: katex
---
<img src="Presentation/img.png" width="200">

# FC26 Squad Optimizer
## Linear Programming for Optimal Football Squad Selection

> Selecting the highest-rated 11-player squad subject to real-world constraints

---

## Motivation

**Problem:** Given ~18,000 professional players and a fixed budget, which 11 should you pick?

- Naïve approach: sort by overall rating → ignores budget, position balance, club rules
- Heuristic approach: pick "good" players manually → sub-optimal, slow, not reproducible
- **LP approach:** encode every rule as a mathematical constraint; let a solver guarantee the global optimum

**Use cases**
- Football Manager / FC26 Ultimate Team squad building
- Fantasy football draft tools
- Real scouting departments with salary-cap constraints

---

## Dataset

**Source:** FC26 player attributes CSV (`FC26_20250921.csv`)

| Statistic | Value |
|-----------|-------|
| Total players loaded | 18,297 |
| Attribute columns used | 33 |
| Position categories | GKP · DEF · MID · FWD |

**33 attribute columns span six groups:**
`movement` · `attacking` · `power` · `skill` · `mentality` · `defending` · `goalkeeping`

All attributes coerced to numeric; NaN filled with 0.

---

## Position-Specific Scoring

Each player receives **four independent scores** — one per position archetype.

| Position | Component | Weight |
|----------|-----------|--------|
| **FWD** | Finishing (finishing, volleys, shot power, long shots, positioning) | 30% |
| | Pace (acceleration, sprint speed) | 20% |
| | Dribbling (dribbling, ball control, agility) | 20% |
| | Physical (heading, jumping, strength) | 15% |
| | Mental (composure, reactions) | 15% |
| **MID** | Passing (short pass, long pass, vision, crossing) | 30% |
| | Pace / Reactions | 20% |
| | Dribbling | 20% |
| | Physical (stamina, physic) | 15% |
| | Defending (interceptions, marking) | 15% |
| **DEF** | Defending (marking, standing tackle, sliding tackle, interceptions) | 40% |
| | Physical (strength, jumping, heading, physic) | 25% |
| | Mental (aggression, composure, reactions) | 20% |
| | Pace | 15% |
| **GK** | Shot-stopping (diving, reflexes, positioning) | 60% |
| | Distribution (handling, kicking) | 30% |
| | Mobility (speed) | 10% |

The **rating used in the LP** is the score matching the player's primary position.

---

## LP Problem Formulation

### Sets & Parameters

| Symbol | Meaning       |
|--------|---------------|
| $I$ | All players $(I = 18{,}297)$ |
| $I_{GK}, I_{DEF}, I_{MID}, I_{FWD}$ | Players whose primary position is GK / DEF / MID / FWD |
| $r_i$ | Positional rating of player $i$ |
| $p_i$ | Market value (€) of player $i$ |
| $B$ | Budget cap    |

### Decision Variables

$$x_i \in \{0, 1\} \quad \forall\, i \in I$$

$x_i = 1$ if player $i$ is selected; $0$ otherwise.

---

## LP Problem Formulation (cont.)

### Objective — Maximise Total Squad Rating

$$\max_{x} \quad \sum_{i \in I} r_i \, x_i$$

### Subject To

$$\sum_{i \in I} p_i \, x_i \;\leq\; B \tag{Budget}$$

$$\sum_{i \in I_{GK}} x_i = 1 \tag{GK}$$

$$\sum_{i \in I_{DEF}} x_i \;\geq\; 2 \tag{DEF min}$$

$$\sum_{i \in I_{MID}} x_i \;\geq\; 2 \tag{MID min}$$

$$\sum_{i \in I_{FWD}} x_i \;\geq\; 2 \tag{FWD min}$$

$$\sum_{i \in I} x_i = 11 \tag{Squad size}$$

$$x_i \in \{0, 1\} \quad \forall\, i \in I \tag{Integrality}$$

**Implementation:** PuLP + CBC solver in Python

```python
model += pulp.lpSum(x[i] * ratings[i] for i in range(n))          # objective
model += pulp.lpSum(x[i] * prices[i]  for i in range(n)) <= BUDGET # budget
model += pulp.lpSum(x[i] for i in range(n) if positions[i] == 'GKP') == 1
model += pulp.lpSum(x) == 11
```

---

## Dual Variables — Shadow Prices

Duals are computed on the **LP relaxation** ($x_i \in [0,1]$) and give the **marginal value** of each constraint.

| Constraint        | Dual symbol | Interpretation                                                                                                                           |
|-------------------|-------------|------------------------------------------------------------------------------------------------------------------------------------------|
| Budget $\leq B$   | $\lambda_B \geq 0$ | Rating gained per extra €1 of budget — the "price of cap space." If $\lambda_B = 2 \times 10^{-7}$, then €5M extra cap ≈ +1 rating point |
| Squad size $= 11$ | $\mu_{squad}$ (unrestricted) | Rating value of a 12th squad slot; positive → the size cap is binding                                                                    |
| GK $= 1$          | $\mu_{GK}$ (unrestricted) | Cost of being forced to hold exactly one GK slot vs. filling it freely                                                                   |
| DEF $\geq 2$      | $\mu_{DEF} \leq 0$ | Rating penalty from requiring ≥2 defenders; $= 0$ means the solver would have picked ≥2 anyway                                           |
| MID $\geq 2$      | $\mu_{MID} \leq 0$ | Same interpretation for midfield floor                                                                                                   |
| FWD $\geq 2$      | $\mu_{FWD} \leq 0$ | Same interpretation for forward floor                                                                                                    |

**Key insight:** $\lambda_B$ bridges the accountant's budget and the coach's ratings — it tells exactly how much squad quality each extra million buys. A manager can use this to justify transfer spending to a board.

---

## Results — Optimal Squad

**Budget:** €1,000M &nbsp;·&nbsp; **Formation:** 1 GK · 3 DEF · 2 MID · 5 FWD &nbsp;

| Position | Player | Club | Rating | Value |
|----------|--------|------|-------:|------:|
| GKP | Alisson | Liverpool | 84.2 | €51.0M |
| DEF | V. van Dijk | Liverpool | 86.8 | €57.0M |
| DEF | Marquinhos | Paris Saint-Germain | 85.1 | €54.5M |
| DEF | A. Rüdiger | Real Madrid | 84.4 | €44.0M |
| MID | J. Kimmich | FC Bayern München | 85.4 | €86.0M |
| MID | N. Barella | Inter | 85.0 | €79.5M |
| FWD | K. Mbappé | Real Madrid | 90.4 | €173.5M |
| FWD | O. Dembélé | Paris Saint-Germain | 88.2 | €122.5M |
| FWD | V. Osimhen | Galatasaray SK | 86.2 | €100.0M |
| FWD | L. Martínez | Inter | 86.2 | €99.0M |
| FWD | V. Gyökeres | Arsenal | 85.8 | €93.0M |

**Total rating: 947.7 &nbsp;·&nbsp; Total cost: €960.0M / €1,000M**

---

## Interactive App — Streamlit

**File:** `FC26LP/app.py` &nbsp;

### Features

| Panel | Controls |
|-------|----------|
| **Sidebar** | Budget slider (€100M – €1,000M) |
| | Formation inputs: DEF / MID / FWD (must sum to 10 outfield) |
| | Nationality & Club multiselect filters |
| | "Optimize Squad" button (disabled if formation invalid) |
| **Results** | Colour-coded tables per position group (GKP · DEF · MID · FWD) |
| | Total Rating metric + Budget Used metric + progress bar |

**Additional constraint in app:** $\sum_{i \in C_c} x_i \leq 3$ for each club $c$ (club diversity rule; auto-relaxed when a club filter is applied)