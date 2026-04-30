import math
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import streamlit as st

DECIMALS = 14
EPS_E = 1e-12


def truncate_14(s: str) -> str:
    s = str(s).strip().replace(",", ".")
    if s == "":
        return s
    if "." in s:
        a, frac = s.split(".", 1)
        if len(frac) > DECIMALS:
            frac = frac[:DECIMALS]
            s = a + "." + frac
    return s


def parse_float_14(value, default: float) -> float:
    s2 = truncate_14(value)
    if s2 == "":
        return default
    try:
        return float(s2)
    except Exception:
        return default


def accelerations(theta1, theta2, omega1, omega2, m1, m2, L1, L2, g):
    dtheta = theta1 - theta2
    s = math.sin(dtheta)
    c = math.cos(dtheta)

    denom = 2 * m1 + m2 - m2 * math.cos(2 * dtheta)
    if abs(denom) < 1e-12:
        denom = 1e-12 if denom >= 0 else -1e-12

    denom1 = L1 * denom
    denom2 = L2 * denom

    num1 = (
        -g * (2 * m1 + m2) * math.sin(theta1)
        - m2 * g * math.sin(theta1 - 2 * theta2)
        - 2 * s * m2 * (omega2 * omega2 * L2 + omega1 * omega1 * L1 * c)
    )
    alpha1 = num1 / denom1

    num2 = (
        2
        * s
        * (
            omega1 * omega1 * L1 * (m1 + m2)
            + g * (m1 + m2) * math.cos(theta1)
            + omega2 * omega2 * L2 * m2 * c
        )
    )
    alpha2 = num2 / denom2

    return alpha1, alpha2


def step(theta1, theta2, omega1, omega2, m1, m2, L1, L2, g, dt):
    a1, a2 = accelerations(theta1, theta2, omega1, omega2, m1, m2, L1, L2, g)
    omega1_new = omega1 + dt * a1
    omega2_new = omega2 + dt * a2
    theta1_new = theta1 + dt * omega1_new
    theta2_new = theta2 + dt * omega2_new
    return theta1_new, theta2_new, omega1_new, omega2_new


def positions(theta1, theta2, L1, L2):
    x1 = L1 * math.sin(theta1)
    y1 = -L1 * math.cos(theta1)
    x2 = x1 + L2 * math.sin(theta2)
    y2 = y1 - L2 * math.cos(theta2)
    return x1, y1, x2, y2


def total_energy(theta1, theta2, omega1, omega2, m1, m2, L1, L2, g):
    y1 = -L1 * math.cos(theta1)
    y2 = y1 - L2 * math.cos(theta2)

    v1_sq = (L1 * omega1) ** 2
    v2_sq = (
        v1_sq
        + (L2 * omega2) ** 2
        + 2 * L1 * L2 * omega1 * omega2 * math.cos(theta1 - theta2)
    )

    t = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
    v = m1 * g * y1 + m2 * g * y2
    return t + v


@dataclass
class PendulumConfig:
    m1: float
    m2: float
    L1: float
    L2: float
    th1_deg: float
    th2_deg: float
    v1: float
    v2: float


DEFAULTS = {
    "g": 9.81,
    "dt": 0.0005,
    "substeps": 50,
    "trail_len": 400,
    "A_m1": 1.0,
    "A_m2": 1.0,
    "A_L1": 1.0,
    "A_L2": 1.0,
    "A_th1_deg": 120.0,
    "A_th2_deg": -10.0,
    "A_v1": 0.0,
    "A_v2": 0.0,
    "B_m1": 1.0,
    "B_m2": 1.0,
    "B_L1": 1.0,
    "B_L2": 1.0,
    "B_th1_deg": 120.0,
    "B_th2_deg": -10.0,
    "B_v1": 0.0,
    "B_v2": 0.0,
}


def number_input(label: str, key: str):
    return st.number_input(
        label,
        value=float(DEFAULTS[key]),
        format="%.14f",
        key=key,
    )


def read_config(prefix: str) -> PendulumConfig:
    return PendulumConfig(
        m1=number_input(f"{prefix} m1 (kg)", f"{prefix}_m1"),
        m2=number_input(f"{prefix} m2 (kg)", f"{prefix}_m2"),
        L1=max(number_input(f"{prefix} L1 (m)", f"{prefix}_L1"), 1e-6),
        L2=max(number_input(f"{prefix} L2 (m)", f"{prefix}_L2"), 1e-6),
        th1_deg=number_input(f"{prefix} th1 (deg)", f"{prefix}_th1_deg"),
        th2_deg=number_input(f"{prefix} th2 (deg)", f"{prefix}_th2_deg"),
        v1=number_input(f"{prefix} v1 = ω1 (rad/s)", f"{prefix}_v1"),
        v2=number_input(f"{prefix} v2 = ω2 (rad/s)", f"{prefix}_v2"),
    )


def init_state(a: PendulumConfig, b: PendulumConfig):
    st.session_state.state_a = [
        math.radians(a.th1_deg),
        math.radians(a.th2_deg),
        a.v1,
        a.v2,
    ]
    st.session_state.state_b = [
        math.radians(b.th1_deg),
        math.radians(b.th2_deg),
        b.v1,
        b.v2,
    ]
    st.session_state.trail_a = []
    st.session_state.trail_b = []
    st.session_state.started_at = time.time()


def draw_plot(params, status_text: str):
    state_a = st.session_state.state_a
    state_b = st.session_state.state_b

    t1a, t2a, _, _ = state_a
    x1a, y1a, x2a, y2a = positions(t1a, t2a, params["A_L1"], params["A_L2"])

    t1b, t2b, _, _ = state_b
    x1b, y1b, x2b, y2b = positions(t1b, t2b, params["B_L1"], params["B_L2"])

    st.session_state.trail_a.append((x2a, y2a))
    st.session_state.trail_b.append((x2b, y2b))
    st.session_state.trail_a = st.session_state.trail_a[-params["trail_len"] :]
    st.session_state.trail_b = st.session_state.trail_b[-params["trail_len"] :]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Two Double Pendulums (A=blue, B=orange)")
    ax.plot([0, x1a, x2a], [0, y1a, y2a], lw=2, label="A")
    ax.plot([0, x1b, x2b], [0, y1b, y2b], lw=2, label="B")

    if st.session_state.trail_a:
        xs, ys = zip(*st.session_state.trail_a)
        ax.plot(xs, ys, lw=1, alpha=0.5)
    if st.session_state.trail_b:
        xs, ys = zip(*st.session_state.trail_b)
        ax.plot(xs, ys, lw=1, alpha=0.5)

    lmax = max(params["A_L1"] + params["A_L2"], params["B_L1"] + params["B_L2"])
    pad = 0.2 * lmax
    ax.set_xlim(-lmax - pad, lmax + pad)
    ax.set_ylim(-lmax - pad, lmax + pad)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)
    st.caption(status_text)


def simulate_one_frame(params):
    g = params["g"]
    dt = params["dt"]

    for _ in range(params["substeps"]):
        t1, t2, o1, o2 = st.session_state.state_a
        st.session_state.state_a = list(
            step(
                t1,
                t2,
                o1,
                o2,
                params["A_m1"],
                params["A_m2"],
                params["A_L1"],
                params["A_L2"],
                g,
                dt,
            )
        )

        t1, t2, o1, o2 = st.session_state.state_b
        st.session_state.state_b = list(
            step(
                t1,
                t2,
                o1,
                o2,
                params["B_m1"],
                params["B_m2"],
                params["B_L1"],
                params["B_L2"],
                g,
                dt,
            )
        )


def energy_status(params):
    t1, t2, o1, o2 = st.session_state.state_a
    ea = total_energy(
        t1, t2, o1, o2, params["A_m1"], params["A_m2"], params["A_L1"], params["A_L2"], params["g"]
    )

    t1, t2, o1, o2 = st.session_state.state_b
    eb = total_energy(
        t1, t2, o1, o2, params["B_m1"], params["B_m2"], params["B_L1"], params["B_L2"], params["g"]
    )

    if "e0_a" not in st.session_state:
        st.session_state.e0_a = ea
    if "e0_b" not in st.session_state:
        st.session_state.e0_b = eb

    e0_avg = 0.5 * (abs(st.session_state.e0_a) + abs(st.session_state.e0_b))
    rel_eab_pct = 100.0 * (ea - eb) / (e0_avg + EPS_E)
    elapsed = time.time() - st.session_state.get("started_at", time.time())
    return f"Tijd sinds start: {elapsed:0.3f} s | (E_A−E_B)/E0 = {rel_eab_pct:+.3e}%"


st.set_page_config(page_title="Dubbele DP Simulatie", layout="wide")
st.title("Dubbele dubbele pendulum simulatie")
st.write("Webversie van de originele Tkinter/Matplotlib-app, omgezet naar Streamlit.")

with st.sidebar:
    st.header("Instellingen")
    g = number_input("g (m/s²)", "g")
    dt = max(number_input("dt (s)", "dt"), 1e-12)
    substeps = max(1, int(st.number_input("substeps (int)", value=int(DEFAULTS["substeps"]), step=1)))
    trail_len = max(10, int(st.number_input("trail_len (int)", value=int(DEFAULTS["trail_len"]), step=10)))

    st.divider()
    st.subheader("Pendulum A")
    config_a = read_config("A")

    st.divider()
    st.subheader("Pendulum B")
    config_b = read_config("B")

    st.divider()
    frames = st.slider("Aantal frames per klik", 1, 500, 120)
    delay = st.slider("Pauze tussen frames (seconden)", 0.0, 0.1, 0.02, 0.005)

params = {
    "g": g,
    "dt": dt,
    "substeps": substeps,
    "trail_len": trail_len,
    "A_m1": config_a.m1,
    "A_m2": config_a.m2,
    "A_L1": config_a.L1,
    "A_L2": config_a.L2,
    "A_th1": math.radians(config_a.th1_deg),
    "A_th2": math.radians(config_a.th2_deg),
    "A_v1": config_a.v1,
    "A_v2": config_a.v2,
    "B_m1": config_b.m1,
    "B_m2": config_b.m2,
    "B_L1": config_b.L1,
    "B_L2": config_b.L2,
    "B_th1": math.radians(config_b.th1_deg),
    "B_th2": math.radians(config_b.th2_deg),
    "B_v1": config_b.v1,
    "B_v2": config_b.v2,
}

col1, col2 = st.columns([1, 1])
with col1:
    restart = st.button("Start / Restart", type="primary", use_container_width=True)
with col2:
    run = st.button("Simuleer frames", use_container_width=True)

if restart or "state_a" not in st.session_state:
    init_state(config_a, config_b)
    st.session_state.e0_a = total_energy(
        *st.session_state.state_a,
        params["A_m1"],
        params["A_m2"],
        params["A_L1"],
        params["A_L2"],
        params["g"],
    )
    st.session_state.e0_b = total_energy(
        *st.session_state.state_b,
        params["B_m1"],
        params["B_m2"],
        params["B_L1"],
        params["B_L2"],
        params["g"],
    )

plot_slot = st.empty()

if run:
    for _ in range(frames):
        simulate_one_frame(params)
        with plot_slot.container():
            draw_plot(params, energy_status(params))
        if delay:
            time.sleep(delay)
else:
    with plot_slot.container():
        draw_plot(params, energy_status(params))
