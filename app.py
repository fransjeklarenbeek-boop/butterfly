import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

st.title("Dubbele Pendule (snelle animatie)")

# Parameters
g = 9.81
L1 = st.slider("Lengte 1", 0.5, 2.0, 1.0)
L2 = st.slider("Lengte 2", 0.5, 2.0, 1.0)

theta1 = st.slider("Theta1", 0.0, np.pi, 1.0)
theta2 = st.slider("Theta2", 0.0, np.pi, 1.0)

# Simulatie instellingen
dt = 0.05
steps = 300

def simulate():
    t1, t2 = theta1, theta2
    v1, v2 = 0, 0

    x1_list, y1_list = [], []
    x2_list, y2_list = [], []

    for _ in range(steps):
        # simpele fysica (vereenvoudigd voor snelheid)
        a1 = -g / L1 * np.sin(t1)
        a2 = -g / L2 * np.sin(t2)

        v1 += a1 * dt
        v2 += a2 * dt

        t1 += v1 * dt
        t2 += v2 * dt

        x1 = L1 * np.sin(t1)
        y1 = -L1 * np.cos(t1)

        x2 = x1 + L2 * np.sin(t2)
        y2 = y1 - L2 * np.cos(t2)

        x1_list.append(x1)
        y1_list.append(y1)
        x2_list.append(x2)
        y2_list.append(y2)

    return x1_list, y1_list, x2_list, y2_list

if st.button("Start animatie"):
    x1, y1, x2, y2 = simulate()

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    line, = ax.plot([], [], 'o-', lw=2)

    def update(i):
        line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        return line,

    ani = FuncAnimation(fig, update, frames=len(x1), interval=30)

    st.pyplot(fig)
