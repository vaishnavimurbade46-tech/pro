import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="Clinical Vaccine Impact Simulator", layout="wide")
st.title("🏥 Professional SVIR Epidemic Simulator")
st.markdown("Analyze the impact of vaccination campaigns on hospital ward safety.")

# --- Sidebar: Advanced Parameters ---
st.sidebar.header("🕹️ Simulation Controls")
N = st.sidebar.number_input("Total Ward Population", value=1000)
days = st.sidebar.slider("Timeline (Days)", 30, 300, 150)

st.sidebar.subheader("Biological Factors")
beta = st.sidebar.slider("Transmission Rate (β)", 0.1, 1.0, 0.4, help="Probability of infection per contact.")
gamma = st.sidebar.slider("Recovery Rate (γ)", 0.05, 0.5, 0.1, help="1/Days to recover.")

st.sidebar.subheader("Intervention Strategy")
v_rate = st.sidebar.slider("Daily Vaccination Rate", 0.0, 0.1, 0.02)
efficacy = st.sidebar.slider("Vaccine Efficacy (ε)", 0.0, 1.0, 0.90, help="How well the vaccine prevents infection.")

# --- The SVIR Engine (Improved) ---
def svir_model(y, t, N, beta, gamma, v_rate, efficacy):
    S, V, I, R = y
    
    # Force of infection
    # We assume vaccinated people have (1 - efficacy) risk
    lambda_inf = beta * I / N
    
    dSdt = -lambda_inf * S - v_rate * S
    dVdt = v_rate * S - (1 - efficacy) * lambda_inf * V
    dIdt = lambda_inf * S + (1 - efficacy) * lambda_inf * V - gamma * I
    dRdt = gamma * I
    
    return dSdt, dVdt, dIdt, dRdt

# --- Execution ---
t = np.linspace(0, days, days)
y0 = (N-1, 0, 1, 0) # Start with 1 infected
results = odeint(svir_model, y0, t, args=(N, beta, gamma, v_rate, efficacy))
S, V, I, R = results.T

# --- Immunity Analysis Calculations ---
r0 = beta / gamma
# Re changes over time; we'll show the current Re based on Susceptible + Vaccinated risk
current_re = (beta / gamma) * ((S[-1] + (1-efficacy)*V[-1]) / N)
herd_threshold = (1 - (1/r0)) * 100 if r0 > 1 else 0

# --- UI: Dashboard Layout ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Basic R0", f"{r0:.2f}")
col2.metric("Current Re", f"{current_re:.2f}", delta=f"{current_re-1:.2f}", delta_color="inverse")
col3.metric("Peak Load", f"{max(I):.0f} Patients")
col4.metric("Herd Threshold", f"{herd_threshold:.1f}%")

# --- UI: Visualization ---
tab1, tab2 = st.tabs(["📈 Infection Dynamics", "📊 Data Table"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(t, I, color="red", alpha=0.2, label="Infection Area")
    ax.plot(t, S, label="Susceptible", color="#1f77b4", lw=2)
    ax.plot(t, V, label="Vaccinated", color="#ff7f0e", lw=2)
    ax.plot(t, I, label="Infectious", color="#d62728", lw=3)
    ax.plot(t, R, label="Recovered", color="#2ca02c", lw=2)
    
    ax.set_xlabel("Days")
    ax.set_ylabel("Population Count")
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

with tab2:
    st.dataframe({"Day": t, "Susceptible": S, "Infected": I, "Vaccinated": V})
