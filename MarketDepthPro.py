import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Streamlit App Title
st.title("Interactive Token Unlock Schedule with Market Depth Provisioning")

# User Inputs
st.sidebar.header("Tokenomics Inputs")

# General Parameters
max_supply = st.sidebar.number_input("Maximum Supply (tokens)", value=1_000_000_000, step=100_000_000)
initial_token_price = st.sidebar.number_input("Initial Token Price (USD)", value=0.1, step=0.01)
token_price = st.sidebar.number_input("Token Price (USD, Price Model)", value=0.1, step=0.01)

# Offset Simulation
offset_month = st.sidebar.number_input("Offset Simulation Start Month", value=0, min_value=0, step=1)

# Price Model Selection
st.sidebar.header("Price Model")
price_model = st.sidebar.radio("Choose Price Model", ("Constant Price", "Stochastic Price (Black-Scholes)"))

if price_model == "Stochastic Price (Black-Scholes)":
    mu = st.sidebar.number_input("Expected Return (mu)", value=0.05, step=0.01)
    sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2, step=0.01)
    time_horizon = 40  # 40 months
    dt = 1 / 12
    np.random.seed(42)
    stochastic_prices = [token_price]
    # Offset adjustment: Start price model simulation at offset_month
    for _ in range(1, time_horizon + offset_month):
        random_shock = np.random.normal(0, 1)
        price = stochastic_prices[-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_shock)
        stochastic_prices.append(price)
    # Adjust stochastic_prices to align with time horizon after offset
    stochastic_prices = stochastic_prices[offset_month:offset_month + time_horizon]
    price_timeline = np.arange(offset_month, offset_month + len(stochastic_prices))
else:
    # Constant price model starting at offset_month
    time_horizon = 40  # Fixed duration
    stochastic_prices = [token_price] * time_horizon
    price_timeline = np.arange(offset_month, offset_month + time_horizon)

# Bear Market Periods
st.sidebar.header("Bear Market Periods")
bear_market_periods = st.sidebar.text_input("Bear Market Periods (e.g., [(10, 16), (28, 34)])", value="[(10, 16), (28, 34)]")
bear_market_coefficient = st.sidebar.number_input("Bear Market Sell Pressure Coefficient", value=1.5, step=0.1)
try:
    bear_market_periods = eval(bear_market_periods)
except:
    st.sidebar.error("Invalid format for bear market periods. Use [(start, end), ...]")

# Vesting Schedule Parameters
st.sidebar.header("Vesting Schedule Parameters")
vesting_columns = ["Category", "TGE (%)", "Unlock (%)", "Lock-up (months)", "Start Month", "End Month", "Color", "Default SP (%)", "Triggered SP (%)", "Trigger ROI (%)"]
vesting_data = [
    ["Pre-Seed", 0.0, 0.005, 0, 1, 30, "#0000FF", 50, 95, 110],
    ["Seed", 0.0, 0.004, 0, 1, 24, "#008000", 50, 95, 110],
    ["Public Sale", 0.01, 0.0, 0, 0, 0, "#FFA500", 50, 95, 110],
    ["Team/Founders", 0.0, 0.003, 12, 13, 40, "#800080", 50, 95, 110],
    ["Treasury", 0.0, 0.002, 0, 1, 35, "#00FFFF", 50, 95, 110],
    ["Airdrop", 0.015, 0.0, 0, 0, 0, "#FF0000", 50, 95, 110],
    ["Marketing", 0.03, 0.005, 3, 4, 9, "#FFC0CB", 50, 95, 110],
    ["Liquidity", 0.01, 0.0, 0, 0, 0, "#808080", 50, 95, 110],
]
vesting_df = pd.DataFrame(vesting_data, columns=vesting_columns)

# Editable Vesting Schedule
st.write("### Edit Vesting Schedule")
edited_vesting_data = []
for index, row in vesting_df.iterrows():
    cols = st.columns(len(vesting_columns))
    edited_row = []
    for i, col in enumerate(cols):
        unique_key = f"{vesting_columns[i]}_{index}"
        if vesting_columns[i] == "Color":
            value = col.color_picker(f"{vesting_columns[i]} ({index})", value=row[i], key=unique_key)
        else:
            value = col.text_input(f"{vesting_columns[i]} ({index})", value=row[i], key=unique_key)
            try:
                value = float(value) if i > 0 else value
            except ValueError:
                pass
        edited_row.append(value)
    edited_vesting_data.append(edited_row)
vesting_df = pd.DataFrame(edited_vesting_data, columns=vesting_columns)

# Dynamic Market Depth
st.sidebar.header("Dynamic Market Depth")
market_depth_threshold = st.sidebar.number_input("Market Depth Threshold (USD)", value=1_000_000, step=100_000)

# Liquidity Provisioning
st.sidebar.header("Liquidity Provisioning")
liquidity_provisioning = st.sidebar.text_input(
    "Liquidity Provisioning Additions (e.g., {15: 500000, 25: 750000})",
    value="{15: 500000, 25: 750000}"
)
try:
    liquidity_provisioning = eval(liquidity_provisioning)
except:
    st.sidebar.error("Invalid format for liquidity provisioning. Use {month: amount, ...}")

dynamic_market_depth = [market_depth_threshold]
for i in range(1, 40):
    added_liquidity = liquidity_provisioning.get(i, 0)
    dynamic_market_depth.append(dynamic_market_depth[-1] + added_liquidity)

# Rewards Allocation
st.sidebar.header("Rewards Allocation")
reward_allocation_percentage = st.sidebar.slider("Rewards Allocation (% of Total Supply)", 0.0, 100.0, 5.0, 0.1)
logistic_center = st.sidebar.slider("Logistic Center (Months)", 0, 40, 20, 1)
logistic_steepness = st.sidebar.slider("Logistic Steepness", 0.1, 10.0, 1.0, 0.1)

# Generate Unlock Schedules
allocations = {}
for _, entry in vesting_df.iterrows():
    schedule = [0] * 40  # Initialize 40 months
    if entry["TGE (%)"] > 0:
        schedule[0] = entry["TGE (%)"]
    if entry["Unlock (%)"] > 0:
        for month in range(max(0, int(entry["Start Month"])), min(40, int(entry["End Month"]) + 1)):
            if month < offset_month:  # Skip unlocks before offset_month
                continue
            price_roi = (stochastic_prices[month - offset_month] / initial_token_price) * 100
            sell_pressure = entry["Default SP (%)"] / 100  # Default value
            if price_roi > entry["Trigger ROI (%)"]:
                sell_pressure = entry["Triggered SP (%)"] / 100
            if any(start <= month <= end for start, end in bear_market_periods):
                sell_pressure *= bear_market_coefficient  # Apply bear market coefficient
            schedule[month] += entry["Unlock (%)"] * sell_pressure
    # Mask unlocks that occur before offset_month
    schedule = [val if idx >= offset_month else 0 for idx, val in enumerate(schedule)]
    allocations[entry["Category"]] = {"color": entry["Color"], "unlock_schedule": schedule}

# Total unlocks in tokens
x = np.arange(40)
logistic_curve = 1 / (1 + np.exp(-logistic_steepness * (x - logistic_center)))
logistic_curve = logistic_curve / logistic_curve.sum() * (reward_allocation_percentage / 100)
allocations["Rewards"] = {"color": "#FFD700", "unlock_schedule": logistic_curve.tolist()}

total_unlocks_tokens = np.zeros(40)
for data in allocations.values():
    total_unlocks_tokens += np.array(data["unlock_schedule"]) * max_supply

# Total unlocks in USD
adjusted_stochastic_prices = np.array(stochastic_prices[:len(total_unlocks_tokens)])
total_unlocks_usd = total_unlocks_tokens * adjusted_stochastic_prices

dynamic_market_depth = np.array(dynamic_market_depth)

# Calculate Overflow
overflow = [max(0, total_unlocks_usd[i] - dynamic_market_depth[i]) for i in range(40)]

# ROI Calculation
roi = [(price / initial_token_price - 1) * 100 for price in stochastic_prices]

# Plot Unlock Schedule
fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(40)
for name, data in allocations.items():
    unlock_usd = np.array(data["unlock_schedule"]) * max_supply * adjusted_stochastic_prices
    ax.bar(range(40), unlock_usd, bottom=bottom, color=data["color"], label=name, alpha=0.7)
    bottom += unlock_usd

# Overflow Hatching
ax.bar(range(40), overflow, bottom=dynamic_market_depth, color='none', edgecolor='red', hatch='//', label="Overflow")

# Market Depth
ax.step(range(40), dynamic_market_depth, where="mid", color="red", linestyle="--", label="Market Depth")

# Second Y-Axis for Token Price
ax2 = ax.twinx()
ax2.plot(price_timeline, stochastic_prices, color="blue", linestyle="-", linewidth=2, label="Token Price")
ax2.set_ylabel("Token Price (USD)", color="blue")
ax2.tick_params(axis='y', labelcolor='blue')

# Finalize Plot
ax.set_title("Token Unlock Schedule with Dynamic Selling Pressure, Market Depth, and Overflow")
ax.set_xlabel("Months")
ax.set_ylabel("Unlock Value (USD)")
ax.legend(loc="upper left")
ax.grid(False)

st.pyplot(fig)

# ROI and Bear Market Periods Plot
fig2, ax3 = plt.subplots(figsize=(12, 6))

# Range slider below the graph for cumulative overflow
st.write("### Select Range for Cumulative Overflow")
range_start, range_end = st.slider("Select Month Range:", 0, 39, (0, 39))
cumulative_overflow = sum(overflow[range_start:range_end + 1])
st.write(f"**Cumulative Overflow (USD):** {cumulative_overflow:,.2f}")

# Primary axis: Overflow
ax3.fill_between(range(40), 0, overflow, color='red', alpha=0.7, linewidth=2)
ax3.set_ylabel("Overflow (USD)", color='red')
ax3.tick_params(axis='y', labelcolor='red')
ax3.set_ylim(bottom=-0.05, top=max(overflow) * 1.2)  # Adjust scale to ensure visibility

# Secondary axis: ROI
ax4 = ax3.twinx()  # Crée un axe secondaire pour le ROI
ax4.plot(range(40), roi, color='green', linewidth=2, label="ROI (%)")
ax4.set_ylabel("ROI (%)", color='green')
ax4.tick_params(axis='y', labelcolor='green')

# Highlight Bear Market Periods
for start, end in bear_market_periods:
    ax3.axvspan(start, end, color='gray', alpha=0.3, label='Bear Market' if start == bear_market_periods[0][0] else "")

# Finalize the Plot
ax3.set_xlabel("Months")
ax3.set_title("ROI Evolution with Bear Markets and Overflow")

# Add legend
fig2.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
st.pyplot(fig2)
