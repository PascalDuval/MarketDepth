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
token_price = st.sidebar.number_input("Token Price (USD)", value=0.1, step=0.01)

# Price Model Selection
st.sidebar.header("Price Model")
price_model = st.sidebar.radio("Choose Price Model", ("Constant Price", "Stochastic Price (Black-Scholes)"))

if price_model == "Stochastic Price (Black-Scholes)":
    mu = st.sidebar.number_input("Expected Return (mu)", value=0.05, step=0.01, format="%.2f")
    sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2, step=0.01, format="%.2f")
    time_horizon = 40  # 40 months
    dt = 1 / 12  # Monthly steps
    time_steps = np.linspace(0, time_horizon * dt, time_horizon)
    
    # Simulate Stochastic Price Path
    np.random.seed(42)
    stochastic_prices = [token_price]
    for t in range(1, time_horizon):
        random_shock = np.random.normal(0, 1)
        price = stochastic_prices[-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_shock)
        stochastic_prices.append(price)
    
    st.sidebar.write("### Simulated Price Path")
    st.sidebar.line_chart(stochastic_prices)
else:
    stochastic_prices = [token_price] * 40  # Constant price for all months

# Bear Market Periods
st.sidebar.header("Bear Market Periods")
bear_market_periods = st.sidebar.text_input(
    "Bear Market Periods (e.g., [(10, 16), (28, 34)])",
    value="[(10, 16), (28, 34)]",
)
bear_market_coefficient = st.sidebar.number_input(
    "Bear Market Sell Pressure Coefficient", value=1.5, step=0.1
)
try:
    bear_market_periods = eval(bear_market_periods)
except:
    st.sidebar.error("Invalid format for bear market periods. Use [(start, end), ...]")

# Vesting Schedule Table
st.sidebar.header("Vesting Schedule Parameters")
vesting_columns = ["Category", "TGE (%)", "Unlock (%)", "Lock-up (months)", "Start Month", "End Month", "Color"]
vesting_data = [
    ["Pre-Seed", 0.0, 0.005, 0, 1, 30, "#0000FF"],
    ["Seed", 0.0, 0.004, 0, 1, 24, "#008000"],
    ["Public Sale", 0.01, 0.0, 0, 0, 0, "#FFA500"],
    ["Team/Founders", 0.0, 0.003, 12, 13, 40, "#800080"],
    ["Treasury", 0.0, 0.002, 0, 1, 35, "#00FFFF"],
    ["Airdrop", 0.015, 0.0, 0, 0, 0, "#FF0000"],
    ["Marketing", 0.03, 0.005, 3, 4, 9, "#FFC0CB"],
    ["Liquidity", 0.01, 0.0, 0, 0, 0, "#808080"],
]
vesting_df = pd.DataFrame(vesting_data, columns=vesting_columns)

# Add/Remove Allocation Rows
if st.sidebar.button("Add New Allocation"):
    vesting_df = pd.concat([
        vesting_df,
        pd.DataFrame([["New Allocation", 0.0, 0.0, 0, 0, 0, "#000000"]], columns=vesting_columns)],
        ignore_index=True
    )

remove_index = st.sidebar.number_input("Index of Allocation to Remove", value=0, min_value=0, max_value=len(vesting_df)-1, step=1)
if st.sidebar.button("Remove Allocation"):
    vesting_df = vesting_df.drop(index=remove_index).reset_index(drop=True)

# Editable Data Table
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
                value = float(value) if i > 0 and i < len(vesting_columns) - 1 else value
            except ValueError:
                pass
        edited_row.append(value)
    edited_vesting_data.append(edited_row)
vesting_df = pd.DataFrame(edited_vesting_data, columns=vesting_columns)

# Dynamic Market Depth
st.sidebar.header("Dynamic Market Depth")
market_depth_threshold = st.sidebar.number_input(
    "Market Depth Threshold (USD)", value=1_000_000, step=100_000
)

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

# Selling Coefficients by Phase
st.sidebar.header("Selling Pressure")
early_phase_coeff = st.sidebar.number_input("Early Phase Sell Pressure (%)", value=10.0, step=1.0)
acceleration_phase_coeff = st.sidebar.number_input("Acceleration Phase Sell Pressure (%)", value=50.0, step=1.0)
growth_phase_coeff = st.sidebar.number_input("Growth Phase Sell Pressure (%)", value=80.0, step=1.0)

# Marketing Banners
st.sidebar.header("Marketing Banners")
marketing_banners = []
num_banners = st.sidebar.number_input("Number of Marketing Banners", value=2, step=1)
for i in range(num_banners):
    st.sidebar.subheader(f"Banner {i+1}")
    start = st.sidebar.number_input(f"Start Month (Banner {i+1})", value=5, step=1, key=f"banner_start_{i}")
    end = st.sidebar.number_input(f"End Month (Banner {i+1})", value=10, step=1, key=f"banner_end_{i}")
    color = st.sidebar.color_picker(f"Color (Banner {i+1})", value="#FFDDC1", key=f"banner_color_{i}")
    marketing_banners.append({"start": start, "end": end, "color": color})

# Generate Unlock Schedules
allocations = {}
for _, entry in vesting_df.iterrows():
    schedule = [0] * 40  # Initialize 40 months
    if entry["TGE (%)"] > 0:
        schedule[0] = entry["TGE (%)"]
    if entry["Unlock (%)"] > 0:
        for month in range(
            max(0, int(entry["Start Month"])), min(40, int(entry["End Month"]) + 1)
        ):
            # Apply sell pressure phases
            pressure = 1.0  # Default no bear market adjustment
            if any(start <= month <= end for start, end in bear_market_periods):
                pressure *= bear_market_coefficient
            
            if month < 6:
                pressure *= early_phase_coeff / 100
            elif 6 <= month < 18:
                pressure *= acceleration_phase_coeff / 100
            else:
                pressure *= growth_phase_coeff / 100

            schedule[month] += entry["Unlock (%)"] * pressure
    allocations[entry["Category"]] = {"color": entry["Color"], "unlock_schedule": schedule}

# Add Rewards Allocation
x = np.arange(40)
logistic_curve = 1 / (1 + np.exp(-logistic_steepness * (x - logistic_center)))
logistic_curve = logistic_curve / logistic_curve.sum() * (reward_allocation_percentage / 100)
allocations["Rewards"] = {"color": "#FFD700", "unlock_schedule": logistic_curve.tolist()}

# Recalculate total unlocks
total_unlocks_tokens = np.zeros(40)
for data in allocations.values():
    total_unlocks_tokens += np.array(data["unlock_schedule"]) * max_supply

# Total unlocks in USD
total_unlocks_usd = total_unlocks_tokens * np.array(stochastic_prices)

# Overflow Calculation
overflow = [max(0, total_unlocks_usd[i] - dynamic_market_depth[i]) for i in range(40)]

# Sum overflow per marketing banner
banner_overflows = []
for banner in marketing_banners:
    banner_sum = sum(overflow[banner["start"]:banner["end"] + 1])
    banner_overflows.append(banner_sum)

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

bottom = np.zeros(40)
for name, data in allocations.items():
    unlock_usd = np.array(data["unlock_schedule"]) * max_supply * np.array(stochastic_prices)
    ax1.bar(range(40), unlock_usd, bottom=bottom, color=data["color"], alpha=0.7, label=name)
    bottom += unlock_usd

# Overflow Highlight
ax1.bar(range(40), overflow, bottom=dynamic_market_depth, color="none", edgecolor="red", hatch="//", label="Overflow")

# Market Depth Line
ax1.step(range(40), dynamic_market_depth, where='mid', color='black', linestyle='--', linewidth=2, label="Market Depth")

# Marketing Banners with Overflow Annotation
for i, banner in enumerate(marketing_banners):
    ax1.axvspan(banner["start"], banner["end"], color=banner["color"], alpha=0.3, label=f"Banner {banner['start']}-{banner['end']}")
    ax1.text((banner["start"] + banner["end"]) / 2, max(dynamic_market_depth) * 1.1, 
             f"Overflow: ${banner_overflows[i]:,.2f}",
             fontsize=9, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))

# Price Line
ax2 = ax1.twinx()
ax2.plot(range(40), stochastic_prices, color="blue", linestyle="-", linewidth=2, label="Token Price")
ax2.set_ylabel("Token Price (USD)", color="blue")
ax2.tick_params(axis='y', labelcolor='blue')

# Final Formatting
ax1.set_title("Token Unlock Schedule with Rewards, Market Depth, Overflow, and Banners")
ax1.set_xlabel("Months")
ax1.set_ylabel("Unlock Value (USD)")
ax1.legend(loc="upper left")
ax1.grid(False)

st.pyplot(fig)
