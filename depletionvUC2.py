import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Dual Depletion Test: Token Allocation and Bonus", layout="wide")

# Titre principal
st.title("Dual Depletion Test: Token Allocation and Bonus Depletion by Phase")

# Paramètres non modifiables
st.sidebar.header("Paramètres de l'allocation")
max_supply = 1_000_000_000  # Valeur fixe pour l'offre maximale
allocation_base_pct = 6.0    # Valeur fixe pour le pourcentage de l'allocation de base
allocation_bonus_pct = 2.0   # Valeur fixe pour le pourcentage de l'allocation de bonus
listing_price = 0.7          # Valeur fixe pour le prix de cotation

# Affichage des paramètres non modifiables dans la barre latérale
st.sidebar.write(f"Max Supply: {max_supply}")
st.sidebar.write(f"Allocation Base (%): {allocation_base_pct}")
st.sidebar.write(f"Allocation Bonus (%): {allocation_bonus_pct}")
st.sidebar.write(f"Listing Price ($): {listing_price}")

# Calcul des allocations
total_tokens_for_employees = max_supply * (allocation_base_pct / 100)
total_tokens_for_bonus = max_supply * (allocation_bonus_pct / 100)

# Données des phases de financement et nombre d'employés (fixes)
fundraising_stages = {
    "Phase": ["Pre-seed", "Strategic Angels", "Seed", "Strategic Bridge", "Series A", "Private Sale", "TGE", "Revenues"],
    "Employees": [7, 3, 20, 15, 20, 5, 10, 15]
}

# Affichage des phases et du nombre d'employés pour chaque phase dans la barre latérale (non modifiable)
st.sidebar.write("### Nombre d'employés par phase")
for phase, employees in zip(fundraising_stages["Phase"], fundraising_stages["Employees"]):
    st.sidebar.write(f"{phase}: {employees} employés")

# Données des coefficients de risque et valeurs fixes
risk_coefficients = [1.0, 0.598, 0.5, 0.32, 0.267, 0.218, 0.19, 0.19]
SL_values = {"Entry Level": 1, "Junior": 1.5, "Mid-Level": 2, "Senior": 2.5, "Lead/Principal": 3, "Manager": 2.5, "Division": 3}
RI_values = {"Engineering": 1, "Business Dev.": 1.5, "Legal": 2, "Marketing": 2.5, "Operations": 3, "Support": 1}
SC_values = {"<100k$": 1.2, "100-150k$": 1.1, "150-200k$": 1, "200k-250k$": 0.9, ">250k$": 0.8}

# Paramètres de performance pour l'allocation de bonus
IP_values = {"Needs Improvement": 2, "Meets Expectations": 1.8, "Exceeds Expectations": 1.5, "Outstanding": 1.2, "Exceptional": 1}
PI_values = {"Standard": 1, "High Impact": 1.25, "Critical Success": 1.5}
IC_values = {"Standard": 1, "Notable Innovation": 1.15, "Significant Innovation": 1.3}
TA_values = {"0-2 Years": 1, "2-4 Years": 1.1, "4+ Years": 1.2}

# SL, RI, SC par phase, modifiable par l'utilisateur
center_choices = {
    phase: {
        "SL": st.sidebar.selectbox(f"{phase} - SL", list(SL_values.keys()), index=list(SL_values.keys()).index("Manager")),
        "RI": st.sidebar.selectbox(f"{phase} - RI", list(RI_values.keys()), index=list(RI_values.keys()).index("Engineering")),
        "SC": st.sidebar.selectbox(f"{phase} - SC", list(SC_values.keys()), index=list(SC_values.keys()).index("100-150k$"))
    }
    for phase in fundraising_stages["Phase"]
}

# IP, PI, IC, TA pour allocation de bonus par phase, modifiable par l'utilisateur
center_choices_bonus = {
    phase: {
        "IP": st.sidebar.selectbox(f"{phase} - IP", list(IP_values.keys()), index=list(IP_values.keys()).index("Meets Expectations")),
        "PI": st.sidebar.selectbox(f"{phase} - PI", list(PI_values.keys()), index=list(PI_values.keys()).index("Standard")),
        "IC": st.sidebar.selectbox(f"{phase} - IC", list(IC_values.keys()), index=list(IC_values.keys()).index("Standard")),
        "TA": st.sidebar.selectbox(f"{phase} - TA", list(TA_values.keys()), index=list(TA_values.keys()).index("2-4 Years"))
    }
    for phase in fundraising_stages["Phase"]
}

# Fonction pour obtenir le coefficient JT
def get_jt(employees):
    JT_values = {(0, 10): 2.0, (10, 25): 1.8, (25, 50): 1.5, (50, 100): 2.5, (100, 150): 1.0}
    for (min_emp, max_emp), jt_value in JT_values.items():
        if min_emp <= employees < max_emp:
            return jt_value
    return 1

# Calcul des coefficients ajustés pour base et bonus allocations
def calculate_adjusted_coefficient(stage, employees):
    SL = SL_values[center_choices[stage]["SL"]]
    RI = RI_values[center_choices[stage]["RI"]]
    SC = SC_values[center_choices[stage]["SC"]]
    JT = get_jt(employees)
    return (SL + RI + SC + JT) / 4

def calculate_adjusted_coefficient_bonus(stage, employees):
    IP = IP_values[center_choices_bonus[stage]["IP"]]
    PI = PI_values[center_choices_bonus[stage]["PI"]]
    IC = IC_values[center_choices_bonus[stage]["IC"]]
    TA = TA_values[center_choices_bonus[stage]["TA"]]
    return (IP + PI + IC + TA) / 4

# Simulation de déplétion
def simulate_depletion(allocation, btu, risk_coefficients, employees_per_phase, stages, adjust_func):
    remaining_allocation = []
    adjusted_tokens_per_employee = []
    for employees, risk_coef, stage in zip(employees_per_phase, risk_coefficients, stages):
        adjusted_coefficient = adjust_func(stage, employees)
        tokens_per_employee = btu * risk_coef * adjusted_coefficient
        adjusted_tokens_per_employee.append(tokens_per_employee)
        for _ in range(employees):
            if allocation > 0:
                remaining_allocation.append(allocation / 1_000_000)
                allocation -= tokens_per_employee
            else:
                remaining_allocation.append(0)
    return remaining_allocation, allocation / 1_000_000, adjusted_tokens_per_employee

# Calculs pour les allocations de base et bonus
BTU_base = total_tokens_for_employees / sum(fundraising_stages["Employees"])
BTU_bonus = total_tokens_for_bonus / sum(fundraising_stages["Employees"])

remaining_base, final_reserve_base, adjusted_tokens_per_employee_base = simulate_depletion(
    total_tokens_for_employees, BTU_base, risk_coefficients, fundraising_stages["Employees"], fundraising_stages["Phase"], calculate_adjusted_coefficient
)
remaining_bonus, final_reserve_bonus, adjusted_tokens_per_employee_bonus = simulate_depletion(
    total_tokens_for_bonus, BTU_bonus, risk_coefficients, fundraising_stages["Employees"], fundraising_stages["Phase"], calculate_adjusted_coefficient_bonus
)

# Affichage des graphiques de déplétion
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Graphique de déplétion de l'allocation de base
ax1.plot(range(len(remaining_base)), remaining_base, marker='.', markersize=4, color='purple', label="Base Allocation Depletion")
ax1.set_xlabel("Number of Employees Hired")
ax1.set_ylabel("Remaining Allocation (Millions)")
ax1.set_title("Depletion of Base Allocation with Adjusted Coefficients")
ax1.legend()
ax1.text(len(remaining_base)-1, remaining_base[-1] + 2, f'Final Reserve: {final_reserve_base:.2f}M', color='purple', ha='right', fontsize=10, weight='bold')

# Graphique de déplétion de l'allocation de bonus
ax2.plot(range(len(remaining_bonus)), remaining_bonus, marker='.', markersize=4, color='orange', label="Bonus Allocation Depletion")
ax2.set_xlabel("Number of Employees Hired")
ax2.set_ylabel("Remaining Allocation (Millions)")
ax2.set_title("Depletion of Bonus Allocation with Adjusted Coefficients")
ax2.legend()
ax2.text(len(remaining_bonus)-1, remaining_bonus[-1] + 2, f'Final Reserve: {final_reserve_bonus:.2f}M', color='orange', ha='right', fontsize=10, weight='bold')

# Affichage dans Streamlit
st.pyplot(fig)

# Histogramme pour la compensation par employé en dollars
compensation_per_employee_dollars = [tokens * listing_price for tokens in adjusted_tokens_per_employee_base]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(fundraising_stages["Phase"], compensation_per_employee_dollars, color='purple')
ax.set_xlabel("Phase")
ax.set_ylabel("Cost per Employee ($)")
ax.set_title("Base Compensation per Employee by Phase in Dollars @TGE listing price")

for bar, compensation in zip(bars, compensation_per_employee_dollars):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"${compensation:,.0f}", ha='center', va='bottom', fontsize=10, color='black')

# Afficher le graphique dans Streamlit
st.pyplot(fig)
