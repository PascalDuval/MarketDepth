## README: Token Unlock Schedule and Market Depth Provisioning

### **Overview**
This Streamlit-based web application is an **interactive visualization tool** designed for projects involving **tokenomics**. It allows users to simulate and analyze token unlock schedules, market depth provisioning, and various sell pressures that impact token prices over time.

---

### **Key Features**

1. **Tokenomics Configuration**  
   - Set initial token price and choose between **Constant Price** or **Stochastic Price (Black-Scholes)** models.
   - Simulate price changes over a 40-month horizon, incorporating random market behavior (stochastic model).

2. **Customizable Vesting Schedule**  
   - Define vesting schedules for different allocation categories (e.g., Pre-Seed, Team, Public Sale).  
   - Edit parameters like **unlock percentages, lock-up periods**, and vesting duration.  
   - Add or remove allocation rows dynamically.

3. **Bear Market Simulation**  
   - Specify bear market periods where sell pressures increase token unlock rates.  
   - Adjust coefficients to simulate bear market impacts.

4. **Dynamic Market Depth**  
   - Configure market liquidity thresholds and simulate liquidity additions over time.  
   - Analyze potential **overflow** scenarios when unlock values exceed available market depth.

5. **Rewards Allocation**  
   - Simulate rewards distribution using a logistic curve to model gradual release over time.  
   - Adjust curve parameters (center and steepness) interactively.

6. **Sell Pressure Phases**  
   - Simulate varying sell pressures for early, acceleration, and growth phases.  
   - Observe how different phases influence the overall unlock schedule.

7. **Marketing Banner Analysis**  
   - Define marketing campaigns over specific time periods with color-coded banners.  
   - Highlight token overflow values during these campaigns.

8. **Interactive Visualization**  
   - Generate bar charts and line plots to display:
     - Token unlock schedules (USD value).
     - Market depth and liquidity thresholds.
     - Token price trends (constant or stochastic).
     - Overflow areas exceeding market liquidity.  

---

### **How to Use**

1. **Install Streamlit and Dependencies**  
   ```bash
   pip install streamlit matplotlib numpy pandas
   ```

2. **Run the Application**  
   From the project directory, execute:
   ```bash
   streamlit run app.py
   ```

3. **Configure Inputs via Sidebar**  
   - Set tokenomics parameters, vesting schedules, and market conditions.  
   - Add/remove allocations, define bear markets, and adjust rewards or marketing banners.

4. **Visualize Results**  
   - The main chart displays unlock schedules, market depth, token price, and overflow values.  
   - Marketing banners and their respective overflow values are highlighted.

---

### **Target Audience**
- Tokenomics analysts and crypto project teams.  
- Financial modelers seeking to simulate market liquidity and token unlock schedules.  
- Marketing teams assessing campaign performance based on liquidity overflow analysis.

---

### **Technologies Used**
- **Streamlit**: For creating the interactive web interface.  
- **Matplotlib**: For dynamic plotting and data visualization.  
- **NumPy**: For stochastic simulations and mathematical computations.  
- **Pandas**: For handling and manipulating vesting schedule data.

---

### **Disclaimer**
This tool provides simulations and analysis for token unlock scenarios. Users are responsible for validating inputs and outputs for their specific projects.

---

### **License**
MIT License.
