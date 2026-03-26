import streamlit as st
import textwrap
import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import json
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import random

# Page configuration
st.set_page_config(
    page_title="FIN 231: Financial Management Mastery",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning design
st.markdown(textwrap.dedent("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    .card {
        background: rgba(102, 126, 234, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
        border-top: 1px solid rgba(102, 126, 234, 0.1);
        border-right: 1px solid rgba(102, 126, 234, 0.1);
        border-bottom: 1px solid rgba(102, 126, 234, 0.1);
        transition: transform 0.3s ease;
        color: inherit !important;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
        color: inherit !important;
    }
    
    .formula-box {
        background: #1e1e2e;
        color: #cdd6f4;
        padding: 1.5rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
        border: 2px solid #89b4fa;
    }
    
    .highlight {
        background: linear-gradient(120deg, #89b4fa 0%, #b4befe 100%);
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        color: inherit !important;
        border-bottom: 2px solid transparent !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    .quiz-option {
        background: rgba(102, 126, 234, 0.05);
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
        color: inherit !important;
    }
    
    .quiz-option:hover {
        border-color: #667eea;
        background: #e7e9ff;
    }
    
    .progress-bar {
        height: 10px;
        background: #e0e0e0;
        border-radius: 5px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.5s ease;
    }
    
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #667eea;
        cursor: help;
    }
    
    .info-badge {
        background: rgba(137, 180, 250, 0.2);
        color: #89b4fa !important;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(137, 180, 250, 0.3);
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: inherit !important;
    }
    
    .success-box {
        background: rgba(40, 167, 69, 0.1);
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: inherit !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stSlider>div>div>div {
        background: #667eea !important;
    }
</style>
"""), unsafe_allow_html=True)

# Initialize session state
if 'quiz_scores' not in st.session_state:
    st.session_state.quiz_scores = {}
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = {}
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# Helper functions
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price, d1, d2

def calculate_z_score(wc_ta, re_ta, ebit_ta, mve_tl, s_ta):
    """Calculate Altman Z-Score"""
    z = 1.2 * wc_ta + 1.4 * re_ta + 3.3 * ebit_ta + 0.6 * mve_tl + 1.0 * s_ta
    return z

def get_z_interpretation(z):
    """Interpret Z-Score"""
    if z > 2.99:
        return "Safe Zone", "green", "Low probability of bankruptcy"
    elif z > 1.81:
        return "Grey Zone", "orange", "Moderate risk of financial distress"
    else:
        return "Distress Zone", "red", "High probability of bankruptcy"

# Sidebar navigation
st.sidebar.markdown(textwrap.dedent("""
<div style="text-align: center; padding: 1rem;">
    <h2 style="color: #667eea;">📚 FIN 231</h2>
    <p style="color: #666;">Financial Management</p>
</div>
"""), unsafe_allow_html=True)

# Course progress
total_units = 8
completed = len(st.session_state.quiz_scores)
progress = completed / total_units

st.sidebar.markdown(textwrap.dedent(f"""
<div style="margin: 2rem 0;">
    <p style="font-size: 0.9rem; color: #666;">Course Progress</p>
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress * 100}%"></div>
    </div>
    <p style="font-size: 0.8rem; color: #999; margin-top: 0.5rem;">{completed}/{total_units} units completed</p>
</div>
"""), unsafe_allow_html=True)

# Navigation
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "📊 Unit 1: Introduction", "⚖️ Unit 2: Capital Structure", 
     "💵 Unit 3: Dividend Policy", "🏦 Unit 4: Raising Capital",
     "📈 Unit 5: Short-Term Planning", "💼 Unit 6: Working Capital",
     "🎯 Unit 7: Derivatives", "⚠️ Unit 8: Special Topics",
     "🎮 Financial Simulator", "📉 Market Data Lab", "🧮 Advanced Calculators"]
)

# Home Page
if page == "🏠 Home":
    st.markdown(textwrap.dedent("""
    <div class="main-header">
        <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">💰 FIN 231</h1>
        <h2 style="font-weight: 300; opacity: 0.9;">Financial Management Mastery</h2>
        <p style="font-size: 1.2rem; margin-top: 1rem; opacity: 0.8;">
            Interactive Learning Platform • Advanced Analytics • Real-World Applications
        </p>
    </div>
    """), unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(textwrap.dedent("""
        <div class="card">
            <h3 style="color: #667eea;">🎓 Comprehensive Curriculum</h3>
            <p>8 detailed units covering all aspects of corporate finance from capital structure to derivatives.</p>
            <ul style="margin-top: 1rem;">
                <li>Interactive visualizations</li>
                <li>Real-time calculations</li>
                <li>Case study analyses</li>
            </ul>
        </div>
        """), unsafe_allow_html=True)
    
    with col2:
        st.markdown(textwrap.dedent("""
        <div class="card">
            <h3 style="color: #667eea;">🧮 Advanced Tools</h3>
            <p>Professional-grade financial calculators and simulators for hands-on learning.</p>
            <ul style="margin-top: 1rem;">
                <li>Black-Scholes calculator</li>
                <li>Z-Score predictor</li>
                <li>Leverage analyzer</li>
                <li>CCC optimizer</li>
            </ul>
        </div>
        """), unsafe_allow_html=True)
    
    with col3:
        st.markdown(textwrap.dedent("""
        <div class="card">
            <h3 style="color: #667eea;">📊 Live Market Data</h3>
            <p>Connect with real financial markets for practical application of concepts.</p>
            <ul style="margin-top: 1rem;">
                <li>Stock price analysis</li>
                <li>Volatility calculations</li>
                <li>Capital structure comparisons</li>
            </ul>
        </div>
        """), unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("---")
    st.subheader("📊 Platform Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Total Topics", "50+", "+12 advanced")
    with stat_col2:
        st.metric("Interactive Tools", "15", "+5 simulators")
    with stat_col3:
        st.metric("Practice Problems", "100+", "with solutions")
    with stat_col4:
        st.metric("Case Studies", "20", "real-world")
    
    # Course roadmap
    st.markdown("---")
    st.subheader("🗺️ Learning Roadmap")
    
    roadmap_data = pd.DataFrame({
        'Unit': ['Unit 1', 'Unit 2', 'Unit 3', 'Unit 4', 'Unit 5', 'Unit 6', 'Unit 7', 'Unit 8'],
        'Topic': ['Introduction', 'Capital Structure', 'Dividend Policy', 'Raising Capital', 
                  'Short-Term Planning', 'Working Capital', 'Derivatives', 'Special Topics'],
        'Hours': [5, 6, 5, 6, 6, 11, 5, 4],
        'Difficulty': ['Easy', 'Medium', 'Medium', 'Hard', 'Medium', 'Hard', 'Hard', 'Medium']
    })
    
    fig = px.timeline(roadmap_data, x_start=[0, 5, 11, 16, 22, 28, 39, 44], 
                      x_end=[5, 11, 16, 22, 28, 39, 44, 48],
                      y="Unit", color="Difficulty",
                      color_discrete_map={'Easy': '#28a745', 'Medium': '#ffc107', 'Hard': '#dc3545'})
    fig.update_layout(title="Course Timeline (Hours)", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Unit 1: Introduction
elif page == "📊 Unit 1: Introduction":
    st.title("📊 Unit 1: Introduction to Financial Management")
    
    tabs = st.tabs(["📚 Theory", "🎯 Agency Theory", "⚖️ Governance", "🧮 Calculator", "📝 Quiz"])
    
    with tabs[0]:
        st.markdown(textwrap.dedent("""
        <div class="card">
            <h3>Definition & Scope</h3>
            <p>Financial management is the <span class="highlight">strategic planning</span>, organizing, directing, 
            and controlling of financial undertakings in an organization.</p>
        </div>
        """), unsafe_allow_html=True)
        
        # Evolution timeline
        years = ['1950s', '1960s-70s', '1980s-90s', '2000s-Present']
        focus = ['Traditional', 'Modern', 'Strategic', 'Integrated']
        concerns = ['Fund Raising', 'Asset Management', 'Risk Management', 'Sustainability']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=[1, 2, 3, 4],
            mode='lines+markers+text',
            text=focus,
            textposition="top center",
            line=dict(color='#667eea', width=4),
            marker=dict(size=20, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
        ))
        fig.update_layout(title="Evolution of Financial Management", 
                         yaxis_visible=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(textwrap.dedent("""
            <div class="card">
                <h4>Key Decisions</h4>
                <ul>
                    <li><b>Investment:</b> Capital budgeting, asset allocation</li>
                    <li><b>Financing:</b> Capital structure, funding sources</li>
                    <li><b>Dividend:</b> Payout policy, retention ratio</li>
                </ul>
            </div>
            """), unsafe_allow_html=True)
        
        with col2:
            st.markdown(textwrap.dedent("""
            <div class="card">
                <h4>Value Maximization</h4>
                <div class="formula-box">
                    Firm Value = Σ(FCFₜ / (1+WACC)ᵗ)
                </div>
                <p style="margin-top: 1rem;">Superior to profit maximization as it considers:</p>
                <ul>
                    <li>Time value of money</li>
                    <li>Risk-return tradeoff</li>
                    <li>Long-term sustainability</li>
                </ul>
            </div>
            """), unsafe_allow_html=True)
    
    with tabs[1]:
        st.subheader("Agency Theory & Relationships")
        
        # Interactive agency diagram
        st.markdown(textwrap.dedent("""
        <div style="background: rgba(102, 126, 234, 0.05); padding: 2rem; border-radius: 15px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.1);">
            <div style="display: flex; justify-content: space-around; align-items: center;">
                <div style="background: #667eea; color: white; padding: 1.5rem; border-radius: 50%; width: 150px; height: 150px; display: flex; align-items: center; justify-content: center; flex-direction: column; box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);">
                    <b>Shareholders</b>
                    <small>Principal</small>
                </div>
                <div style="flex: 1; text-align: center; padding: 0 2rem;">
                    <div style="border: 3px dashed #dc3545; padding: 1rem; border-radius: 10px; background: rgba(220, 53, 69, 0.05); color: inherit !important;">
                        <b style="color: #dc3545;">Agency Problem</b><br>
                        <small>Divergence of interests<br>Information asymmetry</small>
                    </div>
                </div>
                <div style="background: #764ba2; color: white; padding: 1.5rem; border-radius: 50%; width: 150px; height: 150px; display: flex; align-items: center; justify-content: center; flex-direction: column; box-shadow: 0 10px 20px rgba(118, 75, 162, 0.3);">
                    <b>Management</b>
                    <small>Agent</small>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Agency costs breakdown
        cost_type = st.selectbox("Select Agency Cost Type:", 
                                ["Monitoring Costs", "Bonding Costs", "Residual Loss"])
        
        if cost_type == "Monitoring Costs":
            st.info(textwrap.dedent("""
            **Monitoring Costs:** Expenses incurred by principals to supervise agents
            - Audit fees and external audits
            - Board of directors oversight
            - Performance reporting systems
            - External consultant fees
            """))
        elif cost_type == "Bonding Costs":
            st.info(textwrap.dedent("""
            **Bonding Costs:** Expenses incurred by agents to guarantee they will act in principals' interest
            - Management guarantees and commitments
            - Restrictive covenants in contracts
            - Professional liability insurance
            - Performance bonds
            """))
        else:
            st.info(textwrap.dedent("""
            **Residual Loss:** Value reduction due to divergence of interests despite monitoring
            - Suboptimal investment decisions
            - Excessive risk aversion or risk-taking
            - Perquisite consumption
            - Empire building behavior
            """))
        
        # Mitigation mechanisms
        st.subheader("Mitigation Mechanisms")
        mechanisms = {
            "Compensation Design": ["Stock options", "Restricted stock", "Performance shares", "Clawback provisions"],
            "Board Structure": ["Independent directors", "Separation of CEO/Chair", "Board committees", "Regular evaluations"],
            "Market Discipline": ["Takeover threats", "Activist investors", "Proxy contests", "Shareholder litigation"],
            "Debt Discipline": ["Leverage constraints", "Covenant restrictions", "Credit monitoring", "Bankruptcy threat"]
        }
        
        mech_cols = st.columns(4)
        for idx, (mech, items) in enumerate(mechanisms.items()):
            with mech_cols[idx]:
                st.markdown(textwrap.dedent(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 1rem; border-radius: 10px; height: 200px;">
                    <h4 style="margin-bottom: 1rem;">{mech}</h4>
                    <ul style="font-size: 0.9rem;">
                        {''.join([f'<li>{item}</li>' for item in items])}
                    </ul>
                </div>
                """), unsafe_allow_html=True)
    
    with tabs[2]:
        st.subheader("Corporate Governance Framework")
        
        # Interactive governance structure
        governance_layer = st.select_slider(
            "Explore Governance Layers:",
            options=["Shareholders", "Board of Directors", "Committees", "Management", "Operations"]
        )
        
        layer_info = {
            "Shareholders": {
                "role": "Ultimate owners with voting rights",
                "powers": ["Elect directors", "Approve major transactions", "Amend bylaws", "Initiate proxy contests"],
                "responsibilities": ["Monitor performance", "Engage on ESG issues", "Vote responsibly"]
            },
            "Board of Directors": {
                "role": "Strategic oversight and monitoring",
                "powers": ["Hire/fire CEO", "Set compensation", "Declare dividends", "Approve strategy"],
                "responsibilities": ["Fiduciary duty", "Risk oversight", "Succession planning"]
            },
            "Committees": {
                "role": "Specialized oversight functions",
                "powers": ["Audit committee: Financial integrity", "Compensation committee: Executive pay", 
                          "Nominating committee: Board composition"],
                "responsibilities": ["Independent oversight", "Expertise application", "Reporting to board"]
            },
            "Management": {
                "role": "Day-to-day operations execution",
                "powers": ["Operational decisions", "Resource allocation", "Personnel management"],
                "responsibilities": ["Stewardship", "Ethical conduct", "Performance delivery"]
            },
            "Operations": {
                "role": "Value creation activities",
                "powers": ["Process execution", "Customer relations", "Innovation"],
                "responsibilities": ["Quality assurance", "Compliance", "Efficiency"]
            }
        }
        
        info = layer_info[governance_layer]
        st.markdown(textwrap.dedent(f"""
        <div class="card" style="border-left-color: #28a745;">
            <h3>{governance_layer}</h3>
            <p><b>Role:</b> {info['role']}</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 1rem;">
                <div>
                    <h4 style="color: #667eea;">Powers</h4>
                    <ul>{''.join([f'<li>{p}</li>' for p in info['powers']])}</ul>
                </div>
                <div>
                    <h4 style="color: #764ba2;">Responsibilities</h4>
                    <ul>{''.join([f'<li>{r}</li>' for r in info['responsibilities']])}</ul>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)
        
        # Governance codes comparison
        st.subheader("Global Governance Standards")
        
        code_data = pd.DataFrame({
            'Jurisdiction': ['USA', 'UK', 'EU', 'OECD'],
            'Primary Code': ['Sarbanes-Oxley', 'UK Corporate Governance Code', 'Shareholder Rights Directive', 'OECD Principles'],
            'Key Feature': ['Internal controls', 'Comply or explain', 'Transparency', 'International standards'],
            'Focus Area': ['Audit/Accounting', 'Board effectiveness', 'Investor protection', 'Comprehensive framework']
        })
        
        st.dataframe(code_data, use_container_width=True, hide_index=True)
    
    with tabs[3]:
        st.subheader("Financial Management Calculator")
        
        calc_type = st.radio("Select Calculation:", 
                            ["Firm Valuation", "Agency Cost Estimator", "Governance Score"])
        
        if calc_type == "Firm Valuation":
            col1, col2 = st.columns(2)
            with col1:
                fcf = st.number_input("Free Cash Flow ($)", value=1000000, step=100000)
                growth = st.slider("Growth Rate (%)", 0.0, 10.0, 3.0) / 100
                wacc = st.slider("WACC (%)", 5.0, 20.0, 10.0) / 100
            
            with col2:
                if wacc <= growth:
                    st.error("WACC must be greater than growth rate for Gordon model")
                else:
                    firm_value = fcf * (1 + growth) / (wacc - growth)
                    st.metric("Firm Value", f"${firm_value:,.2f}")
                    
                    # Sensitivity analysis
                    growth_range = np.linspace(0, wacc - 0.01, 50)
                    values = [fcf * (1 + g) / (wacc - g) for g in growth_range]
                    
                    fig = px.line(x=growth_range*100, y=values, 
                                 labels={'x': 'Growth Rate (%)', 'y': 'Firm Value ($)'},
                                 title="Value Sensitivity to Growth")
                    fig.add_vline(x=growth*100, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
        
        elif calc_type == "Agency Cost Estimator":
            firm_size = st.number_input("Firm Market Cap ($M)", value=1000, step=100)
            leverage = st.slider("Debt Ratio (%)", 0, 100, 30) / 100
            board_independence = st.slider("Board Independence (%)", 0, 100, 60) / 100
            
            # Estimated agency costs based on research
            base_cost = 0.05  # 5% of firm value
            leverage_reduction = leverage * 0.03  # Debt reduces agency costs
            board_reduction = board_independence * 0.02  # Independent boards help
            
            agency_cost_pct = max(0.01, base_cost - leverage_reduction - board_reduction)
            agency_cost = firm_size * agency_cost_pct
            
            st.metric("Estimated Annual Agency Cost", f"${agency_cost:.2f}M", 
                     f"{agency_cost_pct*100:.1f}% of firm value")
            
            # Visualization
            categories = ['Monitoring', 'Bonding', 'Residual Loss']
            values = [agency_cost * 0.4, agency_cost * 0.2, agency_cost * 0.4]
            
            fig = px.pie(values=values, names=categories, title="Agency Cost Breakdown",
                        color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'])
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # Governance Score
            st.write("Rate your organization's governance (1-10):")
            board_ind = st.slider("Board Independence", 1, 10, 7)
            audit_quality = st.slider("Audit Committee Quality", 1, 10, 8)
            transparency = st.slider("Disclosure Transparency", 1, 10, 7)
            shareholder_rights = st.slider("Shareholder Rights", 1, 10, 6)
            
            score = (board_ind + audit_quality + transparency + shareholder_rights) / 4
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Governance Score", f"{score:.1f}/10")
                if score >= 8:
                    st.success("Excellent governance practices")
                elif score >= 6:
                    st.warning("Good governance with room for improvement")
                else:
                    st.error("Governance weaknesses identified")
            
            with col2:
                categories = ['Board Ind.', 'Audit Quality', 'Transparency', 'Shareholder Rights']
                values = [board_ind, audit_quality, transparency, shareholder_rights]
                
                fig = go.Figure(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name='Your Score'
                ))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                                showlegend=False, title="Governance Radar")
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:
        st.subheader("Unit 1 Assessment")
        
        questions = [
            {
                "question": "Which of the following best describes the primary goal of financial management?",
                "options": ["Profit maximization", "Revenue maximization", "Shareholder wealth maximization", "Market share maximization"],
                "correct": 2,
                "explanation": "Shareholder wealth maximization considers time value, risk, and long-term value creation."
            },
            {
                "question": "Agency costs arise due to:",
                "options": ["High interest rates", "Separation of ownership and control", "Government regulations", "Technological changes"],
                "correct": 1,
                "explanation": "Agency costs occur when principals (shareholders) hire agents (managers) with divergent interests."
            },
            {
                "question": "Which is NOT a type of agency cost?",
                "options": ["Monitoring costs", "Bonding costs", "Transaction costs", "Residual loss"],
                "correct": 2,
                "explanation": "Transaction costs are market friction costs, not specific to principal-agent relationships."
            }
        ]
        
        score = 0
        for idx, q in enumerate(questions):
            st.markdown(f"**Q{idx+1}. {q['question']}**")
            answer = st.radio(f"Select answer for Q{idx+1}:", q['options'], key=f"q1_{idx}")
            
            if st.button(f"Check Answer {idx+1}", key=f"check1_{idx}"):
                if q['options'].index(answer) == q['correct']:
                    st.success(f"✅ Correct! {q['explanation']}")
                    score += 1
                else:
                    st.error(f"❌ Incorrect. {q['explanation']}")
        
        if st.button("Submit Quiz"):
            st.session_state.quiz_scores['Unit 1'] = score
            st.balloons()
            st.success(f"Quiz submitted! Score: {score}/{len(questions)}")

# Unit 2: Capital Structure
elif page == "⚖️ Unit 2: Capital Structure":
    st.title("⚖️ Unit 2: Capital Structure & Leverage")
    
    tabs = st.tabs(["📚 Theory", "🎯 Leverage Analysis", "📊 MM Theorems", "🧮 Calculator", "📝 Quiz"])
    
    with tabs[0]:
        st.markdown(textwrap.dedent("""
        <div class="card">
            <h3>Capital Structure Fundamentals</h3>
            <p>The mix of <span class="highlight">debt</span> and <span class="highlight">equity</span> financing 
            that maximizes firm value by balancing tax benefits against financial distress costs.</p>
        </div>
        """), unsafe_allow_html=True)
        
        # Capital structure visualization
        structure_type = st.selectbox("Select Capital Structure:", 
                                     ["All Equity", "Conservative", "Moderate", "Aggressive", "Highly Levered"])
        
        structures = {
            "All Equity": (0, 100),
            "Conservative": (20, 80),
            "Moderate": (40, 60),
            "Aggressive": (60, 40),
            "Highly Levered": (80, 20)
        }
        
        debt_pct, equity_pct = structures[structure_type]
        
        fig = go.Figure(go.Pie(
            labels=['Debt', 'Equity'],
            values=[debt_pct, equity_pct],
            hole=0.4,
            marker_colors=['#e74c3c', '#3498db'],
            textinfo='label+percent',
            textfont_size=20
        ))
        fig.update_layout(title=f"{structure_type} Capital Structure", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk analysis
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Business Risk", "Medium", "Operating leverage")
        with col2:
            financial_risk = "High" if debt_pct > 50 else "Medium" if debt_pct > 30 else "Low"
            st.metric("Financial Risk", financial_risk, "From debt")
        with col3:
            total_risk = "High" if (debt_pct > 60 or debt_pct < 20) else "Optimal"
            st.metric("Total Risk Profile", total_risk)
    
    with tabs[1]:
        st.subheader("Operating, Financial & Total Leverage")
        
        # Interactive leverage calculator
        col1, col2 = st.columns(2)
        with col1:
            units = st.number_input("Units Sold", value=10000, step=1000)
            price = st.number_input("Price per Unit ($)", value=100.0, step=10.0)
            vc_per_unit = st.number_input("Variable Cost per Unit ($)", value=60.0, step=5.0)
            fixed_costs = st.number_input("Fixed Costs ($)", value=250000, step=50000)
            interest = st.number_input("Interest Expense ($)", value=50000, step=10000)
        
        with col2:
            # Calculations
            sales = units * price
            variable_costs = units * vc_per_unit
            contribution = sales - variable_costs
            ebit = contribution - fixed_costs
            ebt = ebit - interest
            
            # Leverage ratios
            dol = contribution / ebit if ebit != 0 else 0
            dfl = ebit / ebt if ebt != 0 else 0
            dtl = dol * dfl
            
            st.metric("Degree of Operating Leverage (DOL)", f"{dol:.2f}")
            st.metric("Degree of Financial Leverage (DFL)", f"{dfl:.2f}")
            st.metric("Degree of Total Leverage (DTL)", f"{dtl:.2f}")
            
            st.info(textwrap.dedent(f"""
            **Interpretation:**
            - 1% increase in sales → {dol:.2f}% increase in EBIT
            - 1% increase in EBIT → {dfl:.2f}% increase in EPS
            - 1% increase in sales → {dtl:.2f}% increase in EPS
            """))
        
        # Leverage visualization
        sales_range = np.linspace(5000, 15000, 100)
        contributions = sales_range * (price - vc_per_unit)
        ebits = contributions - fixed_costs
        eps_base = (ebits - interest) * (1 - 0.3)  # Assuming 30% tax
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sales_range, y=ebits, name='EBIT', line=dict(color='#e74c3c', width=3)))
        fig.add_trace(go.Scatter(x=sales_range, y=eps_base/1000, name='EPS (thousands)', line=dict(color='#3498db', width=3)))
        fig.add_vline(x=units, line_dash="dash", line_color="green", annotation_text="Current")
        fig.update_layout(title="Leverage Impact on Earnings", xaxis_title="Units Sold", 
                         yaxis_title="Earnings ($)", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Break-even analysis
        st.subheader("Break-Even Analysis")
        accounting_be = fixed_costs / (price - vc_per_unit)
        financial_be = (fixed_costs + interest) / (price - vc_per_unit)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accounting Break-Even", f"{accounting_be:.0f} units", "EBIT = 0")
        with col2:
            st.metric("Financial Break-Even", f"{financial_be:.0f} units", "EPS = 0")
        
        # Safety margin
        safety_margin = (units - accounting_be) / units * 100
        st.progress(safety_margin/100)
        st.caption(f"Margin of Safety: {safety_margin:.1f}%")
    
    with tabs[2]:
        st.subheader("Modigliani-Miller Theorems")
        
        mm_scenario = st.radio("Select Scenario:", 
                              ["MM Proposition I (No Taxes)", 
                               "MM Proposition II (No Taxes)",
                               "MM with Taxes",
                               "Trade-off Theory"])
        
        if mm_scenario == "MM Proposition I (No Taxes)":
            st.markdown(textwrap.dedent("""
            <div class="formula-box">
                V<sub>L</sub> = V<sub>U</sub>
                <br><br>
                Firm value is INDEPENDENT of capital structure in perfect markets.
            </div>
            """), unsafe_allow_html=True)
            
            st.info(textwrap.dedent("""
            **Key Insight:** Investors can create "homemade leverage" to replicate any corporate capital structure.
            Therefore, the firm's financing mix doesn't matter for value creation.
            """))
            
            # Arbitrage demonstration
            st.subheader("Arbitrage Proof")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Levered Firm (L)** overpriced")
                st.write("Strategy:")
                st.write("1. Sell 10% of L's shares")
                st.write("2. Borrow personally (homemade leverage)")
                st.write("3. Buy 10% of Unlevered Firm U")
                st.write("4. Riskless profit until prices equalize")
            
            with col2:
                firm_value = 1000
                debt = 400
                equity_l = 700  # Overpriced
                equity_u = 600
                
                st.metric("Arbitrage Profit", "$100", "Risk-free")
                st.write(f"Firm Value: ${firm_value}M")
                st.write(f"Levered Equity: ${equity_l}M (Overpriced)")
                st.write(f"Unlevered Equity: ${equity_u}M")
        
        elif mm_scenario == "MM Proposition II (No Taxes)":
            st.markdown(textwrap.dedent("""
            <div class="formula-box">
                r<sub>e</sub> = r<sub>0</sub> + (r<sub>0</sub> - r<sub>d</sub>) × (D/E)
                <br><br>
                Cost of equity increases linearly with leverage to exactly offset debt's benefit.
            </div>
            """), unsafe_allow_html=True)
            
            # Visualization
            d_e_ratio = np.linspace(0, 3, 100)
            r0, rd = 0.12, 0.06
            re = r0 + (r0 - rd) * d_e_ratio
            wacc = [r0] * len(d_e_ratio)  # Constant WACC
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d_e_ratio, y=re*100, name='Cost of Equity (re)', 
                                    line=dict(color='#e74c3c', width=3)))
            fig.add_trace(go.Scatter(x=d_e_ratio, y=[r0*100]*len(d_e_ratio), name='WACC', 
                                    line=dict(color='#2ecc71', width=3, dash='dash')))
            fig.add_trace(go.Scatter(x=d_e_ratio, y=[rd*100]*len(d_e_ratio), name='Cost of Debt (rd)', 
                                    line=dict(color='#3498db', width=3)))
            fig.update_layout(title="MM Proposition II: Cost of Capital", 
                            xaxis_title="Debt/Equity Ratio", yaxis_title="Cost of Capital (%)",
                            height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        elif mm_scenario == "MM with Taxes":
            st.markdown(textwrap.dedent("""
            <div class="formula-box">
                V<sub>L</sub> = V<sub>U</sub> + T<sub>c</sub> × D
                <br><br>
                Debt adds value through tax shield. Optimal structure = 100% debt (theoretically).
            </div>
            """), unsafe_allow_html=True)
            
            tax_rate = st.slider("Corporate Tax Rate (%)", 0, 40, 25) / 100
            debt_range = np.linspace(0, 1000, 100)
            vu = 1000
            vl = vu + tax_rate * debt_range
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=debt_range, y=[vu]*len(debt_range), name='Unlevered Value', 
                                    line=dict(color='#95a5a6', width=3)))
            fig.add_trace(go.Scatter(x=debt_range, y=vl, name='Levered Value', 
                                    line=dict(color='#e74c3c', width=3)))
            fig.add_trace(go.Scatter(x=debt_range, y=tax_rate*debt_range, name='Tax Shield', 
                                    line=dict(color='#2ecc71', width=3, dash='dash')))
            fig.update_layout(title="Value of Levered Firm with Taxes", 
                            xaxis_title="Debt ($)", yaxis_title="Firm Value ($)",
                            height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"Tax Shield Value: ${tax_rate * 1000:.0f} at $1000 debt")
        
        else:  # Trade-off Theory
            st.markdown(textwrap.dedent("""
            <div class="formula-box">
                V<sub>L</sub> = V<sub>U</sub> + PV(Tax Shield) - PV(Financial Distress Costs) - PV(Agency Costs)
            </div>
            """), unsafe_allow_html=True)
            
            # Interactive trade-off
            debt_range = np.linspace(0, 1500, 200)
            vu = 1000
            tax_rate = 0.25
            
            tax_shield = tax_rate * debt_range
            distress_cost = 0.00001 * (debt_range ** 2)  # Quadratic increase
            agency_cost = 0.000005 * (debt_range ** 1.5)
            
            vl = vu + tax_shield - distress_cost - agency_cost
            
            optimal_idx = np.argmax(vl)
            optimal_debt = debt_range[optimal_idx]
            max_value = vl[optimal_idx]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=debt_range, y=vl, name='Firm Value', 
                                    line=dict(color='#e74c3c', width=4)))
            fig.add_trace(go.Scatter(x=debt_range, y=vu + tax_shield, name='VU + Tax Shield', 
                                    line=dict(color='#2ecc71', width=2, dash='dash')))
            fig.add_trace(go.Scatter(x=debt_range, y=distress_cost + agency_cost, name='Distress + Agency Costs', 
                                    line=dict(color='#e67e22', width=2, dash='dash')))
            fig.add_vline(x=optimal_debt, line_dash="dash", line_color="purple", 
                         annotation_text=f"Optimal Debt: ${optimal_debt:.0f}")
            fig.update_layout(title="Trade-off Theory: Optimal Capital Structure", 
                            xaxis_title="Debt ($)", yaxis_title="Firm Value ($)",
                            height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Optimal Debt Level", f"${optimal_debt:.0f}", f"Max Value: ${max_value:.0f}")
    
    with tabs[3]:
        st.subheader("Capital Structure Optimizer")
        
        # WACC calculator
        st.write("Calculate Weighted Average Cost of Capital:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            debt = st.number_input("Debt Amount ($M)", value=40.0, step=10.0)
            cost_debt = st.slider("Cost of Debt (%)", 2.0, 15.0, 6.0) / 100
        with col2:
            equity = st.number_input("Equity Amount ($M)", value=60.0, step=10.0)
            cost_equity = st.slider("Cost of Equity (%)", 5.0, 25.0, 12.0) / 100
        with col3:
            tax_rate = st.slider("Tax Rate (%)", 0, 40, 25) / 100
        
        total_value = debt + equity
        wd = debt / total_value
        we = equity / total_value
        
        wacc = wd * cost_debt * (1 - tax_rate) + we * cost_equity
        
        st.metric("WACC", f"{wacc*100:.2f}%", f"D/E Ratio: {debt/equity:.2f}")
        
        # Optimal capital structure finder
        st.subheader("Find Optimal Structure")
        
        vu_input = st.number_input("Unlevered Firm Value ($M)", value=100.0)
        tax_rate_opt = st.slider("Tax Rate (%)", 0, 40, 25, key="tax_opt") / 100
        
        # Simplified model
        debt_opts = np.linspace(0, vu_input * 2, 100)
        tax_shields = debt_opts * tax_rate_opt
        distress = 0.5 * (debt_opts / vu_input) ** 2 * vu_input * 0.1  # Simplified distress cost
        
        values = vu_input + tax_shields - distress
        
        optimal_debt = debt_opts[np.argmax(values)]
        
        fig = px.line(x=debt_opts, y=values, labels={'x': 'Debt ($M)', 'y': 'Firm Value ($M)'},
                     title="Optimal Capital Structure Analysis")
        fig.add_vline(x=optimal_debt, line_dash="dash", line_color="red",
                     annotation_text=f"Optimal: ${optimal_debt:.1f}M")
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:
        st.subheader("Unit 2 Assessment")
        
        questions = [
            {
                "question": "If DOL is 2.5 and DFL is 1.6, what is DTL?",
                "options": ["1.6", "2.5", "4.0", "3.1"],
                "correct": 2,
                "explanation": "DTL = DOL × DFL = 2.5 × 1.6 = 4.0"
            },
            {
                "question": "According to MM Proposition I without taxes:",
                "options": ["Debt increases firm value", "Capital structure affects WACC", 
                           "Firm value is independent of capital structure", "Optimal structure is 100% debt"],
                "correct": 2,
                "explanation": "MM I states that in perfect markets, financing mix doesn't matter."
            },
            {
                "question": "The trade-off theory suggests optimal capital structure occurs when:",
                "options": ["Tax benefits are maximized", "Debt is minimized", 
                           "Marginal tax benefit equals marginal distress cost", "Equity is maximized"],
                "correct": 2,
                "explanation": "Optimal structure balances tax shield benefits against financial distress costs."
            }
        ]
        
        score = 0
        for idx, q in enumerate(questions):
            st.markdown(f"**Q{idx+1}. {q['question']}**")
            answer = st.radio(f"Select answer for Q{idx+1}:", q['options'], key=f"q2_{idx}")
            
            if st.button(f"Check Answer {idx+1}", key=f"check2_{idx}"):
                if q['options'].index(answer) == q['correct']:
                    st.success(f"✅ Correct! {q['explanation']}")
                    score += 1
                else:
                    st.error(f"❌ Incorrect. {q['explanation']}")
        
        if st.button("Submit Quiz", key="submit2"):
            st.session_state.quiz_scores['Unit 2'] = score
            st.balloons()
            st.success(f"Quiz submitted! Score: {score}/{len(questions)}")

# Unit 3: Dividend Policy
elif page == "💵 Unit 3: Dividend Policy":
    st.title("💵 Unit 3: Dividend Policy")
    
    tabs = st.tabs(["📚 Theory", "📊 Policy Types", "🔄 Stock Actions", "🧮 Calculator", "📝 Quiz"])
    
    with tabs[0]:
        st.markdown(textwrap.dedent("""
        <div class="card">
            <h3>Dividend Policy Fundamentals</h3>
            <p>The strategic decision regarding <span class="highlight">earnings distribution</span> vs. 
            <span class="highlight">retention</span> for reinvestment.</p>
        </div>
        """), unsafe_allow_html=True)
        
        # Dividend irrelevance vs. relevance
        theory = st.selectbox("Select Dividend Theory:", 
                             ["Modigliani-Miller Dividend Irrelevance", 
                              "Bird-in-Hand Theory",
                              "Tax Preference Theory",
                              "Signaling Theory",
                              "Clientele Effect"])
        
        theories = {
            "Modigliani-Miller Dividend Irrelevance": {
                "claim": "Dividend policy doesn't affect firm value in perfect markets",
                "assumptions": ["No taxes", "No transaction costs", "Perfect information", "Rational investors"],
                "implication": "Investors indifferent between dividends and capital gains"
            },
            "Bird-in-Hand Theory": {
                "claim": "Investors prefer certain dividends over uncertain capital gains",
                "assumptions": ["Risk aversion", "Dividends less risky than future price appreciation"],
                "implication": "Higher dividends increase stock price and reduce cost of equity"
            },
            "Tax Preference Theory": {
                "claim": "Investors prefer capital gains over dividends due to tax advantages",
                "assumptions": ["Capital gains taxed lower than dividends", "Deferral of capital gains tax"],
                "implication": "Low dividend payout maximizes after-tax returns"
            },
            "Signaling Theory": {
                "claim": "Dividend changes convey management's private information",
                "assumptions": ["Information asymmetry", "Managers know more than investors"],
                "implication": "Dividend increases signal confidence; decreases signal trouble"
            },
            "Clientele Effect": {
                "claim": "Different investor groups prefer different dividend policies",
                "assumptions": ["Heterogeneous investor preferences", "Transaction costs prevent switching"],
                "implication": "Firm should maintain consistent policy to retain its clientele"
            }
        }
        
        t = theories[theory]
        st.markdown(textwrap.dedent(f"""
        <div class="card" style="border-left-color: #e74c3c;">
            <h4>{theory}</h4>
            <p><b>Core Claim:</b> {t['claim']}</p>
            <p><b>Key Assumptions:</b></p>
            <ul>{''.join([f'<li>{a}</li>' for a in t['assumptions']])}</ul>
            <p><b>Practical Implication:</b> {t['implication']}</p>
        </div>
        """), unsafe_allow_html=True)
        
        # Lintner's Model
        st.subheader("Lintner's Partial Adjustment Model")
        
        col1, col2 = st.columns(2)
        with col1:
            target_payout = st.slider("Target Payout Ratio", 0.0, 1.0, 0.5)
            adjustment_speed = st.slider("Adjustment Speed", 0.0, 1.0, 0.3)
            current_eps = st.number_input("Current EPS ($)", value=5.0)
            prev_dps = st.number_input("Previous DPS ($)", value=2.0)
        
        with col2:
            target_dps = target_payout * current_eps
            dps_change = adjustment_speed * (target_dps - prev_dps)
            new_dps = prev_dps + dps_change
            
            st.metric("Target DPS", f"${target_dps:.2f}")
            st.metric("Actual DPS Change", f"${dps_change:.2f}")
            st.metric("New DPS", f"${new_dps:.2f}")
            
            st.info(textwrap.dedent(f"""
            **Interpretation:**
            - Speed of adjustment ({adjustment_speed}): How quickly firm moves to target
            - Lintner found typical speed = 0.30 (30% adjustment per year)
            - Firms avoid dividend cuts (sticky dividends)
            """))
    
    with tabs[1]:
        st.subheader("Dividend Policy Types")
        
        policy = st.selectbox("Select Policy Type:", 
                             ["Residual Policy", "Stable Dividend", "Constant Payout Ratio", "Low Regular Plus Extra"])
        
        # Simulation
        years = list(range(1, 6))
        np.random.seed(42)
        earnings = [100, 120, 80, 140, 110]  # Fluctuating earnings
        investment_needs = [50, 70, 40, 60, 55]  # Varying investment opportunities
        
        if policy == "Residual Policy":
            target_equity = 0.6  # 60% equity financing
            dividends = [max(0, e - (i * target_equity)) for e, i in zip(earnings, investment_needs)]
            description = "Pay dividends only from earnings left after funding all positive-NPV projects"
            
        elif policy == "Stable Dividend":
            base_dividend = 30
            dividends = [base_dividend] * 5
            # Adjust only if sustainable
            for i in range(1, 5):
                if earnings[i] > 100 and dividends[i-1] < earnings[i] * 0.3:
                    dividends[i] = dividends[i-1] * 1.1
            description = "Maintain steady dividend, increase only when sustainable"
            
        elif policy == "Constant Payout Ratio":
            payout = 0.4
            dividends = [e * payout for e in earnings]
            description = "Fixed percentage of earnings (dividends fluctuate with earnings)"
            
        else:  # Low Regular Plus Extra
            base = 20
            extras = [max(0, e - 80) * 0.3 for e in earnings]
            dividends = [base + ex for ex in extras]
            description = "Small base dividend plus year-end extras when earnings strong"
        
        # Visualization
        df = pd.DataFrame({
            'Year': years,
            'Earnings': earnings,
            'Dividends': dividends,
            'Retained': [e - d for e, d in zip(earnings, dividends)]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=years, y=earnings, name='Earnings', marker_color='#3498db'))
        fig.add_trace(go.Bar(x=years, y=dividends, name='Dividends', marker_color='#e74c3c'))
        fig.add_trace(go.Scatter(x=years, y=[e*0.4 for e in earnings], name='40% Payout Line', 
                                line=dict(color='green', dash='dash')))
        fig.update_layout(title=f"{policy}: Earnings vs. Dividends", 
                         barmode='group', height=500,
                         xaxis_title="Year", yaxis_title="Amount ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(textwrap.dedent(f"""
        <div class="card">
            <h4>Policy Characteristics</h4>
            <p>{description}</p>
            <table style="width: 100%; margin-top: 1rem; border-collapse: collapse;">
                <tr style="background: rgba(102, 126, 234, 0.1); border-bottom: 2px solid rgba(102, 126, 234, 0.2);">
                    <th style="padding: 0.5rem;">Year</th><th style="padding: 0.5rem;">Earnings</th><th style="padding: 0.5rem;">Dividends</th><th style="padding: 0.5rem;">Payout Ratio</th>
                </tr>
                {''.join([f'<tr><td>{y}</td><td>${e}</td><td>${d:.0f}</td><td>{d/e*100:.1f}%</td></tr>' 
                         for y, e, d in zip(years, earnings, dividends)])}
            </table>
        </div>
        """), unsafe_allow_html=True)
    
    with tabs[2]:
        st.subheader("Stock Dividends, Splits & Repurchases")
        
        action = st.selectbox("Select Corporate Action:", 
                             ["Stock Dividend", "Stock Split", "Reverse Split", "Stock Repurchase"])
        
        col1, col2 = st.columns(2)
        with col1:
            current_price = st.number_input("Current Stock Price ($)", value=100.0)
            shares_outstanding = st.number_input("Shares Outstanding", value=1000000, step=100000)
            
            if action == "Stock Dividend":
                dividend_pct = st.slider("Stock Dividend %", 0, 50, 10)
                new_shares = shares_outstanding * (1 + dividend_pct/100)
                new_price = current_price / (1 + dividend_pct/100)
                
            elif action == "Stock Split":
                split_ratio = st.selectbox("Split Ratio", ["2:1", "3:1", "3:2", "5:1"])
                ratio = int(split_ratio.split(':')[0]) / int(split_ratio.split(':')[1])
                new_shares = shares_outstanding * ratio
                new_price = current_price / ratio
                
            elif action == "Reverse Split":
                ratio = st.selectbox("Reverse Ratio", ["1:2", "1:5", "1:10"])
                denom = int(ratio.split(':')[1])
                new_shares = shares_outstanding / denom
                new_price = current_price * denom
                
            else:  # Repurchase
                repurchase_pct = st.slider("Repurchase %", 0, 30, 10)
                shares_bought = shares_outstanding * repurchase_pct / 100
                new_shares = shares_outstanding - shares_bought
                new_price = current_price * 1.05  # Assume 5% price increase from signal
        
        with col2:
            st.metric("New Shares Outstanding", f"{new_shares:,.0f}")
            st.metric("New Stock Price", f"${new_price:.2f}")
            st.metric("Market Cap", f"${new_shares * new_price:,.0f}", "Unchanged")
            
            if action == "Stock Repurchase":
                st.success(f"Shares repurchased: {shares_bought:,.0f}")
                st.info("Repurchases signal undervaluation and increase EPS")
        
        # Comparison table
        comparison_data = {
            'Feature': ['Par Value', 'Common Stock Account', 'Additional PIC', 'Retained Earnings', 
                       'Total Equity', 'Shares Outstanding', 'Ownership %'],
            'Stock Dividend (Small)': ['Unchanged', 'Increases', 'Increases', 'Decreases', 
                                      'Unchanged', 'Increases', 'Unchanged'],
            'Stock Split': ['Decreases', 'Unchanged', 'Unchanged', 'Unchanged', 
                           'Unchanged', 'Increases', 'Unchanged'],
            'Reverse Split': ['Increases', 'Unchanged', 'Unchanged', 'Unchanged', 
                             'Unchanged', 'Decreases', 'Unchanged'],
            'Stock Repurchase': ['Unchanged', 'Decreases', 'Unchanged', 'Unchanged', 
                                'Decreases', 'Decreases', 'Increases (for remaining)']
        }
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    with tabs[3]:
        st.subheader("Dividend Calculator & Planner")
        
        calc_type = st.radio("Select Tool:", ["Dividend Yield Calculator", "Repurchase vs. Dividend", "DRIP Planner"])
        
        if calc_type == "Dividend Yield Calculator":
            col1, col2 = st.columns(2)
            with col1:
                annual_dps = st.number_input("Annual Dividend per Share ($)", value=2.50)
                stock_price = st.number_input("Current Stock Price ($)", value=50.0)
            
            with col2:
                yield_pct = (annual_dps / stock_price) * 100
                st.metric("Dividend Yield", f"{yield_pct:.2f}%")
                
                # Comparison with market
                market_yield = 1.5  # S&P 500 average
                if yield_pct > market_yield * 1.5:
                    st.success(f"Above market average ({market_yield}%)")
                elif yield_pct < market_yield * 0.5:
                    st.warning("Below market average - growth stock?")
                else:
                    st.info("Market average yield")
        
        elif calc_type == "Repurchase vs. Dividend":
            col1, col2 = st.columns(2)
            with col1:
                cash_available = st.number_input("Cash Available ($M)", value=100.0)
                shares_out = st.number_input("Shares Outstanding (M)", value=50.0)
                current_price = st.number_input("Stock Price ($)", value=40.0)
                tax_rate_div = st.slider("Dividend Tax Rate (%)", 0, 40, 20) / 100
                tax_rate_cg = st.slider("Capital Gains Tax Rate (%)", 0, 30, 15) / 100
            
            with col2:
                # Dividend scenario
                div_per_share = cash_available / shares_out
                after_tax_div = div_per_share * (1 - tax_rate_div)
                
                # Repurchase scenario
                shares_repurchased = cash_available / current_price
                price_increase = (cash_available / (shares_out - shares_repurchased)) - current_price
                after_tax_cg = price_increase * (1 - tax_rate_cg)
                
                st.metric("Dividend per Share", f"${div_per_share:.2f}")
                st.metric("After-tax Dividend", f"${after_tax_div:.2f}")
                st.metric("Repurchase Price Increase", f"${price_increase:.2f}")
                st.metric("After-tax Capital Gain", f"${after_tax_cg:.2f}")
                
                if after_tax_cg > after_tax_div:
                    st.success("Repurchase more tax-efficient")
                else:
                    st.info("Dividend preferred or neutral")
        
        else:  # DRIP Planner
            initial_investment = st.number_input("Initial Investment ($)", value=10000)
            monthly_contribution = st.number_input("Monthly Contribution ($)", value=500)
            dividend_yield = st.slider("Dividend Yield (%)", 0.0, 10.0, 3.0) / 100
            dividend_growth = st.slider("Dividend Growth Rate (%)", 0.0, 15.0, 5.0) / 100
            price_appreciation = st.slider("Stock Price Appreciation (%)", 0.0, 15.0, 6.0) / 100
            years = st.slider("Investment Horizon (Years)", 1, 40, 20)
            
            # DRIP calculation
            values = []
            shares = initial_investment / 100  # Assume $100 initial price
            price = 100
            
            for year in range(years + 1):
                if year > 0:
                    # Add monthly contributions
                    for month in range(12):
                        shares += monthly_contribution / price
                        price *= (1 + price_appreciation) ** (1/12)
                    
                    # Reinvest dividends
                    annual_dividend = shares * price * dividend_yield * ((1 + dividend_growth) ** year)
                    shares += annual_dividend / price
                
                values.append(shares * price)
            
            final_value = values[-1]
            total_contributed = initial_investment + (monthly_contribution * 12 * years)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Portfolio Value", f"${final_value:,.0f}")
            with col2:
                st.metric("Total Contributed", f"${total_contributed:,.0f}")
            with col3:
                st.metric("Dividend Reinvested", f"${final_value - total_contributed:,.0f}")
            
            # Growth chart
            fig = px.line(x=list(range(years + 1)), y=values, 
                         labels={'x': 'Years', 'y': 'Portfolio Value ($)'},
                         title="DRIP Growth Over Time")
            fig.add_hline(y=total_contributed, line_dash="dash", line_color="red", 
                         annotation_text="Total Contributions")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:
        st.subheader("Unit 3 Assessment")
        
        questions = [
            {
                "question": "According to MM dividend irrelevance, dividend policy matters when:",
                "options": ["Investors are risk-averse", "Taxes exist", "Markets are perfect", "Signaling is important"],
                "correct": 2,
                "explanation": "MM irrelevance holds only in perfect markets without taxes, transaction costs, or asymmetric information."
            },
            {
                "question": "Which policy provides the most stable cash flows to investors?",
                "options": ["Residual policy", "Constant payout ratio", "Stable dividend policy", "No dividend"],
                "correct": 2,
                "explanation": "Stable dividend policy maintains consistent payments, smoothing investor cash flows."
            },
            {
                "question": "A 2-for-1 stock split will:",
                "options": ["Double the firm's equity", "Halve the stock price", "Increase par value", "Reduce retained earnings"],
                "correct": 1,
                "explanation": "Splits increase shares outstanding and proportionally decrease price; total equity unchanged."
            }
        ]
        
        score = 0
        for idx, q in enumerate(questions):
            st.markdown(f"**Q{idx+1}. {q['question']}**")
            answer = st.radio(f"Select answer for Q{idx+1}:", q['options'], key=f"q3_{idx}")
            
            if st.button(f"Check Answer {idx+1}", key=f"check3_{idx}"):
                if q['options'].index(answer) == q['correct']:
                    st.success(f"✅ Correct! {q['explanation']}")
                    score += 1
                else:
                    st.error(f"❌ Incorrect. {q['explanation']}")
        
        if st.button("Submit Quiz", key="submit3"):
            st.session_state.quiz_scores['Unit 3'] = score
            st.balloons()
            st.success(f"Quiz submitted! Score: {score}/{len(questions)}")

# Unit 4: Raising Capital
elif page == "🏦 Unit 4: Raising Capital":
    st.title("🏦 Unit 4: Raising Capital")
    
    tabs = st.tabs(["📚 Financing Methods", "📊 Cost Comparison", "🎯 IPO Simulator", "🧮 Calculator", "📝 Quiz"])
    
    with tabs[0]:
        st.markdown(textwrap.dedent("""
        <div class="card">
            <h3>Long-Term Financing Sources</h3>
            <p>Exploring the spectrum from <span class="highlight">debt</span> to <span class="highlight">equity</span> 
            and hybrid instruments.</p>
        </div>
        """), unsafe_allow_html=True)
        
        instrument = st.selectbox("Select Financing Instrument:", 
                                 ["Term Loan", "Corporate Bonds", "Preferred Stock", 
                                  "Common Stock", "Convertible Bonds", "Warrants", "Leasing"])
        
        instrument_details = {
            "Term Loan": {
                "type": "Debt",
                "maturity": "1-10 years",
                "security": "Collateral required",
                "cost": "Interest + fees",
                "advantages": ["Flexibility", "Speed", "Relationship banking"],
                "disadvantages": ["Covenants restrictive", "Floating rate risk", "Limited amount"]
            },
            "Corporate Bonds": {
                "type": "Debt",
                "maturity": "10-30 years",
                "security": "Secured or unsecured",
                "cost": "Coupon + issuance costs",
                "advantages": ["Large amounts", "Fixed rate available", "Public market access"],
                "disadvantages": ["Disclosure requirements", "Credit rating dependency", "Call risk"]
            },
            "Preferred Stock": {
                "type": "Hybrid",
                "maturity": "Perpetual",
                "security": "None (equity)",
                "cost": "Dividend (not tax-deductible)",
                "advantages": ["No maturity", "No voting dilution", "Strengthens equity base"],
                "disadvantages": ["Higher cost than debt", "Dividend can be deferred", "Complexity"]
            },
            "Common Stock": {
                "type": "Equity",
                "maturity": "Perpetual",
                "security": "Residual claim",
                "cost": "Cost of equity (highest)",
                "advantages": ["No mandatory payments", "Permanent capital", "Flexibility"],
                "disadvantages": ["Dilution", "Highest cost", "Signaling negative"]
            },
            "Convertible Bonds": {
                "type": "Hybrid",
                "maturity": "10-20 years",
                "security": "Unsecured debt",
                "cost": "Lower coupon + conversion option",
                "advantages": ["Lower interest cost", "Delayed equity", "Automatic conversion"],
                "disadvantages": ["Dilution if converted", "Complex valuation", "Forced conversion risk"]
            },
            "Warrants": {
                "type": "Derivative",
                "maturity": "3-10 years",
                "security": "Attached to bonds",
                "cost": "Opportunity cost",
                "advantages": ["Sweetener for debt", "Cash inflow on exercise", "No immediate dilution"],
                "disadvantages": ["Dilution when exercised", "Complex pricing", "May expire worthless"]
            },
            "Leasing": {
                "type": "Alternative",
                "maturity": "Asset life",
                "security": "Leased asset",
                "cost": "Lease payments",
                "advantages": ["100% financing", "Off-balance sheet (historically)", "Tax benefits"],
                "disadvantages": ["Higher cost than buying", "No ownership", "Inflexible"]
            }
        }
        
        details = instrument_details[instrument]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(textwrap.dedent(f"""
            <div class="card">
                <h3>{instrument}</h3>
                <p><span class="info-badge">{details['type']}</span></p>
                <table style="width: 100%; margin: 1rem 0;">
                    <tr><td><b>Maturity</b></td><td>{details['maturity']}</td></tr>
                    <tr><td><b>Security</b></td><td>{details['security']}</td></tr>
                    <tr><td><b>Cost Structure</b></td><td>{details['cost']}</td></tr>
                </table>
            </div>
            """), unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Advantages**")
            for adv in details['advantages']:
                st.markdown(f"✅ {adv}")
            
            st.markdown("**Disadvantages**")
            for dis in details['disadvantages']:
                st.markdown(f"❌ {dis}")
        
        # Capital structure pie chart
        st.subheader("Typical Usage in Capital Structure")
        if instrument in ["Term Loan", "Corporate Bonds"]:
            weights = [30, 0, 0, 70]  # Debt heavy
        elif instrument in ["Preferred Stock", "Convertible Bonds", "Warrants"]:
            weights = [20, 10, 0, 70]  # Hybrid
        else:
            weights = [0, 0, 30, 70]  # Equity heavy
        
        fig = go.Figure(go.Pie(
            labels=['Debt', 'Preferred', 'Common Equity', 'Other'],
            values=weights,
            hole=0.4,
            marker_colors=['#e74c3c', '#f39c12', '#3498db', '#95a5']))
        fig.update_layout(title=f"Typical Structure with {instrument}", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Cost of Capital Comparison")
        
        # Interactive cost comparison
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_free = st.slider("Risk-free Rate (%)", 1.0, 10.0, 3.0) / 100
            market_return = st.slider("Market Return (%)", 5.0, 20.0, 10.0) / 100
        with col2:
            beta = st.slider("Company Beta", 0.5, 2.5, 1.2)
            tax_rate = st.slider("Tax Rate (%)", 0, 40, 25) / 100
        with col3:
            credit_spread = st.slider("Credit Spread (%)", 0.5, 10.0, 3.0) / 100
            debt_ratio = st.slider("Debt Ratio (%)", 0, 100, 40) / 100
        
        # Calculate costs
        cost_equity = risk_free + beta * (market_return - risk_free)
        cost_debt = risk_free + credit_spread
        after_tax_debt = cost_debt * (1 - tax_rate)
        
        wacc = debt_ratio * after_tax_debt + (1 - debt_ratio) * cost_equity
        
        # Visualization
        categories = ['Cost of Debt\n(Before Tax)', 'Cost of Debt\n(After Tax)', 'Cost of Equity', 'WACC']
        values = [cost_debt*100, after_tax_debt*100, cost_equity*100, wacc*100]
        colors = ['#e74c3c', '#c0392b', '#3498db', '#2ecc71']
        
        fig = go.Figure(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f'{v:.2f}%' for v in values],
            textposition='auto'
        ))
        fig.update_layout(title="Cost of Capital Components", yaxis_title="Cost (%)", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Pecking Order Theory
        st.subheader("Pecking Order Theory")
        st.info(textwrap.dedent("""
        **Preference Ranking (Due to Asymmetric Information):**
        1. Internal financing (retained earnings)
        2. Debt financing
        3. Equity financing (last resort)
        
        *Rationale:* Equity issuance signals overvaluation; debt signals confidence.
        """))
        
        # Financing hierarchy visualization
        sources = ['Retained Earnings', 'Bank Debt', 'Corporate Bonds', 'Convertibles', 'Common Stock']
        info_asymmetry = [1, 2, 3, 4, 5]  # Lower is better
        cost_signal = [1, 2, 3, 4, 5]  # Lower is cheaper/better signal
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=info_asymmetry, y=cost_signal,
            mode='markers+text',
            text=sources,
            textposition="top center",
            marker=dict(size=30, color=['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c'])
        ))
        fig.update_layout(
            title="Financing Hierarchy: Information Asymmetry vs. Cost",
            xaxis_title="Information Asymmetry Impact",
            yaxis_title="Relative Cost/Signal Quality",
            showlegend=False,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("🚀 IPO Simulator")
        
        st.markdown(textwrap.dedent("""
        <div class="warning-box">
            Experience the IPO process from filing to first-day trading!
        </div>
        """), unsafe_allow_html=True)
        
        # IPO Setup
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input("Company Name", "TechNova Inc.")
            industry = st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Consumer", "Energy"])
            revenue = st.number_input("Annual Revenue ($M)", value=100, step=10)
            growth_rate = st.slider("Revenue Growth (%)", -20, 100, 30)
        
        with col2:
            shares_offered = st.number_input("Shares Offered (M)", value=10.0, step=1.0)
            price_range_low = st.number_input("Price Range Low ($)", value=15.0)
            price_range_high = st.number_input("Price Range High ($)", value=18.0)
            market_condition = st.select_slider("Market Condition", 
                                               options=["Bear", "Weak", "Neutral", "Strong", "Bull"])
        
        # Market condition multiplier
        condition_mult = {"Bear": 0.8, "Weak": 0.9, "Neutral": 1.0, "Strong": 1.1, "Bull": 1.2}
        
        # Book building simulation
        st.subheader("Book Building Process")
        
        # Generate institutional interest
        np.random.seed(42)
        institutions = ['Fidelity', 'BlackRock', 'Vanguard', 'T. Rowe Price', 'Capital Research',
                       'Wellington', 'JPMorgan AM', 'Goldman Sachs AM']
        
        interest_levels = np.random.uniform(0.5, 1.5, len(institutions))
        market_mult = condition_mult[market_condition]
        
        bids = []
        for i, inst in enumerate(institutions):
            base_bid = (price_range_low + price_range_high) / 2
            bid = base_bid * interest_levels[i] * market_mult * np.random.uniform(0.9, 1.1)
            shares = np.random.randint(100000, 1000000)
            bids.append({'Institution': inst, 'Bid Price': bid, 'Shares': shares, 'Total': bid * shares})
        
        bid_df = pd.DataFrame(bids)
        
        # Price discovery
        total_demand = bid_df['Shares'].sum()
        cover_ratio = total_demand / (shares_offered * 1000000)
        
        # Final pricing
        if cover_ratio > 2:
            final_price = price_range_high * 1.05
        elif cover_ratio > 1.5:
            final_price = price_range_high
        elif cover_ratio > 1:
            final_price = (price_range_low + price_range_high) / 2
        else:
            final_price = price_range_low * 0.95
        
        st.metric("Coverage Ratio", f"{cover_ratio:.2f}x", "Institutional demand")
        st.metric("Final Offer Price", f"${final_price:.2f}")
        
        # First day trading simulation
        st.subheader("First Day Trading Simulation")
        
        # Underpricing calculation
        industry_underpricing = {"Technology": 0.25, "Healthcare": 0.15, "Finance": 0.10, 
                                "Consumer": 0.12, "Energy": 0.08}
        base_underpricing = industry_underpricing[industry]
        market_adj = (market_mult - 1) * 0.5
        final_underpricing = base_underpricing + market_adj + np.random.normal(0, 0.05)
        
        first_day_open = final_price * (1 + final_underpricing * 0.3)
        first_day_close = final_price * (1 + final_underpricing)
        
        # Generate intraday prices
        hours = ['9:30', '10:00', '10:30', '11:00', '11:30', '12:00', 
                '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00']
        prices = [first_day_open]
        for i in range(1, len(hours)):
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=prices, mode='lines+markers', 
                                name='Stock Price', line=dict(color='#2ecc71', width=3)))
        fig.add_hline(y=final_price, line_dash="dash", line_color="red", 
                     annotation_text=f"IPO Price: ${final_price:.2f}")
        fig.update_layout(title=f"{company_name} - First Day Trading", 
                         yaxis_title="Price ($)", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("First Day Open", f"${first_day_open:.2f}")
        with col2:
            st.metric("First Day Close", f"${first_day_close:.2f}")
        with col3:
            pop = (first_day_close - final_price) / final_price * 100
            st.metric("IPO Pop", f"{pop:.1f}%", "Underpricing")
        
        # Money left on table
        money_left = (first_day_close - final_price) * shares_offered * 1000000
        st.warning(f"💸 Money Left on Table: ${money_left:,.0f}")
        
        # Winner's curse demonstration
        st.subheader("Winner's Curse Analysis")
        st.write("""
        In IPOs with high underpricing, uninformed investors receive more allocation 
        (informed investors know it's overpriced and withdraw). This creates adverse selection.
        """)
        
        informed_bid = final_price * 0.95  # Informed know true value
        uninformed_bid = price_range_high * 1.1  # Uninformed overbid
        
        if final_price > informed_bid:
            st.error("Informed investors withdrew - adverse selection for underwriters")
        else:
            st.success("Both types participate")
    
    with tabs[3]:
        st.subheader("Financing Calculators")
        
        calc_type = st.radio("Select Calculator:", 
                            ["Lease vs. Buy", "Convertible Bond Analysis", "Warrant Valuation"])
        
        if calc_type == "Lease vs. Buy":
            col1, col2 = st.columns(2)
            with col1:
                asset_cost = st.number_input("Asset Cost ($)", value=1000000, step=100000)
                lease_payment = st.number_input("Annual Lease Payment ($)", value=250000, step=10000)
                lease_term = st.slider("Lease Term (Years)", 1, 10, 5)
                tax_rate = st.slider("Tax Rate (%)", 0, 40, 25, key="lease_tax") / 100
                discount_rate = st.slider("After-tax Discount Rate (%)", 3.0, 15.0, 6.0) / 100
            
            with col2:
                # Buy analysis
                depreciation = asset_cost / lease_term
                dep_shield = depreciation * tax_rate
                after_tax_lease = lease_payment * (1 - tax_rate)
                
                # NPV of leasing vs. buying
                lease_costs = [after_tax_lease] * lease_term
                buy_costs = [-dep_shield] * lease_term  # Negative because it's a benefit
                buy_costs[0] -= asset_cost  # Initial purchase
                
                npv_lease = npf.npv(discount_rate, [0] + lease_costs)
                npv_buy = npf.npv(discount_rate, [0] + buy_costs)
                
                advantage = abs(npv_lease) - abs(npv_buy)
                
                st.metric("NPV of Leasing", f"${npv_lease:,.0f}")
                st.metric("NPV of Buying", f"${npv_buy:,.0f}")
                
                if npv_lease < npv_buy:
                    st.success(f"Leasing is cheaper by ${advantage:,.0f}")
                else:
                    st.info(f"Buying is cheaper by ${advantage:,.0f}")
            
            # Visualization
            years = list(range(lease_term + 1))
            lease_cashflows = [0] + [after_tax_lease] * lease_term
            buy_cashflows = [-asset_cost] + [-dep_shield] * lease_term
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Lease', x=years, y=lease_cashflows, marker_color='#e74c3c'))
            fig.add_trace(go.Bar(name='Buy', x=years, y=buy_cashflows, marker_color='#3498db'))
            fig.update_layout(title="Cash Flow Comparison", barmode='group', height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        elif calc_type == "Convertible Bond Analysis":
            col1, col2 = st.columns(2)
            with col1:
                face_value = st.number_input("Face Value ($)", value=1000)
                coupon_rate = st.slider("Coupon Rate (%)", 0.0, 10.0, 4.0) / 100
                conversion_price = st.number_input("Conversion Price ($)", value=50.0)
                stock_price = st.number_input("Current Stock Price ($)", value=45.0)
                volatility = st.slider("Stock Volatility (%)", 10.0, 100.0, 30.0) / 100
                time = st.slider("Time to Maturity (Years)", 1, 20, 10)
            
            with col2:
                # Conversion metrics
                conversion_ratio = face_value / conversion_price
                conversion_value = conversion_ratio * stock_price
                conversion_premium = (conversion_price - stock_price) / stock_price * 100
                
                # Straight bond value (simplified)
                straight_value = face_value * 0.9  # Approximate
                
                st.metric("Conversion Ratio", f"{conversion_ratio:.2f} shares")
                st.metric("Conversion Value", f"${conversion_value:.2f}")
                st.metric("Conversion Premium", f"{conversion_premium:.1f}%")
                st.metric("Floor Value", f"${max(straight_value, conversion_value):.2f}")
                
                # Option value approximation
                option_value = max(0, stock_price - conversion_price) * conversion_ratio
                total_value = straight_value + option_value
                
                st.metric("Estimated Bond Value", f"${total_value:.2f}")
        
        else:  # Warrant Valuation
            col1, col2 = st.columns(2)
            with col1:
                s = st.number_input("Stock Price ($)", value=50.0, key="warrant_s")
                k = st.number_input("Exercise Price ($)", value=45.0, key="warrant_k")
                t = st.slider("Time to Expiration (Years)", 0.5, 10.0, 3.0, key="warrant_t")
                r = st.slider("Risk-free Rate (%)", 0.0, 10.0, 3.0, key="warrant_r") / 100
                sigma = st.slider("Volatility (%)", 10.0, 100.0, 35.0, key="warrant_sigma") / 100
                shares_out = st.number_input("Shares Outstanding", value=10000000, step=1000000)
                warrants = st.number_input("Warrants Issued", value=1000000, step=100000)
            
            with col2:
                # Black-Scholes for warrant (simplified, ignoring dilution)
                price, d1, d2 = black_scholes(s, k, t, r, sigma, 'call')
                
                # Dilution adjustment
                dilution_factor = shares_out / (shares_out + warrants)
                warrant_value = price * dilution_factor
                
                st.metric("Warrant Value", f"${warrant_value:.2f}")
                st.metric("Intrinsic Value", f"${max(0, s-k):.2f}")
                st.metric("Time Premium", f"${warrant_value - max(0, s-k):.2f}")
                
                # Break-even
                breakeven = k + warrant_value
                st.metric("Break-even Stock Price", f"${breakeven:.2f}")
    
    with tabs[4]:
        st.subheader("Unit 4 Assessment")
        
        questions = [
            {
                "question": "Which financing source has the lowest priority in bankruptcy?",
                "options": ["Bank debt", "Corporate bonds", "Preferred stock", "Common stock"],
                "correct": 3,
                "explanation": "Common stockholders are residual claimants, last to be paid in liquidation."
            },
            {
                "question": "The main advantage of convertible bonds is:",
                "options": ["Lower risk than straight debt", "Lower interest cost than straight debt", 
                           "No dilution risk", "Higher priority than senior debt"],
                "correct": 1,
                "explanation": "Conversion option allows lower coupon; investors accept less interest for potential equity upside."
            },
            {
                "question": "According to pecking order theory, firms prefer:",
                "options": ["Equity over debt", "Debt over internal funds", 
                           "Internal funds over external financing", "Short-term over long-term debt"],
                "correct": 2,
                "explanation": "Due to asymmetric information, firms prefer internal funds first to avoid adverse selection costs."
            }
        ]
        
        score = 0
        for idx, q in enumerate(questions):
            st.markdown(f"**Q{idx+1}. {q['question']}**")
            answer = st.radio(f"Select answer for Q{idx+1}:", q['options'], key=f"q4_{idx}")
            
            if st.button(f"Check Answer {idx+1}", key=f"check4_{idx}"):
                if q['options'].index(answer) == q['correct']:
                    st.success(f"✅ Correct! {q['explanation']}")
                    score += 1
                else:
                    st.error(f"❌ Incorrect. {q['explanation']}")
        
        if st.button("Submit Quiz", key="submit4"):
            st.session_state.quiz_scores['Unit 4'] = score
            st.balloons()
            st.success(f"Quiz submitted! Score: {score}/{len(questions)}")

# Unit 5: Short-Term Planning
elif page == "📈 Unit 5: Short-Term Planning":
    st.title("📈 Unit 5: Short-Term Financial Planning")
    
    tabs = st.tabs(["📚 Working Capital", "⏱️ Cash Cycles", "📊 Cash Budgeting", "🧮 Calculator", "📝 Quiz"])
    
    with tabs[0]:
        st.markdown(textwrap.dedent("""
        <div class="card">
            <h3>Working Capital Management</h3>
            <p>Managing <span class="highlight">current assets</span> and <span class="highlight">current liabilities</span> 
            to balance profitability and liquidity.</p>
        </div>
        """), unsafe_allow_html=True)
        
        # Working capital policy
        policy = st.select_slider("Select Policy Aggressiveness:", 
                                 options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"])
        
        policy_settings = {
            "Very Conservative": {"ca_ratio": 40, "cl_ratio": 10, "nwc": 30},
            "Conservative": {"ca_ratio": 35, "cl_ratio": 15, "nwc": 20},
            "Moderate": {"ca_ratio": 30, "cl_ratio": 20, "nwc": 10},
            "Aggressive": {"ca_ratio": 25, "cl_ratio": 25, "nwc": 0},
            "Very Aggressive": {"ca_ratio": 20, "cl_ratio": 30, "nwc": -10}
        }
        
        settings = policy_settings[policy]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Assets/Sales", f"{settings['ca_ratio']}%")
        with col2:
            st.metric("Current Liabilities/Assets", f"{settings['cl_ratio']}%")
        with col3:
            st.metric("Net Working Capital", f"{settings['nwc']}%")
        
        # Risk-return tradeoff
        st.subheader("Risk-Return Tradeoff")
        
        policies = list(policy_settings.keys())
        liquidity = [90, 75, 50, 25, 10]  # Higher is safer
        profitability = [10, 25, 50, 75, 90]  # Higher is more profitable
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=policies, y=liquidity,
            mode='lines+markers',
            name='Liquidity (Safety)',
            line=dict(color='#2ecc71', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=policies, y=profitability,
            mode='lines+markers',
            name='Profitability',
            line=dict(color='#e74c3c', width=3)
        ))
        fig.update_layout(title="Policy Trade-offs", yaxis_title="Level (0-100)", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Financing approaches
        st.subheader("Financing Approaches")
        
        approach = st.selectbox("Select Approach:", 
                               ["Maturity Matching", "Aggressive", "Conservative"])
        
        # Visual representation
        months = list(range(1, 13))
        seasonal_need = [30, 25, 35, 45, 50, 60, 55, 45, 40, 35, 30, 25]  # Seasonal pattern
        permanent = [20] * 12  # Permanent current assets
        
        if approach == "Maturity Matching":
            long_term = [p + 20 for p in permanent]  # Permanent + fixed assets
            short_term = [s - 20 for s in seasonal_need]
            title = "Maturity Matching: Short-term finances seasonal, long-term finances permanent"
        elif approach == "Aggressive":
            long_term = [25] * 12  # Minimal long-term
            short_term = [p + s - 25 for p, s in zip(permanent, seasonal_need)]
            title = "Aggressive: Short-term finances part of permanent needs"
        else:  # Conservative
            long_term = [max(seasonal_need) + 10] * 12  # All needs + buffer
            short_term = [0] * 12
            title = "Conservative: Long-term finances all needs; excess invested"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=permanent, name='Permanent CA', 
                                fill='tozeroy', line=dict(color='#3498db')))
        fig.add_trace(go.Scatter(x=months, y=[p+s for p,s in zip(permanent, seasonal_need)], 
                                name='Total CA', fill='tonexty', line=dict(color='#2ecc71')))
        fig.add_trace(go.Scatter(x=months, y=long_term, name='Long-term Financing', 
                                line=dict(color='#e74c3c', dash='dash', width=3)))
        fig.add_trace(go.Scatter(x=months, y=short_term, name='Short-term Financing', 
                                line=dict(color='#f39c12', dash='dot', width=3)))
        fig.update_layout(title=title, xaxis_title="Month", yaxis_title="Amount ($)", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Cash Conversion Cycle")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            inventory_days = st.slider("Inventory Days", 0, 180, 60)
            receivable_days = st.slider("Receivable Days (DSO)", 0, 180, 45)
        with col2:
            payable_days = st.slider("Payable Days (DPO)", 0, 180, 30)
            cogs = st.number_input("Annual COGS ($M)", value=50, step=5)
        with col3:
            sales = st.number_input("Annual Sales ($M)", value=80, step=5)
        
        # Calculations
        operating_cycle = inventory_days + receivable_days
        cash_conversion_cycle = operating_cycle - payable_days
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Operating Cycle", f"{operating_cycle} days")
        with col2:
            st.metric("Cash Conversion Cycle", f"{cash_conversion_cycle} days")
        with col3:
            working_capital_need = (cash_conversion_cycle / 365) * cogs
            st.metric("Working Capital Need", f"${working_capital_need:.2f}M")
        
        # Cycle visualization
        fig = go.Figure()
        
        # Timeline
        days = list(range(max(operating_cycle, payable_days) + 10))
        
        # Inventory period
        fig.add_trace(go.Scatter(
            x=[0, inventory_days], y=[1, 1],
            mode='lines',
            line=dict(color='#3498db', width=10),
            name=f'Inventory Period ({inventory_days} days)'
        ))
        
        # Receivable period
        fig.add_trace(go.Scatter(
            x=[inventory_days, operating_cycle], y=[1, 1],
            mode='lines',
            line=dict(color='#2ecc71', width=10),
            name=f'Receivable Period ({receivable_days} days)'
        ))
        
        # Payable period (negative cash flow)
        fig.add_trace(go.Scatter(
            x=[0, payable_days], y=[0.5, 0.5],
            mode='lines',
            line=dict(color='#e74c3c', width=10),
            name=f'Payable Period ({payable_days} days)'
        ))
        
        # CCC indicator
        fig.add_trace(go.Scatter(
            x=[payable_days, operating_cycle], y=[0.75, 0.75],
            mode='lines',
            line=dict(color='#f39c12', width=5, dash='dash'),
            name=f'Cash Cycle ({cash_conversion_cycle} days)'
        ))
        
        fig.update_layout(
            title="Cash Conversion Cycle Timeline",
            yaxis_visible=False,
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Improvement scenarios
        st.subheader("Improvement Scenarios")
        
        scenario = st.selectbox("Select Improvement:", 
                               ["Reduce Inventory 10 days", "Reduce Receivables 5 days", 
                                "Extend Payables 5 days", "All Improvements"])
        
        improvements = {
            "Reduce Inventory 10 days": (-10, 0, 0),
            "Reduce Receivables 5 days": (0, -5, 0),
            "Extend Payables 5 days": (0, 0, 5),
            "All Improvements": (-10, -5, 5)
        }
        
        inv_chg, rec_chg, pay_chg = improvements[scenario]
        new_ccc = cash_conversion_cycle + inv_chg + rec_chg + pay_chg
        savings = (cash_conversion_cycle - new_ccc) / 365 * cogs
        
        st.metric("New CCC", f"{new_ccc} days", f"{new_ccc - cash_conversion_cycle} days")
        st.metric("Working Capital Freed", f"${savings:.2f}M")
    
    with tabs[2]:
        st.subheader("Cash Budget Builder")
        
        # Monthly cash budget
        st.write("Enter monthly estimates (in $000s):")
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Cash Collections")
            collections = [st.number_input(f"{m} Collections", value=100 + i*10, key=f"coll_{i}") 
                          for i, m in enumerate(months)]
        
        with col2:
            st.write("Cash Disbursements")
            disbursements = [st.number_input(f"{m} Disbursements", value=90 + i*5, key=f"disb_{i}") 
                           for i, m in enumerate(months)]
        
        beginning_cash = st.number_input("Beginning Cash Balance ($000s)", value=50)
        minimum_balance = st.number_input("Minimum Required Balance ($000s)", value=25)
        
        # Calculate cash budget
        cash_budget = []
        balance = beginning_cash
        
        for i, (coll, disb) in enumerate(zip(collections, disbursements)):
            net_flow = coll - disb
            ending_balance = balance + net_flow
            surplus_deficit = ending_balance - minimum_balance
            
            cash_budget.append({
                'Month': months[i],
                'Beginning': balance,
                'Collections': coll,
                'Disbursements': disb,
                'Net Flow': net_flow,
                'Ending': ending_balance,
                'Surplus/(Deficit)': surplus_deficit
            })
            
            balance = ending_balance
        
        budget_df = pd.DataFrame(cash_budget)
        st.dataframe(budget_df.style.applymap(
            lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else 'color: green',
            subset=['Surplus/(Deficit)']
        ), use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Collections', x=months, y=collections, marker_color='#2ecc71'))
        fig.add_trace(go.Bar(name='Disbursements', x=months, y=disbursements, marker_color='#e74c3c'))
        fig.add_trace(go.Scatter(name='Ending Balance', x=months, 
                                y=budget_df['Ending'], mode='lines+markers',
                                line=dict(color='#3498db', width=3)))
        fig.add_hline(y=minimum_balance, line_dash="dash", line_color="orange", 
                     annotation_text="Minimum")
        fig.update_layout(title="Cash Budget Overview", barmode='group', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Financing needs
        deficits = budget_df[budget_df['Surplus/(Deficit)'] < 0]
        if not deficits.empty:
            st.error(f"⚠️ Financing needed in: {', '.join(deficits['Month'].tolist())}")
            st.write(f"Maximum deficit: ${deficits['Surplus/(Deficit)'].min():.0f}k")
        else:
            st.success("✅ No financing needs; surpluses available for investment")
    
    with tabs[3]:
        st.subheader("Working Capital Calculators")
        
        calc_type = st.radio("Select Tool:", ["EOQ Calculator", "Credit Policy Analyzer", "Float Calculator"])
        
        if calc_type == "EOQ Calculator":
            col1, col2 = st.columns(2)
            with col1:
                annual_demand = st.number_input("Annual Demand (units)", value=10000, step=1000)
                order_cost = st.number_input("Ordering Cost ($)", value=100, step=10)
                carrying_cost_per_unit = st.number_input("Carrying Cost per Unit ($)", value=5, step=1)
            with col2:
                carrying_cost_percent = st.slider("Carrying Cost (% of value)", 0.0, 50.0, 20.0) / 100
                unit_cost = st.number_input("Unit Cost ($)", value=25, step=5)
                
                # Use percentage if specified, otherwise per unit
                if carrying_cost_per_unit == 0:
                    carrying_cost = carrying_cost_percent * unit_cost
                else:
                    carrying_cost = carrying_cost_per_unit
            
            # EOQ calculation
            eoq = np.sqrt((2 * annual_demand * order_cost) / carrying_cost)
            total_cost = (annual_demand / eoq) * order_cost + (eoq / 2) * carrying_cost
            orders_per_year = annual_demand / eoq
            
            st.metric("Economic Order Quantity", f"{eoq:.0f} units")
            st.metric("Total Annual Cost", f"${total_cost:,.2f}")
            st.metric("Orders per Year", f"{orders_per_year:.1f}")
            st.metric("Order Cycle", f"{365/orders_per_year:.0f} days")
            
            # Cost curve
            q_range = np.linspace(eoq * 0.2, eoq * 3, 100)
            ordering = (annual_demand / q_range) * order_cost
            carrying = (q_range / 2) * carrying_cost
            total = ordering + carrying
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=q_range, y=ordering, name='Ordering Cost', line=dict(color='#e74c3c')))
            fig.add_trace(go.Scatter(x=q_range, y=carrying, name='Carrying Cost', line=dict(color='#3498db')))
            fig.add_trace(go.Scatter(x=q_range, y=total, name='Total Cost', line=dict(color='#2ecc71', width=3)))
            fig.add_vline(x=eoq, line_dash="dash", line_color="purple", annotation_text=f"EOQ: {eoq:.0f}")
            fig.update_layout(title="EOQ Cost Curves", xaxis_title="Order Quantity", 
                            yaxis_title="Cost ($)", height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        elif calc_type == "Credit Policy Analyzer":
            col1, col2 = st.columns(2)
            with col1:
                current_sales = st.number_input("Current Annual Sales ($)", value=1000000, step=100000)
                current_dso = st.slider("Current DSO (days)", 0, 120, 45)
                proposed_dso = st.slider("Proposed DSO (days)", 0, 120, 60)
            with col2:
                variable_cost_ratio = st.slider("Variable Cost Ratio", 0.0, 1.0, 0.60)
                required_return = st.slider("Required Return (%)", 5.0, 30.0, 15.0) / 100
                bad_debt_current = st.slider("Current Bad Debt (%)", 0.0, 10.0, 2.0) / 100
                bad_debt_proposed = st.slider("Proposed Bad Debt (%)", 0.0, 10.0, 3.0) / 100
            
            # Analysis
            sales_increase = (proposed_dso / current_dso - 1) * current_sales  # Simplified assumption
            new_sales = current_sales + sales_increase
            
            # Costs
            incremental_sales = sales_increase
            incremental_variable_cost = incremental_sales * variable_cost_ratio
            incremental_investment = (new_sales * (proposed_dso/365) - current_sales * (current_dso/365)) * variable_cost_ratio
            carrying_cost = incremental_investment * required_return
            bad_debt_cost = new_sales * bad_debt_proposed - current_sales * bad_debt_current
            
            incremental_profit = (incremental_sales * (1 - variable_cost_ratio)) - carrying_cost - bad_debt_cost
            
            st.metric("Incremental Sales", f"${incremental_sales:,.0f}")
            st.metric("Carrying Cost", f"${carrying_cost:,.0f}")
            st.metric("Incremental Bad Debt", f"${bad_debt_cost:,.0f}")
            st.metric("Net Incremental Profit", f"${incremental_profit:,.0f}", 
                     "Approve" if incremental_profit > 0 else "Reject")
            
            if incremental_profit > 0:
                st.success("Proposed credit policy is profitable")
            else:
                st.error("Current policy is better")
        
        else:  # Float Calculator
            col1, col2 = st.columns(2)
            with col1:
                average_checks_issued = st.number_input("Average Daily Checks Issued ($)", value=50000)
                average_checks_received = st.number_input("Average Daily Checks Received ($)", value=40000)
                mail_float = st.slider("Mail Float (days)", 0, 10, 2)
                processing_float = st.slider("Processing Float (days)", 0, 5, 1)
                clearing_float = st.slider("Clearing Float (days)", 0, 5, 1)
            with col2:
                interest_rate = st.slider("Opportunity Cost (%)", 1.0, 20.0, 8.0) / 100
                
                # Float calculations
                disbursement_float = average_checks_issued * (mail_float + processing_float + clearing_float)
                collection_float = average_checks_received * (mail_float + processing_float + clearing_float)
                net_float = disbursement_float - collection_float
                
                annual_cost = net_float * interest_rate
                
                st.metric("Disbursement Float", f"${disbursement_float:,.0f}")
                st.metric("Collection Float", f"${collection_float:,.0f}")
                st.metric("Net Float", f"${net_float:,.0f}")
                st.metric("Annual Cost of Float", f"${annual_cost:,.0f}")
                
                st.info("Reducing collection float or increasing disbursement float improves cash position")
    
    with tabs[4]:
        st.subheader("Unit 5 Assessment")
        
        questions = [
            {
                "question": "The cash conversion cycle equals:",
                "options": ["Operating cycle + Payables period", "Operating cycle - Payables period",
                           "Inventory period + Receivables period", "Payables period - Operating cycle"],
                "correct": 1,
                "explanation": "CCC = Inventory Days + Receivable Days - Payable Days = Operating Cycle - Payables Period."
            },
            {
                "question": "An aggressive working capital policy:",
                "options": ["Minimizes risk", "Maximizes liquidity", 
                           "Uses short-term financing for permanent assets", "Maintains high cash balances"],
                "correct": 2,
                "explanation": "Aggressive policies use cheaper short-term financing for part of permanent current assets, increasing risk."
            },
            {
                "question": "The EOQ model minimizes:",
                "options": ["Only ordering costs", "Only carrying costs", 
                           "Total inventory costs", "Stockout costs"],
                "correct": 2,
                "explanation": "EOQ balances ordering costs (decrease with larger orders) against carrying costs (increase with larger orders)."
            }
        ]
        
        score = 0
        for idx, q in enumerate(questions):
            st.markdown(f"**Q{idx+1}. {q['question']}**")
            answer = st.radio(f"Select answer for Q{idx+1}:", q['options'], key=f"q5_{idx}")
            
            if st.button(f"Check Answer {idx+1}", key=f"check5_{idx}"):
                if q['options'].index(answer) == q['correct']:
                    st.success(f"✅ Correct! {q['explanation']}")
                    score += 1
                else:
                    st.error(f"❌ Incorrect. {q['explanation']}")
        
        if st.button("Submit Quiz", key="submit5"):
            st.session_state.quiz_scores['Unit 5'] = score
            st.balloons()
            st.success(f"Quiz submitted! Score: {score}/{len(questions)}")

# Unit 6: Working Capital Management
# Unit 6: Working Capital Management
elif page == "💼 Unit 6: Working Capital":
    st.title("💼 Unit 6: Short-Term Financial Management")
    
    tabs = st.tabs(["📚 Cash Management", "💳 Credit Management", "📦 Inventory", "🧮 Calculator", "📝 Quiz"])
    
    with tabs[0]:
        st.markdown("""
        <div class="card">
            <h3>Cash Management Strategies</h3>
            <p>Optimizing <span class="highlight">cash collection</span>, <span class="highlight">disbursement</span>, 
            and <span class="highlight">investment</span> while maintaining liquidity.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cash management techniques
        technique = st.selectbox("Select Technique:", 
                                ["Lockbox System", "Concentration Banking", "Zero-Balance Accounts"])
        
        col1, col2 = st.columns(2)
        with col1:
            daily_receipts = st.number_input("Average Daily Receipts ($)", value=100000, step=10000)
            current_float = st.slider("Current Float (days)", 0, 10, 5)
            improved_float = st.slider("Improved Float (days)", 0, 10, 2)
        
        with col2:
            interest_rate = st.slider("Annual Interest Rate (%)", 1.0, 15.0, 6.0) / 100
            annual_cost = st.number_input("Annual System Cost ($)", value=25000, step=5000)
            
            # Benefits calculation
            float_reduction = current_float - improved_float
            funds_freed = daily_receipts * float_reduction
            annual_benefit = funds_freed * interest_rate
            net_benefit = annual_benefit - annual_cost
            
            st.metric("Float Reduction", f"{float_reduction} days")
            st.metric("Funds Freed", f"${funds_freed:,.0f}")
            st.metric("Annual Benefit", f"${annual_benefit:,.0f}")
            st.metric("Net Annual Benefit", f"${net_benefit:,.0f}", 
                     "Implement" if net_benefit > 0 else "Do Not Implement")
        
        # Investment options
        st.subheader("Cash Investment Options")
        
        instruments = pd.DataFrame({
            'Instrument': ['Treasury Bills', 'Commercial Paper', 'CDs', 'Repurchase Agreements', 'Money Market Funds'],
            'Maturity': ['4-52 weeks', '1-270 days', '1 month-1 year', 'Overnight-30 days', 'Variable'],
            'Yield': ['1.5%', '2.0%', '2.2%', '1.8%', '1.9%'],
            'Risk': ['None', 'Very Low', 'Low', 'Very Low', 'Low'],
            'Liquidity': ['High', 'Medium', 'Low', 'High', 'High']
        })
        
        st.dataframe(instruments, use_container_width=True, hide_index=True)
        
        # Cash holding motives
        st.subheader("Motives for Holding Cash")
        
        motive = st.selectbox("Explore Motive:", 
                             ["Transaction Motive", "Precautionary Motive", "Speculative Motive", "Compensating Balances"])
        
        motives = {
            "Transaction Motive": {
                "description": "Cash needed for day-to-day operations and routine payments",
                "factors": ["Payment cycles", "Collection patterns", "Seasonal fluctuations"],
                "management": "Efficient cash management systems, float reduction"
            },
            "Precautionary Motive": {
                "description": "Safety stock of cash for unexpected needs and emergencies",
                "factors": ["Cash flow uncertainty", "Credit access", "Risk tolerance"],
                "management": "Credit lines, marketable securities, insurance"
            },
            "Speculative Motive": {
                "description": "Cash held to take advantage of unexpected opportunities",
                "factors": ["Investment opportunities", "Acquisition targets", "Market downturns"],
                "management": "Keep in liquid investments, ready access"
            },
            "Compensating Balances": {
                "description": "Minimum balances required by banks for services",
                "factors": ["Loan agreements", "Service fees", "Banking relationships"],
                "management": "Negotiate terms, consolidate accounts"
            }
        }
        
        m = motives[motive]
        st.markdown(f"""
        <div class="card">
            <h4>{motive}</h4>
            <p><b>Description:</b> {m['description']}</p>
            <p><b>Key Factors:</b></p>
            <ul>{''.join([f'<li>{f}</li>' for f in m['factors']])}</ul>
            <p><b>Management Approach:</b> {m['management']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.subheader("Credit and Receivables Management")
        
        # Credit policy components
        st.write("**Credit Policy Components:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="card">
                <h4>Credit Standards</h4>
                <p>Tightness of credit requirements</p>
                <ul>
                    <li>5 C's of Credit</li>
                    <li>Credit scoring</li>
                    <li>Trade references</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4>Credit Terms</h4>
                <p>Payment requirements</p>
                <ul>
                    <li>Credit period</li>
                    <li>Cash discount</li>
                    <li>Discount period</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="card">
                <h4>Collection Policy</h4>
                <p>Overdue account procedures</p>
                <ul>
                    <li>Reminder notices</li>
                    <li>Collection calls</li>
                    <li>Legal action</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # 5 C's interactive
        st.subheader("5 C's of Credit Analysis")
        
        c_selected = st.selectbox("Explore the 5 C's:", 
                                 ["Character", "Capacity", "Capital", "Collateral", "Conditions"])
        
        c_details = {
            "Character": {"desc": "Willingness to pay", "indicators": ["Credit history", "Payment record", "References"], "assessment": "Credit reports, trade references"},
            "Capacity": {"desc": "Ability to pay", "indicators": ["Cash flow", "Debt service coverage", "Financial ratios"], "assessment": "Financial statement analysis"},
            "Capital": {"desc": "Financial reserves", "indicators": ["Net worth", "Working capital", "Equity base"], "assessment": "Balance sheet strength"},
            "Collateral": {"desc": "Security for credit", "indicators": ["Asset quality", "Lien position", "Appraised value"], "assessment": "Asset valuation"},
            "Conditions": {"desc": "Economic environment", "indicators": ["Industry trends", "Economic cycle", "Competitive position"], "assessment": "Macro analysis"}
        }
        
        c = c_details[c_selected]
        st.markdown(f"""
        <div class="card">
            <h3>{c_selected}: {c['desc']}</h3>
            <p><b>Key Indicators:</b> {', '.join(c['indicators'])}</p>
            <p><b>Assessment Method:</b> {c['assessment']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Credit terms analysis
        st.subheader("Credit Terms Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            credit_terms = st.text_input("Credit Terms (e.g., 2/10 net 30)", value="2/10 net 30")
            invoice_amount = st.number_input("Invoice Amount ($)", value=10000)
        
        with col2:
            # Parse credit terms
            try:
                if "/" in credit_terms and "net" in credit_terms:
                    parts = credit_terms.split()
                    discount_pct = float(parts[0].split("/")[0]) / 100
                    discount_days = int(parts[0].split("/")[1])
                    net_days = int(parts[1].replace("net", ""))
                    
                    # Cost of not taking discount
                    cost_of_trade_credit = (discount_pct / (1 - discount_pct)) * (365 / (net_days - discount_days))
                    
                    st.metric("Discount Percentage", f"{discount_pct*100:.0f}%")
                    st.metric("Discount Period", f"{discount_days} days")
                    st.metric("Net Period", f"{net_days} days")
                    st.metric("Cost of Not Taking Discount", f"{cost_of_trade_credit*100:.1f}% annualized")
                    
                    discount_savings = invoice_amount * discount_pct
                    st.success(f"Take discount: Save ${discount_savings:.2f} if pay by day {discount_days}")
            except:
                st.error("Please enter terms in format: '2/10 net 30'")
        
        # Aging schedule
        st.subheader("Aging of Accounts Receivable")
        
        aging_data = pd.DataFrame({
            'Age Category': ['0-30 days', '31-60 days', '61-90 days', '90+ days'],
            'Amount ($)': [50000, 30000, 15000, 5000],
            'Percentage': [50, 30, 15, 5],
            'Industry Avg (%)': [60, 25, 10, 5]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=aging_data['Age Category'], y=aging_data['Percentage'], 
                            name='Your Company', marker_color='#e74c3c'))
        fig.add_trace(go.Bar(x=aging_data['Age Category'], y=aging_data['Industry Avg (%)'], 
                            name='Industry Average', marker_color='#95a5a6'))
        fig.update_layout(title="Aging Comparison", barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        if aging_data['Percentage'].iloc[2:].sum() > aging_data['Industry Avg (%)'].iloc[2:].sum():
            st.warning("⚠️ Higher than average receivables in older categories - collection issues suspected")
        
        # Collection policy timeline
        st.subheader("Collection Policy Timeline")
        
        days_past_due = st.slider("Days Past Due", 0, 120, 45)
        
        if days_past_due <= 30:
            action = "Send reminder statement"
            cost = "Low"
            effectiveness = "High"
        elif days_past_due <= 60:
            action = "Phone call, demand letter"
            cost = "Medium"
            effectiveness = "Medium"
        elif days_past_due <= 90:
            action = "Collection agency, attorney letter"
            cost = "High"
            effectiveness = "Low-Medium"
        else:
            action = "Legal action, write-off consideration"
            cost = "Very High"
            effectiveness = "Low"
        
        st.markdown(f"""
        <div class="card">
            <h4>Recommended Action at {days_past_due} days</h4>
            <p><b>Action:</b> {action}</p>
            <p><b>Cost:</b> {cost}</p>
            <p><b>Effectiveness:</b> {effectiveness}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.subheader("Inventory Management")
        
        # ABC Analysis
        st.write("**ABC Classification System:**")
        
        abc_data = pd.DataFrame({
            'Category': ['A', 'B', 'C'],
            'Items (%)': [10, 20, 70],
            'Value (%)': [70, 20, 10],
            'Control Level': ['Tight', 'Moderate', 'Simple'],
            'Review Frequency': ['Continuous', 'Periodic', 'Annual'],
            'Safety Stock': ['Low', 'Medium', 'High']
        })
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(abc_data, use_container_width=True, hide_index=True)
        
        with col2:
            # Pareto chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['A Items', 'B Items', 'C Items'],
                y=[70, 20, 10],
                name='Cumulative Value %',
                marker_color=['#e74c3c', '#f39c12', '#2ecc71']
            ))
            fig.update_layout(title="Pareto Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Inventory control systems
        system = st.selectbox("Select Inventory System:", 
                             ["Fixed Order Quantity (Q-System)", "Fixed Order Period (P-System)", "Just-in-Time"])
        
        if system == "Fixed Order Quantity (Q-System)":
            st.info("""
            **Continuous Review System:**
            - Order fixed quantity Q when inventory reaches reorder point ROP
            - Suitable for: High-value items (Category A)
            - Requires: Continuous monitoring
            """)
            
            # ROP calculation
            col1, col2 = st.columns(2)
            with col1:
                daily_demand = st.number_input("Average Daily Demand", value=100)
                lead_time = st.slider("Lead Time (days)", 1, 30, 7)
                demand_std = st.number_input("Demand Std Deviation", value=20)
            with col2:
                service_level = st.slider("Service Level (%)", 80, 99, 95) / 100
                z_score = norm.ppf(service_level)
                safety_stock = z_score * demand_std * np.sqrt(lead_time)
                rop = daily_demand * lead_time + safety_stock
                
                st.metric("Reorder Point (ROP)", f"{rop:.0f} units")
                st.metric("Safety Stock", f"{safety_stock:.0f} units")
                st.metric("Z-Score", f"{z_score:.2f}")
        
        elif system == "Fixed Order Period (P-System)":
            st.info("""
            **Periodic Review System:**
            - Order up to target level T every P periods
            - Suitable for: Multiple items from same supplier
            - Advantage: Coordinated ordering, less monitoring
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                review_period = st.slider("Review Period (days)", 7, 30, 14)
                avg_daily_demand = st.number_input("Average Daily Demand", value=100, key="p_system_demand")
                lead_time = st.slider("Lead Time (days)", 1, 14, 5, key="p_system_lt")
            with col2:
                demand_variability = st.number_input("Demand Std Dev", value=25, key="p_system_std")
                service_level = st.slider("Service Level (%)", 80, 99, 95, key="p_system_sl") / 100
                
                protection_period = review_period + lead_time
                target_level = (avg_daily_demand * protection_period + 
                               norm.ppf(service_level) * demand_variability * np.sqrt(protection_period))
                
                st.metric("Protection Period", f"{protection_period} days")
                st.metric("Target Inventory Level", f"{target_level:.0f} units")
        
        else:  # Just-in-Time
            st.info("""
            **Just-in-Time (JIT) Inventory:**
            - Minimum inventory, frequent deliveries
            - Requirements: Reliable suppliers, quality control, flexible production
            - Risk: Supply chain vulnerability
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Typical Inventory Reduction", "70-90%")
                st.metric("Delivery Frequency", "Multiple times daily")
            with col2:
                st.metric("Quality Requirement", "Near-zero defects")
                st.metric("Supplier Relationship", "Strategic partnership")
            
            st.warning("⚠️ Risks: Supply disruption, price volatility, loss of volume discounts")
        
        # EOQ deep dive
        st.subheader("Economic Order Quantity (EOQ) Deep Dive")
        
        col1, col2 = st.columns(2)
        with col1:
            annual_demand = st.number_input("Annual Demand (units)", value=10000, key="eoq_d")
            order_cost = st.number_input("Ordering Cost ($)", value=100, key="eoq_s")
            carrying_cost_unit = st.number_input("Carrying Cost per Unit ($)", value=5, key="eoq_h")
        with col2:
            unit_cost = st.number_input("Unit Cost ($)", value=25, key="eoq_c")
            carrying_cost_pct = st.slider("Carrying Cost (% of value)", 0.0, 50.0, 20.0, key="eoq_hp") / 100
            
            # Determine which carrying cost to use
            if carrying_cost_unit > 0:
                h = carrying_cost_unit
            else:
                h = carrying_cost_pct * unit_cost
        
        # EOQ calculations
        eoq = np.sqrt((2 * annual_demand * order_cost) / h)
        total_ordering_cost = (annual_demand / eoq) * order_cost
        total_carrying_cost = (eoq / 2) * h
        total_cost = total_ordering_cost + total_carrying_cost
        orders_per_year = annual_demand / eoq
        cycle_time = 365 / orders_per_year
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("EOQ", f"{eoq:.0f} units")
        with col2:
            st.metric("Total Cost", f"${total_cost:,.0f}")
        with col3:
            st.metric("Orders/Year", f"{orders_per_year:.1f}")
        with col4:
            st.metric("Cycle Time", f"{cycle_time:.0f} days")
        
        # EOQ sensitivity
        st.subheader("EOQ Sensitivity Analysis")
        
        sensitivity = st.selectbox("Vary Parameter:", ["Demand", "Ordering Cost", "Carrying Cost"])
        
        if sensitivity == "Demand":
            range_vals = np.linspace(annual_demand * 0.5, annual_demand * 2, 100)
            eoq_vals = [np.sqrt((2 * d * order_cost) / h) for d in range_vals]
            fig = px.line(x=range_vals, y=eoq_vals, labels={'x': 'Annual Demand', 'y': 'EOQ'})
        elif sensitivity == "Ordering Cost":
            range_vals = np.linspace(order_cost * 0.5, order_cost * 3, 100)
            eoq_vals = [np.sqrt((2 * annual_demand * s) / h) for s in range_vals]
            fig = px.line(x=range_vals, y=eoq_vals, labels={'x': 'Ordering Cost', 'y': 'EOQ'})
        else:
            range_vals = np.linspace(h * 0.5, h * 2, 100)
            eoq_vals = [np.sqrt((2 * annual_demand * order_cost) / c) for c in range_vals]
            fig = px.line(x=range_vals, y=eoq_vals, labels={'x': 'Carrying Cost', 'y': 'EOQ'})
        
        fig.add_vline(x={'Demand': annual_demand, 'Ordering Cost': order_cost, 'Carrying Cost': h}[sensitivity],
                     line_dash="dash", annotation_text="Current")
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.subheader("Working Capital Calculators")
        
        calc = st.radio("Select Calculator:", ["Credit Scoring", "Collection Efficiency", "Inventory Turnover", "Cash Conversion Cycle"])
        
        if calc == "Credit Scoring":
            st.write("Enter customer data for credit score calculation:")
            
            col1, col2 = st.columns(2)
            with col1:
                credit_history = st.slider("Credit History Score (0-100)", 0, 100, 80)
                debt_income = st.slider("Debt-to-Income Ratio (%)", 0, 100, 30)
                employment = st.slider("Employment Stability (years)", 0, 20, 5)
            with col2:
                net_worth = st.number_input("Net Worth ($000s)", value=50)
                payment_history = st.slider("Payment History Score (0-100)", 0, 100, 85)
            
            # Weighted scoring model
            score = (credit_history * 0.25 + (100 - debt_income) * 0.20 + 
                    min(employment * 5, 100) * 0.15 + min(net_worth, 100) * 0.25 + 
                    payment_history * 0.15)
            
            st.metric("Credit Score", f"{score:.0f}/100")
            
            # Visual gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Credit Score"},
                gauge = {'axis': {'range': [None, 100]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 40], 'color': "#f8d7da"},
                            {'range': [40, 60], 'color': "#fff3cd"},
                            {'range': [60, 80], 'color': "#d4edda"},
                            {'range': [80, 100], 'color': "#c3e6cb"}],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': score}}
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            if score >= 80:
                st.success("Excellent - Extend maximum credit")
            elif score >= 60:
                st.info("Good - Standard credit terms")
            elif score >= 40:
                st.warning("Fair - Limited credit, monitor closely")
            else:
                st.error("Poor - Cash only or secured transactions")
        
        elif calc == "Collection Efficiency":
            col1, col2 = st.columns(2)
            with col1:
                beginning_ar = st.number_input("Beginning A/R ($)", value=500000)
                credit_sales = st.number_input("Credit Sales ($)", value=2000000)
                ending_ar = st.number_input("Ending A/R ($)", value=600000)
            with col2:
                beginning_total = st.number_input("Beginning Total Assets ($)", value=2000000)
                ending_total = st.number_input("Ending Total Assets ($)", value=2200000)
            
            # Calculations
            avg_ar = (beginning_ar + ending_ar) / 2
            avg_total = (beginning_total + ending_total) / 2
            
            ar_turnover = credit_sales / avg_ar
            dso = 365 / ar_turnover
            ar_to_total = avg_ar / avg_total
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("A/R Turnover", f"{ar_turnover:.2f}x")
            with col2:
                st.metric("Days Sales Outstanding (DSO)", f"{dso:.0f} days")
            with col3:
                st.metric("A/R to Total Assets", f"{ar_to_total*100:.1f}%")
            
            # Trend analysis
            if dso > 60:
                st.error("DSO above 60 days - collection issues likely")
            elif dso > 45:
                st.warning("DSO elevated - monitor closely")
            else:
                st.success("DSO healthy")
        
        elif calc == "Inventory Turnover":
            col1, col2 = st.columns(2)
            with col1:
                cogs = st.number_input("COGS ($)", value=5000000, key="inv_cogs")
                beginning_inv = st.number_input("Beginning Inventory ($)", value=800000)
                ending_inv = st.number_input("Ending Inventory ($)", value=1200000)
            with col2:
                avg_inventory = (beginning_inv + ending_inv) / 2
                turnover = cogs / avg_inventory
                days_inventory = 365 / turnover
                
                st.metric("Average Inventory", f"${avg_inventory:,.0f}")
                st.metric("Inventory Turnover", f"{turnover:.2f}x")
                st.metric("Days Inventory Outstanding", f"{days_inventory:.0f} days")
                
                if turnover < 4:
                    st.warning("Low turnover - excess inventory or obsolescence risk")
                elif turnover > 12:
                    st.info("High turnover - efficient but risk of stockouts")
                else:
                    st.success("Healthy turnover ratio")
        
        else:  # Cash Conversion Cycle
            st.subheader("Cash Conversion Cycle Calculator")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                inventory_days = st.slider("Inventory Days", 0, 180, 60, key="ccc_inv")
                receivable_days = st.slider("Receivable Days (DSO)", 0, 180, 45, key="ccc_ar")
            with col2:
                payable_days = st.slider("Payable Days (DPO)", 0, 180, 30, key="ccc_ap")
                cogs_ccc = st.number_input("Annual COGS ($M)", value=50, key="ccc_cogs")
            with col3:
                sales_ccc = st.number_input("Annual Sales ($M)", value=80, key="ccc_sales")
            
            operating_cycle = inventory_days + receivable_days
            ccc = operating_cycle - payable_days
            working_capital_need = (ccc / 365) * cogs_ccc
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Operating Cycle", f"{operating_cycle} days")
            with col2:
                st.metric("Cash Conversion Cycle", f"{ccc} days")
            with col3:
                st.metric("Working Capital Need", f"${working_capital_need:.2f}M")
            
            # CCC visualization
            fig = go.Figure()
            
            # Timeline
            fig.add_trace(go.Bar(
                name='Inventory Period',
                x=['Operating Cycle'],
                y=[inventory_days],
                marker_color='#3498db'
            ))
            fig.add_trace(go.Bar(
                name='Receivable Period',
                x=['Operating Cycle'],
                y=[receivable_days],
                base=[inventory_days],
                marker_color='#2ecc71'
            ))
            fig.add_trace(go.Bar(
                name='Payable Period (negative)',
                x=['Cash Cycle'],
                y=[-payable_days],
                marker_color='#e74c3c'
            ))
            fig.add_trace(go.Bar(
                name='Cash Conversion Cycle',
                x=['Cash Cycle'],
                y=[ccc],
                base=[0],
                marker_color='#f39c12'
            ))
            
            fig.update_layout(title="Cash Conversion Cycle Components", barmode='relative', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Improvement scenarios
            st.subheader("Improvement Scenarios")
            
            scenario = st.selectbox("Select Improvement:", 
                                   ["Reduce Inventory 10 days", "Reduce Receivables 5 days", 
                                    "Extend Payables 5 days", "All Improvements"])
            
            improvements = {
                "Reduce Inventory 10 days": (-10, 0, 0),
                "Reduce Receivables 5 days": (0, -5, 0),
                "Extend Payables 5 days": (0, 0, 5),
                "All Improvements": (-10, -5, 5)
            }
            
            inv_chg, rec_chg, pay_chg = improvements[scenario]
            new_ccc = ccc + inv_chg + rec_chg + pay_chg
            savings = (ccc - new_ccc) / 365 * cogs_ccc
            
            st.metric("New CCC", f"{new_ccc} days", f"{new_ccc - ccc} days")
            st.metric("Working Capital Freed", f"${savings:.2f}M")
    
    with tabs[4]:
        st.subheader("Unit 6 Assessment")
        
        questions = [
            {
                "question": "The primary objective of cash management is to:",
                "options": ["Maximize cash holdings", "Minimize transaction costs", 
                           "Balance liquidity and profitability", "Eliminate all risk"],
                "correct": 2,
                "explanation": "Cash management seeks optimal balance between having enough liquidity and investing excess profitably."
            },
            {
                "question": "In the 5 C's of credit, 'Capacity' refers to:",
                "options": ["Moral character of borrower", "Ability to repay from cash flow",
                           "Financial reserves", "Collateral available"],
                "correct": 1,
                "explanation": "Capacity assesses the borrower's ability to generate sufficient cash flow to service debt."
            },
            {
                "question": "ABC inventory analysis suggests:",
                "options": ["All items deserve equal attention", "Most attention on highest value items",
                           "Focus on fastest moving items", "Eliminate all C items"],
                "correct": 1,
                "explanation": "Category A items (few in number, high in value) deserve tightest control and most attention."
            },
            {
                "question": "The EOQ model minimizes:",
                "options": ["Only ordering costs", "Only carrying costs", 
                           "Total inventory costs", "Stockout costs"],
                "correct": 2,
                "explanation": "EOQ balances ordering costs (decrease with larger orders) against carrying costs (increase with larger orders)."
            },
            {
                "question": "A lockbox system is used to:",
                "options": ["Reduce inventory", "Accelerate cash collections", 
                           "Slow down disbursements", "Improve credit analysis"],
                "correct": 1,
                "explanation": "Lockboxes reduce mail and processing float, accelerating availability of collected funds."
            }
        ]
        
        score = 0
        for idx, q in enumerate(questions):
            st.markdown(f"**Q{idx+1}. {q['question']}**")
            answer = st.radio(f"Select answer for Q{idx+1}:", q['options'], key=f"q6_{idx}")
            
            if st.button(f"Check Answer {idx+1}", key=f"check6_{idx}"):
                if q['options'].index(answer) == q['correct']:
                    st.success(f"✅ Correct! {q['explanation']}")
                    score += 1
                else:
                    st.error(f"❌ Incorrect. {q['explanation']}")
        
        if st.button("Submit Quiz", key="submit6"):
            st.session_state.quiz_scores['Unit 6'] = score
            st.balloons()
            st.success(f"Quiz submitted! Score: {score}/{len(questions)}")

# Unit 7: Derivatives
elif page == "🎯 Unit 7: Derivatives":
    st.title("🎯 Unit 7: Introduction to Derivatives")
    
    tabs = st.tabs(["📚 Options", "📈 Futures", "🔧 Warrants & Convertibles", "🧮 Calculator", "📝 Quiz"])
    
    with tabs[0]:
        st.markdown(textwrap.dedent("""
        <div class="card">
            <h3>Options Fundamentals</h3>
            <p>Contracts providing the <span class="highlight">right but not obligation</span> to buy (call) 
            or sell (put) an underlying asset at a predetermined price.</p>
        </div>
        """), unsafe_allow_html=True)
        
        # Option basics
        option_type = st.selectbox("Select Option Type:", ["Call Option", "Put Option"])
        
        col1, col2 = st.columns(2)
        with col1:
            S = st.number_input("Current Stock Price ($)", value=50.0, key="opt_S")
            K = st.number_input("Strike Price ($)", value=50.0, key="opt_K")
            T = st.slider("Time to Expiration (years)", 0.1, 2.0, 1.0, key="opt_T")
        with col2:
            r = st.slider("Risk-free Rate (%)", 0.0, 10.0, 5.0, key="opt_r") / 100
            sigma = st.slider("Volatility (%)", 10.0, 100.0, 30.0, key="opt_sigma") / 100
        
        # Calculate option price
        price, d1, d2 = black_scholes(S, K, T, r, sigma, 'call' if option_type == "Call Option" else 'put')
        
        # Greeks calculation (simplified)
        delta = norm.cdf(d1) if option_type == "Call Option" else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Option Price", f"${price:.2f}")
        with col2:
            st.metric("Delta", f"{delta:.3f}")
        with col3:
            st.metric("Gamma", f"{gamma:.4f}")
        with col4:
            st.metric("Theta", f"{theta:.2f}")
        with col5:
            st.metric("Vega", f"{vega:.2f}")
        
        # Payoff diagram
        st.subheader("Payoff Diagram")
        
        stock_prices = np.linspace(max(0, K * 0.5), K * 1.5, 100)
        
        if option_type == "Call Option":
            payoffs = np.maximum(stock_prices - K, 0)
            profits = payoffs - price
        else:
            payoffs = np.maximum(K - stock_prices, 0)
            profits = payoffs - price
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_prices, y=payoffs, mode='lines', 
                                name='Payoff at Expiration', line=dict(color='#2ecc71', width=3)))
        fig.add_trace(go.Scatter(x=stock_prices, y=profits, mode='lines', 
                                name='Profit/Loss', line=dict(color='#e74c3c', width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=K, line_dash="dash", line_color="blue", annotation_text="Strike")
        fig.update_layout(title=f"{option_type} Payoff Diagram", 
                         xaxis_title="Stock Price at Expiration", 
                         yaxis_title="Payoff ($)", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Option strategies
        st.subheader("Option Strategies")
        
        strategy = st.selectbox("Select Strategy:", 
                               ["Covered Call", "Protective Put", "Straddle", "Bull Spread"])
        
        if strategy == "Covered Call":
            st.info(textwrap.dedent("""
            **Covered Call:** Own stock, sell call option
            - Generates income from premium
            - Caps upside potential
            - Best for: Neutral to slightly bullish outlook
            """))
        elif strategy == "Protective Put":
            st.info(textwrap.dedent("""
            **Protective Put:** Own stock, buy put option
            - Insurance against price decline
            - Maintains upside potential
            - Cost: Put premium
            """))
        elif strategy == "Straddle":
            st.info(textwrap.dedent("""
            **Straddle:** Buy call and put at same strike
            - Profits from large price move in either direction
            - Loss if price stays stable
            - Used when expecting volatility
            """))
        else:
            st.info(textwrap.dedent("""
            **Bull Spread:** Buy low strike call, sell high strike call
            - Limits both upside and downside
            - Cheaper than outright call
            - Moderate bullish outlook
            """))
    
    with tabs[1]:
        st.subheader("Futures and Forwards")
        
        # Futures vs Forwards comparison
        comparison = pd.DataFrame({
            'Feature': ['Trading', 'Standardization', 'Liquidity', 'Credit Risk', 'Settlement'],
            'Futures': ['Exchange-traded', 'Standardized', 'High', 'Minimal (clearinghouse)', 'Daily marking-to-market'],
            'Forwards': ['OTC', 'Customized', 'Low', 'Counterparty risk', 'At maturity']
        })
        
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        
        # Hedging example
        st.subheader("Hedging with Futures")
        
        hedge_type = st.selectbox("Select Hedge Type:", ["Long Hedge (Buy Futures)", "Short Hedge (Sell Futures)"])
        
        col1, col2 = st.columns(2)
        with col1:
            spot_price = st.number_input("Current Spot Price ($)", value=100.0)
            future_price = st.number_input("Futures Price ($)", value=102.0)
            quantity = st.number_input("Quantity", value=1000)
        
        with col2:
            if hedge_type == "Long Hedge (Buy Futures)":
                st.write("**Scenario:** Need to buy commodity in future, afraid price will rise")
                price_change = st.slider("Price Change at Maturity (%)", -20, 20, 5) / 100
            else:
                st.write("**Scenario:** Own commodity, afraid price will fall")
                price_change = st.slider("Price Change at Maturity (%)", -20, 20, -5) / 100
            
            new_spot = spot_price * (1 + price_change)
            
            if hedge_type == "Long Hedge (Buy Futures)":
                spot_loss = (new_spot - spot_price) * quantity  # Extra cost
                futures_gain = (new_spot - future_price) * quantity  # Gain on futures
            else:
                spot_loss = (new_spot - spot_price) * quantity  # Loss on spot
                futures_gain = (future_price - new_spot) * quantity  # Gain on futures
            
            net_result = spot_loss + futures_gain
            
            st.metric("Spot Market Impact", f"${spot_loss:,.0f}")
            st.metric("Futures Gain/Loss", f"${futures_gain:,.0f}")
            st.metric("Net Hedged Result", f"${net_result:,.0f}", "Effective hedge" if abs(net_result) < abs(spot_loss) * 0.1 else "Basis risk")
    
    with tabs[2]:
        st.subheader("Warrants and Convertibles")
        
        instrument = st.selectbox("Select Instrument:", ["Warrants", "Convertible Bonds", "Convertible Preferred"])
        
        if instrument == "Warrants":
            col1, col2 = st.columns(2)
            with col1:
                stock_price = st.number_input("Stock Price ($)", value=50.0, key="warr_S")
                exercise_price = st.number_input("Exercise Price ($)", value=45.0, key="warr_K")
                warrant_price = st.number_input("Warrant Price ($)", value=8.0)
                shares_per_warrant = st.number_input("Shares per Warrant", value=1)
            
            with col2:
                intrinsic_value = max(0, stock_price - exercise_price) * shares_per_warrant
                time_premium = warrant_price - intrinsic_value
                
                st.metric("Intrinsic Value", f"${intrinsic_value:.2f}")
                st.metric("Time Premium", f"${time_premium:.2f}")
                st.metric("Premium %", f"{(warrant_price/intrinsic_value - 1)*100:.1f}%" if intrinsic_value > 0 else "N/A")
                
                leverage = (stock_price * shares_per_warrant) / warrant_price
                st.metric("Leverage Ratio", f"{leverage:.2f}x")
        
        else:  # Convertibles
            col1, col2 = st.columns(2)
            with col1:
                face_value = st.number_input("Face Value ($)", value=1000, key="conv_face")
                coupon_rate = st.slider("Coupon Rate (%)", 0.0, 10.0, 4.0, key="conv_coupon") / 100
                conversion_price = st.number_input("Conversion Price ($)", value=50.0, key="conv_price")
                stock_price = st.number_input("Stock Price ($)", value=45.0, key="conv_stock")
            
            with col2:
                conversion_ratio = face_value / conversion_price
                conversion_value = conversion_ratio * stock_price
                conversion_premium = (conversion_price - stock_price) / stock_price * 100
                
                st.metric("Conversion Ratio", f"{conversion_ratio:.2f} shares")
                st.metric("Conversion Value", f"${conversion_value:.2f}")
                st.metric("Conversion Premium", f"{conversion_premium:.1f}%")
                
                floor_value = max(conversion_value, face_value * 0.9)  # Approximate bond floor
                st.metric("Floor Value", f"${floor_value:.2f}")
    
    with tabs[3]:
        st.subheader("Derivatives Calculator")
        
        calc = st.radio("Select Calculator:", ["Black-Scholes", "Binomial Option Pricing", "Implied Volatility"])
        
        if calc == "Black-Scholes":
            st.write("Professional option pricing model")
            
            col1, col2 = st.columns(2)
            with col1:
                bs_S = st.number_input("Stock Price", value=100.0, key="bs_S")
                bs_K = st.number_input("Strike Price", value=100.0, key="bs_K")
                bs_T = st.number_input("Time to Expiration (years)", value=1.0, key="bs_T")
            with col2:
                bs_r = st.number_input("Risk-free Rate", value=0.05, key="bs_r")
                bs_sigma = st.number_input("Volatility", value=0.30, key="bs_sigma")
                option_type = st.selectbox("Option Type", ["Call", "Put"], key="bs_type")
            
            call_price, d1, d2 = black_scholes(bs_S, bs_K, bs_T, bs_r, bs_sigma, 'call')
            put_price = call_price - bs_S + bs_K * np.exp(-bs_r * bs_T)  # Put-call parity
            
            if option_type == "Call":
                st.metric("Call Option Price", f"${call_price:.2f}")
            else:
                st.metric("Put Option Price", f"${put_price:.2f}")
            
            # Sensitivity analysis
            st.subheader("Sensitivity Analysis")
            sens_param = st.selectbox("Vary Parameter:", ["Stock Price", "Volatility", "Time"])
            
            if sens_param == "Stock Price":
                S_range = np.linspace(bs_K * 0.5, bs_K * 1.5, 50)
                prices = [black_scholes(s, bs_K, bs_T, bs_r, bs_sigma, 'call')[0] for s in S_range]
                fig = px.line(x=S_range, y=prices, labels={'x': 'Stock Price', 'y': 'Option Price'})
            elif sens_param == "Volatility":
                sigma_range = np.linspace(0.1, 1.0, 50)
                prices = [black_scholes(bs_S, bs_K, bs_T, bs_r, s, 'call')[0] for s in sigma_range]
                fig = px.line(x=sigma_range, y=prices, labels={'x': 'Volatility', 'y': 'Option Price'})
            else:
                T_range = np.linspace(0.1, 2.0, 50)
                prices = [black_scholes(bs_S, bs_K, t, bs_r, bs_sigma, 'call')[0] for t in T_range]
                fig = px.line(x=T_range, y=prices, labels={'x': 'Time to Expiration', 'y': 'Option Price'})
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif calc == "Binomial Option Pricing":
            st.write("Discrete-time option pricing model")
            
            col1, col2 = st.columns(2)
            with col1:
                bin_S = st.number_input("Stock Price", value=50.0, key="bin_S")
                bin_K = st.number_input("Strike Price", value=50.0, key="bin_K")
                bin_u = st.slider("Up Factor", 1.01, 2.0, 1.2, key="bin_u")
                bin_d = st.slider("Down Factor", 0.1, 0.99, 0.8, key="bin_d")
            with col2:
                bin_r = st.slider("Risk-free Rate (%)", 0.0, 20.0, 5.0, key="bin_r") / 100
                periods = st.slider("Number of Periods", 1, 5, 2, key="bin_n")
            
            # Risk-neutral probability
            p = (1 + bin_r - bin_d) / (bin_u - bin_d)
            
            st.metric("Risk-Neutral Probability (p)", f"{p:.3f}")
            
            # Build tree (simplified for 2 periods max display)
            if periods <= 2:
                st.write("**Stock Price Tree:**")
                for i in range(periods + 1):
                    level_prices = []
                    for j in range(i + 1):
                        price = bin_S * (bin_u ** (i - j)) * (bin_d ** j)
                        level_prices.append(f"${price:.2f}")
                    st.write(f"Period {i}: {' → '.join(level_prices)}")
        
        else:  # Implied Volatility
            st.write("Calculate volatility implied by market price")
            
            col1, col2 = st.columns(2)
            with col1:
                iv_S = st.number_input("Stock Price", value=100.0, key="iv_S")
                iv_K = st.number_input("Strike Price", value=100.0, key="iv_K")
                iv_T = st.number_input("Time (years)", value=1.0, key="iv_T")
                iv_r = st.number_input("Risk-free Rate", value=0.05, key="iv_r")
            with col2:
                market_price = st.number_input("Market Option Price", value=10.0, key="iv_price")
                option_type = st.selectbox("Option Type", ["Call", "Put"], key="iv_type")
            
            # Simple bisection method for implied vol
            def implied_volatility(S, K, T, r, market_price, option_type):
                sigma_low, sigma_high = 0.001, 5.0
                for _ in range(100):
                    sigma_mid = (sigma_low + sigma_high) / 2
                    price, _, _ = black_scholes(S, K, T, r, sigma_mid, option_type.lower())
                    if price > market_price:
                        sigma_high = sigma_mid
                    else:
                        sigma_low = sigma_mid
                return (sigma_low + sigma_high) / 2
            
            iv = implied_volatility(iv_S, iv_K, iv_T, iv_r, market_price, option_type)
            st.metric("Implied Volatility", f"{iv*100:.2f}%")
            
            if iv > 0.50:
                st.info("High implied volatility suggests market expects large price swings")
            elif iv < 0.15:
                st.info("Low implied volatility suggests market expects stable prices")
    
    with tabs[4]:
        st.subheader("Unit 7 Assessment")
        
        questions = [
            {
                "question": "A call option is in-the-money when:",
                "options": ["Stock price < Strike price", "Stock price > Strike price",
                           "Stock price = Strike price", "Option has time value"],
                "correct": 1,
                "explanation": "Call options are ITM when stock price exceeds strike price (can exercise profitably)."
            },
            {
                "question": "Futures differ from forwards in that futures:",
                "options": ["Are customized contracts", "Have higher credit risk",
                           "Are marked-to-market daily", "Trade over-the-counter"],
                "correct": 2,
                "explanation": "Daily marking-to-market and clearinghouse guarantee distinguish futures from forwards."
            },
            {
                "question": "The conversion value of a convertible bond is:",
                "options": ["Face value of bond", "Stock price × Conversion ratio",
                           "Bond price minus stock price", "Coupon payment × Years"],
                "correct": 1,
                "explanation": "Conversion value = Current stock price × Number of shares received upon conversion."
            }
        ]
        
        score = 0
        for idx, q in enumerate(questions):
            st.markdown(f"**Q{idx+1}. {q['question']}**")
            answer = st.radio(f"Select answer for Q{idx+1}:", q['options'], key=f"q7_{idx}")
            
            if st.button(f"Check Answer {idx+1}", key=f"check7_{idx}"):
                if q['options'].index(answer) == q['correct']:
                    st.success(f"✅ Correct! {q['explanation']}")
                    score += 1
                else:
                    st.error(f"❌ Incorrect. {q['explanation']}")
        
        if st.button("Submit Quiz", key="submit7"):
            st.session_state.quiz_scores['Unit 7'] = score
            st.balloons()
            st.success(f"Quiz submitted! Score: {score}/{len(questions)}")

# Unit 8: Special Topics
elif page == "⚠️ Unit 8: Special Topics":
    st.title("⚠️ Unit 8: Mergers, Acquisitions & Financial Distress")
    
    tabs = st.tabs(["📚 M&A Fundamentals", "⚠️ Financial Distress", "🔍 Z-Score Calculator", "🧮 Calculator", "📝 Quiz"])
    
    with tabs[0]:
        st.markdown(textwrap.dedent("""
        <div class="card">
            <h3>Mergers and Acquisitions</h3>
            <p>Strategic combinations involving <span class="highlight">corporate restructuring</span>, 
            <span class="highlight">synergy realization</span>, and <span class="highlight">value creation</span> or destruction.</p>
        </div>
        """), unsafe_allow_html=True)
        
        # M&A Types
        ma_type = st.selectbox("Select M&A Type:", 
                              ["Horizontal Merger", "Vertical Merger", "Conglomerate Merger", "Acquisition"])
        
        ma_details = {
            "Horizontal Merger": {
                "definition": "Combination of competitors in same industry",
                "motives": ["Economies of scale", "Market power", "Eliminate competition", "Efficiency gains"],
                "examples": ["Exxon-Mobil", "Disney-Fox", "Dow-DuPont"],
                "risks": ["Antitrust issues", "Integration challenges", "Cultural clash"]
            },
            "Vertical Merger": {
                "definition": "Combination of firms at different production stages",
                "motives": ["Supply chain control", "Cost reduction", "Quality assurance", "Information advantages"],
                "examples": ["Amazon-Whole Foods", "Netflix-Movie Production", "Tesla-SolarCity"],
                "risks": ["Loss of supplier flexibility", "Capital intensity", "Unfamiliar operations"]
            },
            "Conglomerate Merger": {
                "definition": "Combination of unrelated businesses",
                "motives": ["Diversification", "Risk reduction", "Allocative efficiency", "Market timing"],
                "examples": ["Berkshire Hathaway", "General Electric (historical)", "3M"],
                "risks": ["Conglomerate discount", "Management complexity", "Resource misallocation"]
            },
            "Acquisition": {
                "definition": "Purchase of one firm by another",
                "motives": ["Speed to market", "Asset acquisition", "Talent acquisition", "Market entry"],
                "examples": ["Facebook-Instagram", "Google-YouTube", "Microsoft-LinkedIn"],
                "risks": ["Overpayment", "Integration failure", "Key employee departure"]
            }
        }
        
        details = ma_details[ma_type]
        
        st.markdown(textwrap.dedent(f"""
        <div class="card">
            <h3>{ma_type}</h3>
            <p><b>Definition:</b> {details['definition']}</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                <div>
                    <h4 style="color: #2ecc71;">Motives</h4>
                    <ul>{''.join([f'<li>{m}</li>' for m in details['motives']])}</ul>
                </div>
                <div>
                    <h4 style="color: #e74c3c;">Risks</h4>
                    <ul>{''.join([f'<li>{r}</li>' for r in details['risks']])}</ul>
                </div>
            </div>
            <p style="margin-top: 1rem;"><b>Notable Examples:</b> {', '.join(details['examples'])}</p>
        </div>
        """), unsafe_allow_html=True)
        
        # Synergy analysis
        st.subheader("Synergy Valuation")
        
        col1, col2 = st.columns(2)
        with col1:
            target_value = st.number_input("Target Standalone Value ($M)", value=500, step=50)
            revenue_synergy = st.number_input("Annual Revenue Synergy ($M)", value=50, step=10)
            cost_synergy = st.number_input("Annual Cost Synergy ($M)", value=30, step=5)
        with col2:
            tax_rate = st.slider("Tax Rate (%)", 0, 40, 25, key="ma_tax") / 100
            wacc = st.slider("WACC (%)", 5.0, 15.0, 10.0, key="ma_wacc") / 100
            growth = st.slider("Synergy Growth (%)", 0.0, 5.0, 2.0) / 100
        
        # Calculate synergy value
        total_synergy_annual = (revenue_synergy + cost_synergy) * (1 - tax_rate)
        synergy_value = total_synergy_annual / (wacc - growth) if wacc > growth else 0
        
        max_bid = target_value + synergy_value
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Synergy Value", f"${synergy_value:.1f}M")
        with col2:
            st.metric("Maximum Bid", f"${max_bid:.1f}M")
        with col3:
            premium = (max_bid / target_value - 1) * 100
            st.metric("Maximum Premium", f"{premium:.1f}%")
        
        # Acquisition premium analysis
        st.subheader("Historical Acquisition Premiums")
        
        premium_data = pd.DataFrame({
            'Decade': ['1980s', '1990s', '2000s', '2010s', '2020s'],
            'Average Premium (%)': [35, 42, 38, 45, 50],
            'Success Rate (%)': [60, 55, 45, 40, 35]
        })
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=premium_data['Decade'], y=premium_data['Average Premium (%)'], 
                  name="Average Premium", marker_color='#e74c3c'),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=premium_data['Decade'], y=premium_data['Success Rate (%)'], 
                      name="Success Rate", mode='lines+markers', line=dict(color='#2ecc71')),
            secondary_y=True
        )
        fig.update_layout(title="Acquisition Premiums vs. Success Rates", height=500)
        fig.update_yaxes(title_text="Premium (%)", secondary_y=False)
        fig.update_yaxes(title_text="Success Rate (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning("⚠️ Higher premiums correlate with lower success rates - the 'Winner's Curse'")
    
    with tabs[1]:
        st.subheader("Financial Distress and Bankruptcy")
        
        # Distress stages
        stage = st.select_slider("Explore Distress Stages:", 
                                options=["Economic Distress", "Financial Distress", "Default", "Bankruptcy", "Liquidation"])
        
        stage_info = {
            "Economic Distress": {
                "characteristics": ["Poor performance", "Negative cash flows", "Declining market share"],
                "indicators": ["Falling sales", "Rising costs", "Margin compression"],
                "options": ["Operational restructuring", "Cost reduction", "Strategic review"]
            },
            "Financial Distress": {
                "characteristics": ["Debt service problems", "Covenant violations", "Liquidity crisis"],
                "indicators": ["Low interest coverage", "High leverage", "Declining credit rating"],
                "options": ["Debt restructuring", "Asset sales", "Equity infusion", "Covenant waiver"]
            },
            "Default": {
                "characteristics": ["Missed payments", "Technical or absolute default", "Cross-default triggers"],
                "indicators": ["Payment delays", "Creditor notifications", "Acceleration clauses"],
                "options": ["Forbearance agreements", "Workout negotiations", "Prepackaged bankruptcy"]
            },
            "Bankruptcy": {
                "characteristics": ["Legal protection", "Automatic stay", "Court supervision"],
                "indicators": ["Chapter 11 filing", "Debtor-in-possession", "Creditor committees"],
                "options": ["Reorganization plan", "Asset sales", "Debt-equity swaps", "Fresh start"]
            },
            "Liquidation": {
                "characteristics": ["Cease operations", "Asset disposition", "Final distribution"],
                "indicators": ["Chapter 7 filing", "Trustee appointment", "Going out of business"],
                "options": ["Orderly liquidation", "Assignment for benefit of creditors", "ABC"]
            }
        }
        
        info = stage_info[stage]
        st.markdown(textwrap.dedent(f"""
        <div class="card" style="border-left-color: {'#e74c3c' if stage in ['Default', 'Bankruptcy', 'Liquidation'] else '#f39c12'};">
            <h3>{stage}</h3>
            <p><b>Characteristics:</b></p>
            <ul>{''.join([f'<li>{c}</li>' for c in info['characteristics']])}</ul>
            <p><b>Key Indicators:</b></p>
            <ul>{''.join([f'<li>{i}</li>' for i in info['indicators']])}</ul>
            <p><b>Available Options:</b></p>
            <ul>{''.join([f'<li>{o}</li>' for o in info['options']])}</ul>
        </div>
        """), unsafe_allow_html=True)
        
        # Bankruptcy comparison
        st.subheader("Chapter 11 vs. Chapter 7")
        
        bankruptcy_comp = pd.DataFrame({
            'Aspect': ['Objective', 'Management', 'Operations', 'Outcome', 'Timeline', 'Creditor Recovery'],
            'Chapter 11 (Reorganization)': ['Restructure and continue', 'Debtor-in-possession', 
                                           'Continue business', 'Emergence as going concern', 
                                           'Months to years', 'Often higher (ongoing value)'],
            'Chapter 7 (Liquidation)': ['Sell assets and distribute', 'Trustee appointed', 
                                       'Cease operations', 'Dissolution', 
                                       'Months', 'Often lower (fire sale)']
        })
        
        st.dataframe(bankruptcy_comp, use_container_width=True, hide_index=True)
        
        # Absolute priority rule
        st.subheader("Absolute Priority Rule (APR)")
        
        priority = [
            ("1. Secured Creditors", "Collateral value", "#2ecc71"),
            ("2. Administrative Expenses", "Bankruptcy costs", "#27ae60"),
            ("3. Priority Claims", "Wages, taxes", "#f39c12"),
            ("4. General Unsecured", "Trade creditors, bonds", "#e67e22"),
            ("5. Subordinated Debt", "Junior creditors", "#e74c3c"),
            ("6. Preferred Stock", "Preferred shareholders", "#c0392b"),
            ("7. Common Stock", "Residual claimants", "#8e44ad")
        ]
        
        for claim, desc, color in priority:
            st.markdown(textwrap.dedent(f"""
            <div style="background: {color}; color: white; padding: 0.8rem; margin: 0.2rem 0; 
                        border-radius: 5px; display: flex; justify-content: space-between;">
                <span><b>{claim}</b></span>
                <span>{desc}</span>
            </div>
            """), unsafe_allow_html=True)
        
        st.info("Note: APR is often violated in practice to expedite reorganization and gain creditor approval.")
    
    with tabs[2]:
        st.subheader("Altman Z-Score Predictor")
        
        st.markdown(textwrap.dedent("""
        <div class="formula-box">
            Z = 1.2X₁ + 1.4X₂ + 3.3X₃ + 0.6X₄ + 1.0X₅
            <br><br>
            X₁ = Working Capital / Total Assets<br>
            X₂ = Retained Earnings / Total Assets<br>
            X₃ = EBIT / Total Assets<br>
            X₄ = Market Value of Equity / Book Value of Total Liabilities<br>
            X₅ = Sales / Total Assets
        </div>
        """), unsafe_allow_html=True)
        
        # Input data
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Financial Data (in millions):**")
            current_assets = st.number_input("Current Assets", value=500.0, step=10.0)
            current_liabilities = st.number_input("Current Liabilities", value=300.0, step=10.0)
            total_assets = st.number_input("Total Assets", value=1000.0, step=10.0)
            retained_earnings = st.number_input("Retained Earnings", value=200.0, step=10.0)
        with col2:
            ebit = st.number_input("EBIT", value=150.0, step=10.0)
            market_cap = st.number_input("Market Cap (Equity)", value=800.0, step=10.0)
            total_liabilities = st.number_input("Total Liabilities", value=600.0, step=10.0)
            sales = st.number_input("Sales", value=1200.0, step=10.0)
        
        # Calculate ratios
        x1 = (current_assets - current_liabilities) / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_cap / total_liabilities
        x5 = sales / total_assets
        
        z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
        
        # Display ratios
        st.subheader("Component Ratios")
        ratio_cols = st.columns(5)
        ratios = [("X₁ (WC/TA)", x1), ("X₂ (RE/TA)", x2), ("X₃ (EBIT/TA)", x3), 
                 ("X₄ (MVE/TL)", x4), ("X₅ (Sales/TA)", x5)]
        
        for col, (name, value) in zip(ratio_cols, ratios):
            col.metric(name, f"{value:.3f}")
        
        # Z-Score result
        st.markdown("---")
        
        zone, color, interpretation = get_z_interpretation(z_score)
        
        result_html = f"""
        <div style="background: {'#d4edda' if color == 'green' else '#fff3cd' if color == 'orange' else '#f8d7da'}; 
                    padding: 2rem; border-radius: 15px; text-align: center; border: 3px solid {color};">
            <h1 style="font-size: 4rem; margin: 0; color: {color};">{z_score:.2f}</h1>
            <h2 style="color: {color};">{zone}</h2>
            <p style="font-size: 1.2rem;">{interpretation}</p>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)
        
        # Historical accuracy
        st.subheader("Model Accuracy")
        accuracy_data = pd.DataFrame({
            'Zone': ['Safe (Z > 2.99)', 'Grey (1.81-2.99)', 'Distress (Z < 1.81)'],
            '1-Year Accuracy': ['95%', '70%', '85%'],
            '2-Year Accuracy': ['85%', '60%', '75%']
        })
        st.dataframe(accuracy_data, use_container_width=True, hide_index=True)
        
        # Comparison with other models
        st.subheader("Compare with Other Models")
        
        # Ohlson O-Score (simplified)
        total_liabilities_ohlson = total_liabilities / total_assets
        net_income_ohlson = ebit * 0.7 / total_assets  # Approximate
        o_score = -1.32 - 0.407*np.log(total_assets) + 6.03*total_liabilities_ohlson - 1.43*x1
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Altman Z-Score", f"{z_score:.2f}", zone)
        with col2:
            o_zone = "High Risk" if o_score > 0.5 else "Low Risk"
            st.metric("Ohlson O-Score", f"{o_score:.2f}", o_zone)
    
    with tabs[3]:
        st.subheader("M&A and Distress Calculators")
        
        calc = st.radio("Select Calculator:", ["Merger Valuation", "Distress Cost Estimator", "Recovery Analysis"])
        
        if calc == "Merger Valuation":
            col1, col2 = st.columns(2)
            with col1:
                acquirer_shares = st.number_input("Acquirer Shares Outstanding (M)", value=100.0)
                acquirer_price = st.number_input("Acquirer Share Price ($)", value=50.0)
                target_shares = st.number_input("Target Shares Outstanding (M)", value=50.0)
                target_price = st.number_input("Target Share Price ($)", value=40.0)
            with col2:
                offer_premium = st.slider("Offer Premium (%)", 0, 100, 30) / 100
                synergy_value = st.number_input("Present Value of Synergies ($M)", value=500.0)
                stock_fraction = st.slider("Stock in Consideration (%)", 0, 100, 50) / 100
            
            # Calculations
            offer_price = target_price * (1 + offer_premium)
            total_consideration = offer_price * target_shares
            stock_component = total_consideration * stock_fraction / acquirer_price
            cash_component = total_consideration * (1 - stock_fraction)
            
            value_created = synergy_value - (offer_price - target_price) * target_shares
            
            st.metric("Offer Price per Share", f"${offer_price:.2f}")
            st.metric("Total Consideration", f"${total_consideration:.1f}M")
            st.metric("New Shares Issued", f"{stock_component:.1f}M")
            st.metric("Value Created (Destroyed)", f"${value_created:.1f}M", 
                     "Accretive" if value_created > 0 else "Dilutive")
        
        elif calc == "Distress Cost Estimator":
            col1, col2 = st.columns(2)
            with col1:
                firm_value = st.number_input("Current Firm Value ($M)", value=1000)
                debt_ratio = st.slider("Debt Ratio", 0.0, 1.0, 0.6)
                volatility = st.slider("Asset Volatility (%)", 10, 100, 40) / 100
            with col2:
                risk_free = st.slider("Risk-free Rate (%)", 1, 10, 3) / 100
                
                # Simplified Merton model for distance to default
                asset_value = firm_value
                debt_face = firm_value * debt_ratio
                distance_to_default = (np.log(asset_value/debt_face) + (risk_free + 0.5*volatility**2)) / volatility
            
            st.metric("Distance to Default", f"{distance_to_default:.2f}")
            
            if distance_to_default < 1:
                st.error("High probability of default within 1 year")
            elif distance_to_default < 2:
                st.warning("Moderate default risk")
            else:
                st.success("Low default risk")
            
            # Estimated default probability
            default_prob = norm.cdf(-distance_to_default)
            st.metric("Estimated Default Probability", f"{default_prob*100:.1f}%")
        
        else:  # Recovery Analysis
            col1, col2 = st.columns(2)
            with col1:
                total_claims = st.number_input("Total Claims ($M)", value=1000)
                asset_recovery = st.slider("Asset Recovery Rate (%)", 0, 100, 60) / 100
            with col2:
                priority_claims = st.number_input("Priority Claims ($M)", value=200)
                secured_claims = st.number_input("Secured Claims ($M)", value=300)
                unsecured_claims = st.number_input("Unsecured Claims ($M)", value=400)
                equity_claims = st.number_input("Equity Claims ($M)", value=100)
            
            total_recovery = total_claims * asset_recovery
            
            # Waterfall
            priority_recovery = min(priority_claims, total_recovery)
            remaining = total_recovery - priority_recovery
            
            secured_recovery = min(secured_claims, remaining)
            remaining -= secured_recovery
            
            unsecured_recovery = min(unsecured_claims, remaining)
            remaining -= unsecured_recovery
            
            equity_recovery = max(0, remaining)
            
            recoveries = {
                'Priority': (priority_recovery/priority_claims*100 if priority_claims > 0 else 0),
                'Secured': (secured_recovery/secured_claims*100 if secured_claims > 0 else 0),
                'Unsecured': (unsecured_recovery/unsecured_claims*100 if unsecured_claims > 0 else 0),
                'Equity': (equity_recovery/equity_claims*100 if equity_claims > 0 else 0)
            }
            
            for claim_type, recovery_pct in recoveries.items():
                st.progress(min(recovery_pct/100, 1.0))
                st.write(f"{claim_type}: {recovery_pct:.1f}% recovery")
    
    with tabs[4]:
        st.subheader("Unit 8 Assessment")
        
        questions = [
            {
                "question": "The primary motive for horizontal mergers is:",
                "options": ["Supply chain control", "Economies of scale",
                           "Diversification", "Market entry"],
                "correct": 1,
                "explanation": "Horizontal mergers combine competitors, primarily seeking economies of scale and market power."
            },
            {
                "question": "A Z-Score below 1.81 indicates:",
                "options": ["Safe financial condition", "Moderate risk",
                           "High bankruptcy risk", "Strong performance"],
                "correct": 2,
                "explanation": "Z < 1.81 places a firm in the distress zone with high probability of bankruptcy."
            },
            {
                "question": "In Chapter 11 bankruptcy:",
                "options": ["Operations cease immediately", "Management is always replaced",
                           "The firm attempts to reorganize", "Assets are liquidated"],
                "correct": 2,
                "explanation": "Chapter 11 is reorganization bankruptcy where the firm attempts to restructure and continue operations."
            }
        ]
        
        score = 0
        for idx, q in enumerate(questions):
            st.markdown(f"**Q{idx+1}. {q['question']}**")
            answer = st.radio(f"Select answer for Q{idx+1}:", q['options'], key=f"q8_{idx}")
            
            if st.button(f"Check Answer {idx+1}", key=f"check8_{idx}"):
                if q['options'].index(answer) == q['correct']:
                    st.success(f"✅ Correct! {q['explanation']}")
                    score += 1
                else:
                    st.error(f"❌ Incorrect. {q['explanation']}")
        
        if st.button("Submit Quiz", key="submit8"):
            st.session_state.quiz_scores['Unit 8'] = score
            st.balloons()
            st.success(f"Quiz submitted! Score: {score}/{len(questions)}")

# Financial Simulator
elif page == "🎮 Financial Simulator":
    st.title("🎮 Financial Management Simulator")
    
    st.markdown(textwrap.dedent("""
    <div class="main-header">
        <h2>Corporate Finance Simulation</h2>
        <p>Make strategic decisions and see their impact on firm value over 5 years</p>
    </div>
    """), unsafe_allow_html=True)
    
    # Initialize simulation state
    if 'sim_year' not in st.session_state:
        st.session_state.sim_year = 1
        st.session_state.firm_value = 1000
        st.session_state.sales = 500
        st.session_state.profit_margin = 0.10
        st.session_state.debt_ratio = 0.40
        st.session_state.dividend_payout = 0.30
        st.session_state.history = [{'Year': 0, 'Value': 1000, 'Sales': 500, 'EPS': 2.00}]
    
    # Display current state
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Year", st.session_state.sim_year)
    with col2:
        st.metric("Firm Value", f"${st.session_state.firm_value:.0f}M")
    with col3:
        eps = (st.session_state.sales * st.session_state.profit_margin * 
               (1 - st.session_state.dividend_payout)) / 50  # Simplified
        st.metric("EPS", f"${eps:.2f}")
    with col4:
        st.metric("Stock Price", f"${st.session_state.firm_value/50:.2f}")
    
    # Historical chart
    if len(st.session_state.history) > 1:
        hist_df = pd.DataFrame(st.session_state.history)
        fig = px.line(hist_df, x='Year', y='Value', title='Firm Value History',
                     labels={'Value': 'Firm Value ($M)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Decision panel
    st.subheader("Strategic Decisions")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Investment Decision**")
        capex = st.slider("Capital Expenditure ($M)", 0, 200, 50, key="sim_capex")
        rd_spend = st.slider("R&D Spending ($M)", 0, 100, 20, key="sim_rd")
        
        st.write("**Financing Decision**")
        new_debt = st.slider("New Debt Issuance ($M)", 0, 100, 20, key="sim_debt")
        stock_buyback = st.slider("Stock Buyback ($M)", 0, 50, 0, key="sim_buyback")
    
    with col2:
        st.write("**Operations**")
        price_change = st.slider("Price Change (%)", -10, 10, 0, key="sim_price") / 100
        cost_reduction = st.slider("Cost Reduction Program (%)", 0, 20, 5, key="sim_cost") / 100
        
        st.write("**Dividend Policy**")
        dividend_change = st.select_slider("Dividend Change", 
                                          options=["Cut 50%", "Cut 25%", "Maintain", "Increase 10%", "Increase 25%"])
    
    # Economic scenario
    st.subheader("Economic Environment")
    economy = st.select_slider("Economic Scenario", 
                              options=["Recession", "Slow Growth", "Normal", "Strong Growth", "Boom"])
    
    economic_factors = {
        "Recession": {"growth": -0.05, "risk_premium": 0.08},
        "Slow Growth": {"growth": 0.02, "risk_premium": 0.06},
        "Normal": {"growth": 0.05, "risk_premium": 0.05},
        "Strong Growth": {"growth": 0.08, "risk_premium": 0.04},
        "Boom": {"growth": 0.12, "risk_premium": 0.03}
    }
    
    if st.button("Advance One Year", key="advance_year"):
        factor = economic_factors[economy]
        
        # Calculate impacts
        sales_growth = factor["growth"] + (price_change * 0.5) + (rd_spend / 100)
        new_sales = st.session_state.sales * (1 + sales_growth)
        
        margin_improvement = cost_reduction * 0.02
        new_margin = min(st.session_state.profit_margin + margin_improvement, 0.25)
        
        # Value impact
        investment_value = (capex * 0.15 + rd_spend * 0.25)  # NPV of investments
        financing_impact = new_debt * 0.10 - stock_buyback * 0.05  # Leverage and signaling
        dividend_signal = {"Cut 50%": -0.10, "Cut 25%": -0.05, "Maintain": 0, 
                          "Increase 10%": 0.05, "Increase 25%": 0.10}[dividend_change]
        
        value_change = (investment_value + financing_impact + dividend_signal + 
                       new_sales * new_margin * 0.5)
        
        # Update state
        st.session_state.sim_year += 1
        st.session_state.sales = new_sales
        st.session_state.profit_margin = new_margin
        st.session_state.firm_value = max(100, st.session_state.firm_value + value_change)
        
        st.session_state.history.append({
            'Year': st.session_state.sim_year - 1,
            'Value': st.session_state.firm_value,
            'Sales': new_sales,
            'EPS': new_sales * new_margin * (1 - st.session_state.dividend_payout) / 50
        })
        
        st.rerun()
    
    if st.button("Reset Simulation"):
        del st.session_state.sim_year
        del st.session_state.firm_value
        del st.session_state.history
        st.rerun()

# Market Data Lab
elif page == "📉 Market Data Lab":
    st.title("📉 Market Data Laboratory")
    
    st.markdown(textwrap.dedent("""
    <div class="card">
        <h3>Real-Time Financial Analysis</h3>
        <p>Analyze actual market data to apply financial management concepts.</p>
    </div>
    """), unsafe_allow_html=True)
    
    # Stock analysis
    ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
    
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
            with col2:
                st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B")
            with col3:
                st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            with col4:
                st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
            
            # Historical data
            period = st.selectbox("Select Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
            hist = stock.history(period=period)
            
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Price'
            ))
            fig.update_layout(title=f"{ticker} Stock Price", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate returns and volatility
            hist['Returns'] = hist['Close'].pct_change()
            volatility = hist['Returns'].std() * np.sqrt(252) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Annualized Volatility", f"{volatility:.1f}%")
            with col2:
                beta = info.get('beta', 'N/A')
                st.metric("Beta", f"{beta:.2f}" if isinstance(beta, (int, float)) else beta)
            
            # Capital structure analysis
            st.subheader("Capital Structure Analysis")
            
            try:
                balance_sheet = stock.balance_sheet
                if not balance_sheet.empty:
                    latest = balance_sheet.columns[0]
                    total_debt = balance_sheet.loc['Total Debt', latest] if 'Total Debt' in balance_sheet.index else 0
                    total_equity = balance_sheet.loc['Stockholders Equity', latest] if 'Stockholders Equity' in balance_sheet.index else 1
                    
                    debt_ratio = total_debt / (total_debt + total_equity)
                    
                    # Structure visualization
                    fig = go.Figure(go.Pie(
                        labels=['Debt', 'Equity'],
                        values=[total_debt, total_equity],
                        hole=0.4,
                        marker_colors=['#e74c3c', '#3498db']
                    ))
                    fig.update_layout(title="Capital Structure", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Balance sheet data not available")
                
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")

# Advanced Calculators
elif page == "🧮 Advanced Calculators":
    st.title("🧮 Advanced Financial Calculators")
    
    calc_category = st.selectbox("Select Calculator Category:", 
                                ["Time Value of Money", "Bond Analysis", "Option Strategies", "WACC Calculator"])
    
    if calc_category == "Time Value of Money":
        st.subheader("Comprehensive TVM Calculator")
        
        solve_for = st.radio("Solve for:", ["Future Value", "Present Value", "Payment", "Rate", "Periods"])
        
        col1, col2 = st.columns(2)
        with col1:
            if solve_for != "Present Value":
                pv = st.number_input("Present Value ($)", value=1000.0)
            if solve_for != "Future Value":
                fv = st.number_input("Future Value ($)", value=2000.0)
            if solve_for != "Payment":
                pmt = st.number_input("Periodic Payment ($)", value=0.0)
        with col2:
            if solve_for != "Rate":
                rate = st.slider("Interest Rate (%)", 0.0, 20.0, 5.0) / 100
            if solve_for != "Periods":
                nper = st.slider("Number of Periods", 1, 50, 10)
            periods_per_year = st.selectbox("Periods per Year", [1, 2, 4, 12, 365], index=3)
        
        # Calculate based on selection
        r_per_period = rate / periods_per_year if solve_for != "Rate" else None
        n_total = nper * periods_per_year if solve_for != "Periods" else None
        
        if solve_for == "Future Value":
            fv_calc = pv * (1 + r_per_period) ** n_total + pmt * (((1 + r_per_period) ** n_total - 1) / r_per_period)
            st.metric("Future Value", f"${fv_calc:,.2f}")
        elif solve_for == "Present Value":
            pv_calc = fv / (1 + r_per_period) ** n_total
            st.metric("Present Value", f"${pv_calc:,.2f}")
        elif solve_for == "Payment":
            if r_per_period > 0:
                pmt_calc = (fv - pv * (1 + r_per_period) ** n_total) / (((1 + r_per_period) ** n_total - 1) / r_per_period)
            else:
                pmt_calc = (fv - pv) / n_total
            st.metric("Periodic Payment", f"${pmt_calc:,.2f}")
    
    elif calc_category == "Bond Analysis":
        st.subheader("Bond Valuation & Yield Calculator")
        
        col1, col2 = st.columns(2)
        with col1:
            face_value = st.number_input("Face Value ($)", value=1000.0)
            coupon_rate = st.slider("Coupon Rate (%)", 0.0, 15.0, 5.0) / 100
            years_to_maturity = st.slider("Years to Maturity", 1, 30, 10)
        with col2:
            market_rate = st.slider("Market Interest Rate (%)", 0.0, 15.0, 6.0) / 100
            payments_per_year = st.selectbox("Payments per Year", [1, 2, 4, 12], index=1)
        
        # Bond calculations
        nper = years_to_maturity * payments_per_year
        pmt = face_value * coupon_rate / payments_per_year
        rate_per_period = market_rate / payments_per_year
        
        pv = (pmt * (1 - (1 + rate_per_period) ** (-nper)) / rate_per_period + 
              face_value / (1 + rate_per_period) ** nper)
        
        # Duration calculation (simplified)
        durations = []
        for t in range(1, nper + 1):
            cf = pmt if t < nper else pmt + face_value
            pv_cf = cf / (1 + rate_per_period) ** t
            durations.append(t * pv_cf / pv / payments_per_year)
        
        duration = sum(durations)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bond Price", f"${pv:,.2f}")
        with col2:
            st.metric("Duration", f"{duration:.2f} years")
        with col3:
            premium_discount = "Premium" if pv > face_value else "Discount" if pv < face_value else "Par"
            st.metric("Status", premium_discount)
    
    elif calc_category == "Option Strategies":
        st.subheader("Option Strategy Payoff Analyzer")
        
        strategy = st.selectbox("Select Strategy:", 
                               ["Covered Call", "Protective Put", "Bull Call Spread", "Straddle", "Iron Condor"])
        
        col1, col2 = st.columns(2)
        with col1:
            S = st.number_input("Current Stock Price ($)", value=100.0)
            K1 = st.number_input("Strike Price 1 ($)", value=100.0)
            K2 = st.number_input("Strike Price 2 ($)", value=110.0) if "Spread" in strategy or "Condor" in strategy else 0
        with col2:
            premium1 = st.number_input("Premium 1 ($)", value=5.0)
            premium2 = st.number_input("Premium 2 ($)", value=3.0) if "Spread" in strategy or "Condor" in strategy else 0
        
        # Generate payoff diagram
        prices = np.linspace(S * 0.5, S * 1.5, 100)
        
        if strategy == "Covered Call":
            payoffs = np.minimum(prices, K1) - S + premium1
        elif strategy == "Protective Put":
            payoffs = np.maximum(prices, K1) - S - premium1
        elif strategy == "Bull Call Spread":
            payoffs = np.maximum(np.minimum(prices, K2) - K1, 0) - premium1 + premium2
        elif strategy == "Straddle":
            payoffs = np.abs(prices - K1) - 2 * premium1
        else:  # Iron Condor (simplified)
            payoffs = np.where((prices > K1) & (prices < K2), premium1 + premium2, 
                             np.minimum(np.abs(prices - K1), np.abs(prices - K2)) - premium1 - premium2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=payoffs, mode='lines', 
                                fill='tozeroy', name='Payoff'))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(title=f"{strategy} Payoff", xaxis_title="Stock Price", 
                         yaxis_title="Profit/Loss ($)", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # WACC Calculator
        st.subheader("Weighted Average Cost of Capital")
        
        col1, col2 = st.columns(2)
        with col1:
            # Equity
            risk_free = st.slider("Risk-free Rate (%)", 0.0, 10.0, 3.0) / 100
            market_return = st.slider("Expected Market Return (%)", 5.0, 20.0, 10.0) / 100
            beta = st.slider("Beta", 0.0, 3.0, 1.2)
            equity_value = st.number_input("Market Value of Equity ($M)", value=600.0)
        with col2:
            # Debt
            debt_value = st.number_input("Market Value of Debt ($M)", value=400.0)
            interest_rate = st.slider("Interest Rate on Debt (%)", 0.0, 15.0, 6.0) / 100
            tax_rate = st.slider("Corporate Tax Rate (%)", 0.0, 40.0, 25.0) / 100
        
        # Calculate WACC
        cost_equity = risk_free + beta * (market_return - risk_free)
        cost_debt = interest_rate * (1 - tax_rate)
        
        total_value = equity_value + debt_value
        wacc = (equity_value / total_value) * cost_equity + (debt_value / total_value) * cost_debt
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cost of Equity", f"{cost_equity*100:.2f}%")
        with col2:
            st.metric("After-tax Cost of Debt", f"{cost_debt*100:.2f}%")
        with col3:
            st.metric("WACC", f"{wacc*100:.2f}%")
        
        # Capital structure optimization
        st.subheader("Capital Structure Optimization")
        
        debt_ratios = np.linspace(0, 0.8, 50)
        waccs = []
        
        for d_ratio in debt_ratios:
            # Simplified: cost of equity increases with leverage
            levered_beta = beta * (1 + (1 - tax_rate) * d_ratio / (1 - d_ratio))
            re = risk_free + levered_beta * (market_return - risk_free)
            rd = interest_rate * (1 + d_ratio * 0.5)  # Cost of debt increases with leverage
            waccs.append((1 - d_ratio) * re + d_ratio * rd * (1 - tax_rate))
        
        optimal_idx = np.argmin(waccs)
        
        fig = px.line(x=debt_ratios*100, y=np.array(waccs)*100, 
                     labels={'x': 'Debt Ratio (%)', 'y': 'WACC (%)'},
                     title="WACC vs. Capital Structure")
        fig.add_vline(x=debt_ratios[optimal_idx]*100, line_dash="dash", 
                     annotation_text=f"Optimal: {debt_ratios[optimal_idx]*100:.0f}%")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(textwrap.dedent("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>FIN 231: Financial Management Interactive Learning Platform</p>
    <p style="font-size: 0.8rem;">Built with Streamlit • Designed for Educational Excellence</p>
</div>
"""), unsafe_allow_html=True)
