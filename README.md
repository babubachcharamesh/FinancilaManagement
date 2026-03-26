# 🚀 FIN 231: Financial Management Mastery

A comprehensive, high-fidelity interactive learning platform for Financial Management. Built with Streamlit, this application provides a deep dive into core financial concepts with stunning visualizations, simulators, and calculators.

## 🌟 Key Features

- **8 Comprehensive Units**: Covering Capital Structure, Dividend Policy, Raising Capital, Working Capital, Derivatives, and more.
- **🎮 Financial Simulator**: Interactive year-by-year financial simulation with state management.
- **📊 Market Data Lab**: Real-time stock data fetching and analysis using `yfinance`.
- **🧮 Advanced Calculators**: 
  - Black-Scholes Option Pricing
  - Lease vs. Buy NPV Analysis
  - Convertible Bond & Warrant Valuation
- **🎯 Interactive Quizzes**: Unit-specific quizzes to test knowledge retention.
- **🎨 Premium UI/UX**: Custom CSS with support for both Light and Dark modes.

## 🛠️ Local Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd FinancialManagement
   ```

2. **Set up a virtual environment** (recommended with `uv`):
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run main.py
   ```

## 🌐 Deployment (Streamlit Cloud)

1. Push your code to a GitHub repository.
2. Connect your GitHub account to [Streamlit Cloud](https://share.streamlit.io/).
3. Select your repository, branch, and `main.py` as the entry point.
4. Streamlit will automatically detect `requirements.txt` and install all necessary packages.

## 📝 License

This project is for educational purposes. All rights reserved.
