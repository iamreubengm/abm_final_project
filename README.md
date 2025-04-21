# 💰 AI Personal Financial Portal

An intelligent, interactive financial planning app powered by Anthropic Claude and Streamlit. This tool helps users manage their finances, analyze spending habits, and receive AI-generated advice across budgeting, savings, investments, debt, and taxes.

---

## 🚀 Features

- 🔍 **Personalized Financial Dashboard**  
  View your net worth, income, expenses, and savings rate at a glance.

- 🤖 **AI Financial Advisor**  
  Get holistic insights and strategic advice powered by Claude (Anthropic).

- 📊 **Budget Analysis & Planning**  
  Understand your spending and create optimized budget plans.

- 📈 **Investment & Retirement Guidance**  
  Receive tailored investment strategies and future planning insights.

- 💳 **Debt Management Plans**  
  AI-generated repayment strategies using debt avalanche/snowball methods.

- 💡 **Savings Goal Planner**  
  Set and track savings goals with customized monthly contributions.

- 📚 **RAG-powered Explanations**  
  Retrieval-Augmented Generation for rich financial education.

## ⚙️ Setup Instructions (Windows)

## 1. Clone the repository
git clone [https://github.com/your-username/financial-portal.git](https://github.com/iamreubengm/abm_final_project)
cd abm_final_project

## 2. Set up virtual environment
python -m venv venv
.\venv\Scripts\activate

## 3. Install dependencies
pip install -r requirements.txt

## 4. Create .env file and add your Claude API key
echo ANTHROPIC_API_KEY=your_claude_api_key > .env

## 5. Run the Streamlit app
streamlit run app.py
