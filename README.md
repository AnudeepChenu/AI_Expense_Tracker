# 💰 AI Expense Analyzer (Streamlit + Gemini + ML)

The AI Expense Analyzer helps you understand your spending habits using charts, machine learning prediction, and AI-generated insights.

---

## ⭐ Features

- Upload and analyze CSV expense history
- Automatic cleaning and formatting
- Visual breakdown by month and category
- Machine learning forecasting using Linear Regression
- AI-powered natural language spending summary (Gemini API)
- Works on Windows + macOS

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
Open project folder in VSCODE
cd AI-Expense-Analyzer
```

### 2️⃣ Install Dependencies


```powershell
pip install pandas streamlit numpy requests json scikit-learn
```

---

## 🔑 Setup Google API Key

1. Go to Google AI Studio → Create a Gemini API key.
2. Open `app.py`
3. Replace placeholder:

```python
API_KEY = "YOUR_API_KEY_HERE"
```

⚠️ Do **not** commit your real API key to GitHub.

---

## 🚀 Run the App

```powershell
streamlit run app.py
```

The browser will open automatically.

---

## 📁 CSV Format Requirements

| Column Name      | Example |
|------------------|--------|
| Date             | 2025-01-14 |
| Category         | Food |
| Amount (INR)     | 450 |
| Description      | Domino’s |

> The app auto-detects similar column variations like: `amount, spent, price`.

---

## ❗ Troubleshooting

| Problem | Solution |
|--------|----------|
| App doesn't start | Ensure Python ≥ 3.9 |
| Charts empty | Ensure numeric values in `Amount` column |
| AI summary blank | Verify Gemini API key |

---


