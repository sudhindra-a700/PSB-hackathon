# SecureWealth Twin

**Intelligent Wealth Growth with Built-in Blockchain & AI Fraud Protection**

Project for **PSBs Hackathon Series 2026**

---

## 📋 Project Overview

SecureWealth Twin is a **"Brain & Shield"** system that helps customers grow their wealth intelligently while protecting them from fraud (social engineering, OTP misuse, coerced transactions, long-term schemes, etc.).

- **Brain** → Personalized wealth intelligence (goal-based advice, scenario simulation, net-worth view)
- **Shield** → Mandatory, tamper-proof fraud protection layer (real-time + long-term detection)

The system follows the official hackathon problem statement and the original project documentation.

---

## ✨ Current Progress (as of April 20, 2026)

| Component                  | Status       | Details |
|---------------------------|--------------|-------|
| Risk Engine (Shield)      | ✅ Complete | PerpetualBooster model trained on team's real mock data |
| Real Data Integration     | ✅ Complete | Uses `SecureWealth_Twin_MockData.xlsx` (User Profiles, Transactions, Net Worth) |
| KYC Check                 | ✅ Complete | Mandatory verification before any critical action |
| New Device Verification   | ✅ Complete | Blocks or flags unverified devices |
| Long-term Fraud Detection | ✅ Complete | Churn rate, behavioral drift, micro-transfers, cumulative risk, return consistency |
| Layman-Friendly Explanations | ✅ Complete | Simple, human-readable reasons (no technical jargon) |
| Blockchain Ready          | ✅ Contracts Ready | AuditTrail + Timelock contracts prepared |
| FastAPI Backend           | ✅ Ready     | `/risk-score` and `/wealth-advice` endpoints with Shield protection |
| Audit Logging             | ✅ Ready     | Every decision can be logged |

The core **Shield layer** is fully functional and production-grade for a hackathon prototype.

---

## 🛠 Tech Stack

- **Risk Engine**: PerpetualBooster (gradient boosting) + SHAP-style contributions
- **Data**: Real mock dataset (`SecureWealth_Twin_MockData.xlsx`) from team
- **Backend**: FastAPI + Python 3
- **Blockchain**: Solidity smart contracts (AuditTrail + Timelock) on Polygon Amoy testnet
- **Frontend (planned)**: React / Flutter
- **Explainability**: Middle-ground reasoning system (external `explanation_rules.json`)

---

## 🚀 How to Run (Current Version)

1. Place `SecureWealth_Twin_MockData.xlsx` in the project root
2. Install dependencies:
   ```bash
   pip install perpetual pandas numpy joblib
