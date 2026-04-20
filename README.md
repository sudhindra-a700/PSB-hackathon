# SecureWealth Twin

**Intelligent Wealth Growth with Built-in Blockchain & AI Fraud Protection**

Project for **PSBs Hackathon Series 2026**

---

## 📋 Project Overview

SecureWealth Twin is a **"Brain & Shield"** architecture that helps customers grow their wealth intelligently while protecting them from both instant and long-term fraud (social engineering, OTP misuse, micro-siphoning, churning, gradual account draining, etc.).

- **Brain** → Personalized wealth intelligence, goal-based advisory, scenario simulation
- **Shield** → Mandatory, explainable fraud protection layer (real-time + longitudinal detection)

Fully aligned with the official PSBs Hackathon problem statement and the original project PDF.

---

## 📊 Current Progress (as of April 20, 2026)

| Component                        | Status          | Details |
|----------------------------------|-----------------|-------|
| Risk Engine (PerpetualBooster)   | ✅ **Completed** | Trained on team's real mock data (`SecureWealth_Twin_MockData.xlsx`) |
| Real Data Integration            | ✅ **Completed** | Loads User Profiles, Transactions, Net Worth from Excel |
| KYC Check                        | ✅ **Completed** | Mandatory verification before critical actions |
| New Device Verification          | ✅ **Completed** | Detailed check (verified / OTP / not verified) |
| Long-term Fraud Detection        | ✅ **Completed** | Churn rate, behavioral drift, micro-transfers, cumulative risk, return consistency |
| Layman-Friendly Explanations     | ✅ **Completed** | Simple, easy-to-understand reasons (middle-ground reasoning system) |
| FastAPI Backend                  | ❌ Not Started   | Planned next |
| Blockchain Integration           | ❌ Not Started   | AuditTrail + Timelock contracts ready, integration pending |
| Audit Logging                    | ❌ Not Started   | JSONL + blockchain logging planned |

**Core Shield layer is fully functional and ready for demonstration.**

---

## 🛠 Tech Stack (Current)

- **Risk Engine**: PerpetualBooster (gradient boosting) with native Shapley contributions
- **Data**: Team’s real mock dataset (`SecureWealth_Twin_MockData.xlsx`)
- **Explainability**: Lightweight middle-ground reasoning system (`explanation_rules.json`)
- **Future Option**: Ollama (local LLM) can be added for even more natural explanations

---

## 🚀 How to Run (Current Version)

1. Place `SecureWealth_Twin_MockData.xlsx` in the project root
2. Install dependencies:
   ```bash
   pip install perpetual pandas numpy joblib

## Future Additions & Completion Plan
**Immediate Next Steps (Next 3–4 days)**

- Implement FastAPI backend (main.py) with /risk-score and /wealth-advice endpoints
- Add Blockchain Integration (AuditTrail + Timelock contracts on Polygon Amoy testnet using Web3.py)
- Implement Audit Logging (every decision saved to audit_log.jsonl + immutable blockchain record)
- React/Flutter dashboard with clean UI, goal progress charts, and portfolio health report

**Short-term Enhancements (Before Final Submission)**

- Ollama Integration (optional): Replace or augment the current explanation system with a local LLM (llama3.2:1b or phi3) for more natural, conversational explanations
- Real Account Aggregator sandbox integration (Setu / FinVu / FinVu)
- `/portfolio-health` endpoint for monthly summaries and drift alerts
