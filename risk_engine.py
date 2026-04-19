# enhanced_risk_engine_perpetual_xai_fixed.py
# =====================================================
# SecureWealth Twin – Risk Engine (PerpetualBooster ONLY)
# Fixed: KeyError None + proper model + feature_names save/load
# Native SHAPLEY XAI via predict_contributions()
# =====================================================

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from perpetual import PerpetualBooster
except ImportError:
    raise ImportError("Run: pip install perpetual")


class EnhancedRiskEngine:
    def __init__(self, model_path: str = "risk_model_perpetual.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        self.load_or_train_model()

    def _simulate_account_aggregator_data(self, n_samples: int = 20000):
        np.random.seed(42)
        data = {
            "device_trust": np.random.choice([0, 30], n_samples, p=[0.85, 0.15]),
            "velocity_seconds": np.random.randint(5, 3600, n_samples),
            "amount": np.random.lognormal(8, 1.2, n_samples).astype(int),
            "historical_avg_amount": np.random.lognormal(7.5, 1.0, n_samples).astype(int),
            "retries": np.random.poisson(0.8, n_samples).clip(0, 5),
            "income_stability": np.random.beta(8, 2, n_samples),
            "balance_to_amount_ratio": np.random.uniform(0.5, 50, n_samples),
            "transaction_frequency_24h": np.random.poisson(3, n_samples),
            "unusual_category_spend": np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),
            "goal_deviation": np.random.uniform(-1, 1, n_samples),
            "new_account_linked": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            "large_withdrawal_from_goal": np.random.choice([0, 1], n_samples, p=[0.96, 0.04]),
            "spending_spike_ratio": np.random.lognormal(0, 0.6, n_samples),
            "net_worth": np.random.lognormal(12, 1.5, n_samples).astype(int),
            "loan_outstanding_ratio": np.random.uniform(0, 0.8, n_samples),
            "investment_volatility": np.random.uniform(0, 1, n_samples),
            "time_since_last_login_hours": np.random.uniform(0.1, 720, n_samples),
            "multiple_banks_linked": np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.25, 0.1, 0.05]),
        }
        df = pd.DataFrame(data)
        df["amount_vs_history"] = df["amount"] / (df["historical_avg_amount"] + 1e-8)
        df["velocity_vs_income"] = df["velocity_seconds"] / (df["income_stability"] + 0.01)

        df["is_fraud"] = 0
        fraud_mask = (
            ((df["device_trust"] == 30) & (df["amount_vs_history"] > 3) & (df["velocity_seconds"] < 60)) |
            (df["retries"] >= 3) |
            (df["large_withdrawal_from_goal"] == 1) |
            ((df["new_account_linked"] == 1) & (df["amount"] > 500000))
        )
        df.loc[fraud_mask, "is_fraud"] = 1
        df.loc[np.random.rand(n_samples) < 0.008, "is_fraud"] = 1

        self.feature_names = [col for col in df.columns if col != "is_fraud"]
        return df[self.feature_names], df["is_fraud"]

    def load_or_train_model(self):
        if self.model_path.exists():
            try:
                saved_data = joblib.load(self.model_path)
                self.model, self.feature_names = saved_data
                print("✅ Loaded pre-trained PerpetualBooster model + features")
                return
            except Exception:
                print("⚠️  Model file issue – retraining...")

        print("🚀 Training new PerpetualBooster model (runs only once)...")
        X, y = self._simulate_account_aggregator_data()
        X_np = X.to_numpy(dtype=np.float32)
        y_np = y.to_numpy(dtype=np.float32)

        self.model = PerpetualBooster(objective="LogLoss", budget=1.0)
        self.model.fit(X_np, y_np)

        # Save BOTH model and feature names
        joblib.dump((self.model, self.feature_names), self.model_path)
        print("✅ PerpetualBooster trained and saved successfully!")

    def extract_features(self, action: dict) -> np.ndarray:
        aa = action.get("aa_context", {})
        features = {
            "device_trust": float(action.get("device_trust", 0)),
            "velocity_seconds": float(action.get("velocity_seconds", 300)),
            "amount": float(action.get("amount", 10000)),
            "historical_avg_amount": float(aa.get("historical_avg_amount", 8000)),
            "retries": float(action.get("retries", 0)),
            "income_stability": float(aa.get("income_stability", 0.85)),
            "balance_to_amount_ratio": float(aa.get("balance_to_amount_ratio", 15.0)),
            "transaction_frequency_24h": float(aa.get("transaction_frequency_24h", 2)),
            "unusual_category_spend": float(aa.get("unusual_category_spend", 0)),
            "goal_deviation": float(aa.get("goal_deviation", 0.2)),
            "new_account_linked": float(aa.get("new_account_linked", 0)),
            "large_withdrawal_from_goal": float(aa.get("large_withdrawal_from_goal", 0)),
            "spending_spike_ratio": float(aa.get("spending_spike_ratio", 1.2)),
            "net_worth": float(aa.get("net_worth", 1200000)),
            "loan_outstanding_ratio": float(aa.get("loan_outstanding_ratio", 0.1)),
            "investment_volatility": float(aa.get("investment_volatility", 0.3)),
            "time_since_last_login_hours": float(aa.get("time_since_last_login_hours", 24)),
            "multiple_banks_linked": float(aa.get("multiple_banks_linked", 1)),
            "amount_vs_history": float(action.get("amount", 10000)) / (float(aa.get("historical_avg_amount", 8000)) + 1e-8),
            "velocity_vs_income": float(action.get("velocity_seconds", 300)) / (float(aa.get("income_stability", 0.85)) + 0.01),
        }
        df = pd.DataFrame([features])
        # Critical fix: use exact column order from training
        return df[self.feature_names].to_numpy(dtype=np.float32)[0]

    def calculate_risk_score(self, action: dict) -> dict:
        features_array = self.extract_features(action)
        
        prob = float(self.model.predict(features_array.reshape(1, -1))[0])
        risk_score = int(prob * 100)

        # Hybrid safety rule
        if (action.get("retries", 0) >= 4) or (action.get("device_trust", 0) == 30 and risk_score < 60):
            risk_score = max(risk_score, 75)

        # === NATIVE XAI (Shapley values) ===
        contrib = self.model.predict_contributions(
            features_array.reshape(1, -1),
            method="Shapley"
        )[0]

        contrib_dict = dict(zip(self.feature_names, contrib[:-1]))  # exclude bias
        top_features = sorted(contrib_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        explanation_lines = [
            f"• {name.replace('_', ' ').title()}" 
            for name, value in top_features if abs(value) > 0.01
        ]
        explanation = "Transaction risk increased because of:\n" + "\n".join(explanation_lines) if explanation_lines else "No major risk signals detected"

        decision = "Block" if risk_score >= 71 else "Timelock (24h)" if risk_score >= 40 else "Allow"

        return {
            "risk_score": risk_score,
            "decision": decision,
            "explanation": explanation,
            "probability": round(prob, 4),
            "timestamp": datetime.now().isoformat()
        }


# ====================== TEST ======================
if __name__ == "__main__":
    engine = EnhancedRiskEngine()   # First run trains (~20-30 sec), after that instant
    
    test_action = {
        "device_trust": 30,
        "velocity_seconds": 8,
        "amount": 450000,
        "retries": 2,
        "aa_context": {
            "historical_avg_amount": 12000,
            "large_withdrawal_from_goal": 1,
            "income_stability": 0.95,
            "net_worth": 1800000,
        }
    }
    
    result = engine.calculate_risk_score(test_action)
    print("\n" + "="*70)
    print("🔥 SECUREWEALTH TWIN RISK ENGINE (PERPETUALBOOSTER + NATIVE XAI)")
    print("="*70)
    print(f"Risk Score  : {result['risk_score']}/100")
    print(f"Decision    : {result['decision']}")
    print(f"Probability : {result['probability']}")
    print(f"\nExplanation:\n{result['explanation']}")
    print("="*70)