# enhanced_risk_engine_perpetual_longterm_fixed.py
# =====================================================
# SecureWealth Twin – Risk Engine (PerpetualBooster ONLY)
# FIXED: Missing interaction features (amount_vs_history, same_party_anomaly)
# Includes Real-time + Long-term Fraud Detection
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
    def __init__(self, model_path: str = "risk_model_perpetual_longterm.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        self.load_or_train_model()

    def _simulate_account_aggregator_data(self, n_samples: int = 8000):
        np.random.seed(42)
        data = {
            # Real-time signals
            "device_trust": np.random.choice([0, 30], n_samples, p=[0.85, 0.15]),
            "velocity_seconds": np.random.randint(5, 3600, n_samples),
            "amount": np.random.lognormal(8, 1.2, n_samples).astype(int),
            "historical_avg_amount": np.random.lognormal(7.5, 1.0, n_samples).astype(int),
            "retries": np.random.poisson(0.8, n_samples).clip(0, 5),
            "new_device_verification_status": np.random.choice([0, 1, 2], n_samples, p=[0.12, 0.18, 0.70]),

            # Same-party checks
            "same_party_past_tx_count": np.random.poisson(2.5, n_samples).clip(0, 15),
            "same_party_avg_amount": np.random.lognormal(8, 1.0, n_samples).astype(int),

            # Long-term fraud features
            "churn_rate_30d": np.random.poisson(1.2, n_samples).clip(0, 12),
            "return_consistency_score": np.random.uniform(0.1, 1.0, n_samples),
            "micro_tx_count_90d": np.random.poisson(4, n_samples).clip(0, 30),
            "behavior_drift_score": np.random.uniform(0.0, 0.8, n_samples),
            "same_party_micro_anomaly": np.random.lognormal(0, 0.9, n_samples),
            "cumulative_risk_90d": np.random.poisson(25, n_samples).clip(0, 300),
        }

        df = pd.DataFrame(data)

        # Interaction features (MUST be created both in training and runtime)
        df["amount_vs_history"] = df["amount"] / (df["historical_avg_amount"] + 1e-8)
        df["same_party_anomaly"] = df["amount"] / (df["same_party_avg_amount"] + 1e-8)

        # Target
        df["is_fraud"] = 0
        fraud_mask = (
            (df["new_device_verification_status"] == 0) |
            (df["retries"] >= 3) |
            (df["churn_rate_30d"] > 8) |
            (df["return_consistency_score"] < 0.25) |
            (df["micro_tx_count_90d"] > 20) |
            (df["behavior_drift_score"] > 0.6) |
            (df["cumulative_risk_90d"] > 180)
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
                print("✅ Loaded pre-trained model with long-term fraud detection")
                return
            except Exception:
                print("⚠️  Retraining...")

        print("🚀 Training new PerpetualBooster model with long-term fraud detection...")
        X, y = self._simulate_account_aggregator_data()
        X_np = X.to_numpy(dtype=np.float32)
        y_np = y.to_numpy(dtype=np.float32)

        self.model = PerpetualBooster(objective="LogLoss", budget=0.5)
        self.model.fit(X_np, y_np)

        joblib.dump((self.model, self.feature_names), self.model_path)
        print("✅ Model trained and saved successfully!")

    def extract_features(self, action: dict) -> np.ndarray:
        aa = action.get("aa_context", {})
        features = {
            "device_trust": float(action.get("device_trust", 0)),
            "velocity_seconds": float(action.get("velocity_seconds", 300)),
            "amount": float(action.get("amount", 10000)),
            "historical_avg_amount": float(aa.get("historical_avg_amount", 8000)),
            "retries": float(action.get("retries", 0)),
            "new_device_verification_status": float(action.get("new_device_verification_status", 2)),
            "same_party_past_tx_count": float(action.get("same_party_past_tx_count", 3)),
            "same_party_avg_amount": float(action.get("same_party_avg_amount", 15000)),
            "churn_rate_30d": float(action.get("churn_rate_30d", 1)),
            "return_consistency_score": float(action.get("return_consistency_score", 0.7)),
            "micro_tx_count_90d": float(action.get("micro_tx_count_90d", 2)),
            "behavior_drift_score": float(action.get("behavior_drift_score", 0.1)),
            "same_party_micro_anomaly": float(action.get("amount", 10000)) / (float(action.get("same_party_avg_amount", 5000)) + 1e-8),
            "cumulative_risk_90d": float(action.get("cumulative_risk_90d", 30)),
        }

        df = pd.DataFrame([features])

        # CRITICAL FIX: Recreate the interaction features here
        df["amount_vs_history"] = df["amount"] / (df["historical_avg_amount"] + 1e-8)
        df["same_party_anomaly"] = df["amount"] / (df["same_party_avg_amount"] + 1e-8)

        # Now select columns in exact training order
        return df[self.feature_names].to_numpy(dtype=np.float32)[0]

    def calculate_risk_score(self, action: dict) -> dict:
        features_array = self.extract_features(action)
        prob = float(self.model.predict(features_array.reshape(1, -1))[0])
        risk_score = int(prob * 100)

        # Hybrid safety rules
        if action.get("new_device_verification_status", 2) == 0:
            risk_score = max(risk_score, 80)
        if action.get("churn_rate_30d", 1) > 8:
            risk_score = max(risk_score, 75)
        if action.get("cumulative_risk_90d", 30) > 180:
            risk_score = max(risk_score, 85)

        # Native XAI
        contrib = self.model.predict_contributions(features_array.reshape(1, -1), method="Shapley")[0]
        contrib_dict = dict(zip(self.feature_names, contrib[:-1]))
        top_features = sorted(contrib_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:6]

        explanation_lines = [f"• {name.replace('_', ' ').title()}" for name, value in top_features if abs(value) > 0.01]
        explanation = "Transaction risk increased because of:\n" + "\n".join(explanation_lines) if explanation_lines else "No major risk signals"

        decision = "Block" if risk_score >= 71 else "Timelock (24h)" if risk_score >= 40 else "Allow"

        return {
            "risk_score": risk_score,
            "decision": decision,
            "explanation": explanation,
            "probability": round(prob, 4),
            "timestamp": datetime.now().isoformat()
        }


# ====================== QUICK TEST ======================
if __name__ == "__main__":
    engine = EnhancedRiskEngine()
    
    test_action = {
        "device_trust": 0,
        "velocity_seconds": 300,
        "amount": 15000,
        "retries": 0,
        "new_device_verification_status": 2,
        "same_party_past_tx_count": 5,
        "churn_rate_30d": 11,
        "return_consistency_score": 0.15,
        "micro_tx_count_90d": 25,
        "behavior_drift_score": 0.65,
        "cumulative_risk_90d": 210,
        "aa_context": {"historical_avg_amount": 12000}
    }
    
    result = engine.calculate_risk_score(test_action)
    print("\n" + "="*80)
    print("🔥 LONG-TERM FRAUD ENABLED RISK ENGINE (FIXED)")
    print("="*80)
    print(f"Risk Score  : {result['risk_score']}/100")
    print(f"Decision    : {result['decision']}")
    print(f"\nExplanation:\n{result['explanation']}")
    print("="*80)