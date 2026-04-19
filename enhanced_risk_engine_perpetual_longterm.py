# enhanced_risk_engine_perpetual_longterm_fixed.py
# =====================================================
# SecureWealth Twin – Risk Engine (PerpetualBooster ONLY)
# NOW USES YOUR TEAM'S REAL MOCK DATA (SecureWealth_Twin_MockData.xlsx)
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
        self.data_file = Path("SecureWealth_Twin_MockData.xlsx")
        self.model = None
        self.feature_names = None
        self.load_or_train_model()

    def _load_real_mock_data(self):
        """Load your team's real mock dataset"""
        if not self.data_file.exists():
            raise FileNotFoundError(f"❌ File not found: {self.data_file}\nPlease place 'SecureWealth_Twin_MockData.xlsx' in the same folder.")

        print(f"📊 Loading real mock data from {self.data_file}...")
        df = pd.read_excel(self.data_file, sheet_name="Transactions")

        # Rename columns to match our feature names
        df = df.rename(columns={
            "Device Trust Score": "device_trust",
            "Velocity Flag": "velocity_seconds",      # treating flag as proxy
            "Amount Anomaly Flag": "amount_anomaly_flag",
            "Behavior Flag": "behavior_flag",
            "Risk Score": "risk_score",
            "Shield Decision": "shield_decision"
        })

        # Basic feature mapping (you can expand later)
        df["device_trust"] = df["device_trust"].fillna(0)
        df["velocity_seconds"] = df.get("velocity_seconds", 300)
        df["amount"] = df.get("Amount Inr", 10000).fillna(10000)
        df["historical_avg_amount"] = 8000  # placeholder – can be improved
        df["retries"] = 0
        df["new_device_verification_status"] = df["device_trust"].apply(lambda x: 0 if x == 30 else 2)
        df["churn_rate_30d"] = np.random.poisson(1.2, len(df)).clip(0, 12)
        df["return_consistency_score"] = np.random.uniform(0.3, 0.9, len(df))
        df["micro_tx_count_90d"] = np.random.poisson(4, len(df)).clip(0, 30)
        df["behavior_drift_score"] = np.random.uniform(0.0, 0.6, len(df))
        df["cumulative_risk_90d"] = np.random.poisson(40, len(df)).clip(0, 300)

        # Target
        df["is_fraud"] = (df["shield_decision"] == "Block").astype(int)

        self.feature_names = [
            "device_trust", "velocity_seconds", "amount", "historical_avg_amount",
            "retries", "new_device_verification_status", "churn_rate_30d",
            "return_consistency_score", "micro_tx_count_90d", "behavior_drift_score",
            "cumulative_risk_90d"
        ]
        return df[self.feature_names], df["is_fraud"]

    def load_or_train_model(self):
        if self.model_path.exists():
            try:
                saved_data = joblib.load(self.model_path)
                self.model, self.feature_names = saved_data
                print("✅ Loaded pre-trained model (using your team's mock data)")
                return
            except Exception:
                print("⚠️  Retraining with real mock data...")

        print("🚀 Training PerpetualBooster using your team's Excel mock data...")
        X, y = self._load_real_mock_data()
        X_np = X.to_numpy(dtype=np.float32)
        y_np = y.to_numpy(dtype=np.float32)

        self.model = PerpetualBooster(objective="LogLoss", budget=0.5)
        self.model.fit(X_np, y_np)

        joblib.dump((self.model, self.feature_names), self.model_path)
        print("✅ Model trained successfully on real mock dataset!")

    def extract_features(self, action: dict) -> np.ndarray:
        """Extract features for a single incoming action"""
        features = {
            "device_trust": float(action.get("device_trust", 0)),
            "velocity_seconds": float(action.get("velocity_seconds", 300)),
            "amount": float(action.get("amount", 10000)),
            "historical_avg_amount": float(action.get("historical_avg_amount", 8000)),
            "retries": float(action.get("retries", 0)),
            "new_device_verification_status": float(action.get("new_device_verification_status", 2)),
            "churn_rate_30d": float(action.get("churn_rate_30d", 1)),
            "return_consistency_score": float(action.get("return_consistency_score", 0.7)),
            "micro_tx_count_90d": float(action.get("micro_tx_count_90d", 2)),
            "behavior_drift_score": float(action.get("behavior_drift_score", 0.1)),
            "cumulative_risk_90d": float(action.get("cumulative_risk_90d", 30)),
        }
        df = pd.DataFrame([features])
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
        "device_trust": 30,
        "velocity_seconds": 12,
        "amount": 650000,
        "new_device_verification_status": 0,
        "churn_rate_30d": 11,
        "cumulative_risk_90d": 210,
        "aa_context": {"historical_avg_amount": 15000}
    }
    
    result = engine.calculate_risk_score(test_action)
    print("\n" + "="*80)
    print("🔥 RISK ENGINE USING YOUR TEAM'S REAL MOCK DATA")
    print("="*80)
    print(f"Risk Score  : {result['risk_score']}/100")
    print(f"Decision    : {result['decision']}")
    print(f"\nExplanation:\n{result['explanation']}")
    print("="*80)
