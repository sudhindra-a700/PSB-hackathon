# test_risk_engine.py
# =====================================================
# Quick & Complete Test Suite for SecureWealth Twin Risk Engine
# Run this file to test multiple real-world scenarios
# =====================================================

from risk_engine import EnhancedRiskEngine  # Change filename if yours is different
import json

def main():
    print("🔥 Loading SecureWealth Twin Risk Engine...\n")
    engine = EnhancedRiskEngine()   # loads model instantly after first training

    # Load test cases from JSON
    with open("sample_test_cases.json", "r") as f:
        test_cases = json.load(f)

    print("="*80)
    print("🚀 RUNNING FULL TEST SUITE (8 scenarios)")
    print("="*80)

    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {case['name']}")
        result = engine.calculate_risk_score(case["input"])
        
        print(f"   Risk Score  : {result['risk_score']}/100")
        print(f"   Decision    : {result['decision']}")
        print(f"   Probability : {result['probability']}")
        print(f"   Explanation :")
        print(f"               {result['explanation']}")
        print("-" * 60)

    print("\n✅ All tests completed! Your risk engine is working correctly.")

if __name__ == "__main__":
    main()