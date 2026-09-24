# ============================================================
# CREDIT CARD FRAUD DETECTION DEMONSTRATION
# Python + Scikit-learn
# ============================================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ------------------------------------------------------------
# 1. CREATE SAMPLE TRANSACTION DATA
# ------------------------------------------------------------

data = pd.DataFrame({

    "amount": [
        25.50, 42.18, 86.40, 19.95,
        214.75, 2480.90, 3200.00,
        55.25, 18.99, 1750.00,
        34.20, 2900.00
    ],

    "distance_from_home": [
        2, 6, 10, 3,
        15, 850, 1200,
        4, 1, 700,
        8, 950
    ],

    "transactions_1h": [
        1, 2, 2, 1,
        3, 9, 12,
        1, 1, 8,
        2, 10
    ],

    "device_mismatch": [
        0, 0, 0, 0,
        0, 1, 1,
        0, 0, 1,
        0, 1
    ],

    "is_fraud": [
        0, 0, 0, 0,
        0, 1, 1,
        0, 0, 1,
        0, 1
    ]
})


# ------------------------------------------------------------
# 2. SELECT THE FEATURES
# ------------------------------------------------------------

features = [
    "amount",
    "distance_from_home",
    "transactions_1h",
    "device_mismatch"
]

X = data[features]

# Target:
# 0 = legitimate
# 1 = fraud

y = data["is_fraud"]


# ------------------------------------------------------------
# 3. SPLIT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 4. CREATE THE MACHINE-LEARNING MODEL
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)


# ------------------------------------------------------------
# 5. TRAIN THE MODEL
# ------------------------------------------------------------

model.fit(X_train, y_train)


# ------------------------------------------------------------
# 6. FRAUD-DETECTION FUNCTION
# ------------------------------------------------------------

def check_transaction(transaction):

    # Put transaction information into a DataFrame
    # using the same columns the model was trained on.

    transaction_data = pd.DataFrame(
        [transaction],
        columns=features
    )

    # Calculate fraud probability

    fraud_probability = model.predict_proba(
        transaction_data
    )[0][1]


    # --------------------------------------------------------
    # DECISION RULES
    # --------------------------------------------------------

    if fraud_probability >= 0.80:

        decision = "BLOCK"

    elif fraud_probability >= 0.45:

        decision = "REVIEW"

    else:

        decision = "APPROVE"


    return decision, fraud_probability


# ------------------------------------------------------------
# 7. SIMULATE A NORMAL TRANSACTION
# ------------------------------------------------------------

normal_transaction = {

    "amount": 42.18,

    "distance_from_home": 6,

    "transactions_1h": 2,

    "device_mismatch": 0
}


decision, probability = check_transaction(
    normal_transaction
)


print("\n--------------------------------")
print("NORMAL TRANSACTION")
print("--------------------------------")

print("Amount: $", normal_transaction["amount"])

print(
    "Fraud Probability:",
    f"{probability:.1%}"
)

print(
    "Decision:",
    decision
)


# ------------------------------------------------------------
# 8. SIMULATE A SUSPICIOUS TRANSACTION
# ------------------------------------------------------------

suspicious_transaction = {

    "amount": 2480.90,

    "distance_from_home": 850,

    "transactions_1h": 9,

    "device_mismatch": 1
}


decision, probability = check_transaction(
    suspicious_transaction
)


print("\n--------------------------------")
print("SUSPICIOUS TRANSACTION")
print("--------------------------------")

print(
    "Amount: $",
    suspicious_transaction["amount"]
)

print(
    "Distance From Home:",
    suspicious_transaction["distance_from_home"],
    "miles"
)

print(
    "Transactions Last Hour:",
    suspicious_transaction["transactions_1h"]
)

print(
    "Device Mismatch:",
    suspicious_transaction["device_mismatch"]
)

print(
    "Fraud Probability:",
    f"{probability:.1%}"
)

print(
    "Decision:",
    decision
)


# ------------------------------------------------------------
# 9. TEST MODEL PERFORMANCE
# ------------------------------------------------------------

predictions = model.predict(X_test)

print("\n--------------------------------")
print("MODEL TEST RESULTS")
print("--------------------------------")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)