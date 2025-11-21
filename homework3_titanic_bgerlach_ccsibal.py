import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    d = pd.read_csv("train.csv")
    y = d.Survived.to_numpy()
    sex = d.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass = d.Pclass.to_numpy()

    from homework3_bgerlach_ccsibal import softmaxRegression, compute_accuracy, one_hot

def prepare_features(data, sibsp_categories=None, pclass_categories=None, fit=False):
    """
    Prepare features from the Titanic dataset.
    Uses Sex, Pclass, and SibSp as specified in the homework.

    We'll treat Pclass and SibSp as categorical variables for better performance.
    """
    n = len(data)

    # Sex: binary (0=male, 1=female)
    sex = data['Sex'].map({"male": 0, "female": 1}).to_numpy()

    # Pclass: categorical (1, 2, 3) one-hot encode
    pclass = data['Pclass'].to_numpy()

    # SibSp: number of siblings/spouses one-hot encode
    sibsp = data['SibSp'].to_numpy()
    sibsp_capped = np.minimum(sibsp, 5)  # Cap at 5

    if fit:
        pclass_categories = np.unique(pclass)
        sibsp_categories = np.unique(sibsp_capped)

    pclass_onehot = np.zeros((n, len(pclass_categories)))
    for i, val in enumerate(pclass):
        idx = np.where(pclass_categories == val)[0]
        if len(idx) > 0:
            pclass_onehot[i, idx[0]] = 1

    sibsp_onehot = np.zeros((n, len(sibsp_categories)))
    for i, val in enumerate(sibsp_capped):
        idx = np.where(sibsp_categories == val)[0]
        if len(idx) > 0:
            sibsp_onehot[i, idx[0]] = 1

    # Combine features: sex + pclass_onehot + sibsp_onehot
    features = np.column_stack([
        sex.reshape(-1, 1),
        pclass_onehot,
        sibsp_onehot
    ])

    if fit:
        return features, pclass_categories, sibsp_categories
    else:
        return features

if __name__ == "__main__":
    print("="*80)
    print("TITANIC SURVIVAL PREDICTION - SOFTMAX REGRESSION")
    print("="*80)

    # Load training data
    print("\nLoading training data...")
    train_data = pd.read_csv("train.csv")
    y_train = train_data['Survived'].to_numpy()

    print(f"Training samples: {len(train_data)}")
    print(f"Survival rate: {y_train.mean():.2%}")

    print("\nPreparing features")
    X_train, pclass_cats, sibsp_cats = prepare_features(train_data, fit=True)
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"  - Sex: 1 feature (binary)")
    print(f"  - Pclass: {len(pclass_cats)} categories (one-hot)")
    print(f"  - SibSp: {len(sibsp_cats)} categories (one-hot, capped at 5)")

    X_train = np.column_stack([X_train, np.ones(len(X_train))])
    print(f"After adding bias: {X_train.shape}")

    y_train_onehot = one_hot(y_train, 2)

    n_train = int(0.8 * len(X_train))
    indices = np.random.permutation(len(X_train))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train

    print(f"x_tr: {X_train.shape}")
    print(f"y_tr: {y_train.shape}")

    print(f"\nSplit data:")
    print(f"  Training: {len(X_tr)} samples")
    print(f"  Validation: {len(X_val)} samples")

    W = softmaxRegression(
        X_tr, y_tr, X_tr, y_tr,
        epsilon=0.01,
        batchSize=32,
        alpha=1.0
    )[:-1]

    # Evaluate on full training set
    print("\n" + "="*80)
    print("EVALUATION ON TRAINING DATA")
    print("="*80)
    train_acc = compute_accuracy(X_train, y_train, W)
    print(f"Full Training Set Accuracy: {100 * train_acc:.2f}%")

    # Load test data
    print("\n" + "="*80)
    print("MAKING PREDICTIONS ON TEST DATA")
    print("="*80)
    test_data = pd.read_csv("test.csv")
    passenger_ids = test_data['PassengerId'].to_numpy()

    print(f"Test samples: {len(test_data)}")

    X_test = prepare_features(test_data, sibsp_categories=sibsp_cats, pclass_categories=pclass_cats, fit=False)

    X_test = np.column_stack([X_test, np.ones(len(X_test))])

    scores = X_test @ W
    predictions = np.argmax(scores, axis=1)

    print(f"Predicted survival rate: {predictions.mean():.2%}")

    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })

    submission.to_csv('titanic_submission.csv', index=False)

    print("\nFirst 10 predictions:")
    print(submission.head(10))

    print("\n" + "="*80)
    print("WEIGHT ANALYSIS")
    print("="*80)

    feature_names = ['Sex'] + [f'Pclass_{int(c)}' for c in pclass_cats] + [f'SibSp_{int(c)}' for c in sibsp_cats] + ['Bias']

    survived_weights = W[:, 1]

    print("\nWeights for 'Survived' class:")
    for name, weight in zip(feature_names, survived_weights):
        sign = "+" if weight > 0 else ""
        print(f"  {name:15s}: {sign}{weight:7.4f}")

    plt.figure(figsize=(10, 6))
    colors = ['green' if w > 0 else 'red' for w in survived_weights[:-1]]
    plt.bar(range(len(survived_weights)-1), survived_weights[:-1], color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xticks(range(len(feature_names)-1), feature_names[:-1], rotation=45, ha='right')
    plt.ylabel('Weight')
    plt.title('Feature Weights for Survival Prediction')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('titanic_weights.png', dpi=150, bbox_inches='tight')
