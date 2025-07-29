#!/usr/bin/env python3
"""
Simple ML Binary Classification Demo

This script demonstrates the basic usage of the comprehensive ML binary
classification model with XGBoost and Random Forest.
"""

import numpy as np
from ml_binary_classifier import BinaryClassifier, generate_sample_data

def quick_demo():
    """
    Quick demonstration of the ML binary classifier.
    """
    print("🚀 ML Binary Classification Quick Demo")
    print("=" * 50)
    
    # Generate sample data
    print("📊 Generating sample data...")
    X, y = generate_sample_data(n_samples=1000, n_features=10, n_informative=8)
    print(f"✅ Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📈 Class distribution: {np.bincount(y)}")
    
    # Initialize classifier
    print("\n🤖 Initializing ML classifier...")
    classifier = BinaryClassifier(random_state=42)
    
    # Load and split data (60% train, 20% validation, 20% test)
    print("🔄 Splitting data into train/validation/test sets...")
    classifier.load_data(X, y, test_size=0.2, val_size=0.25)
    
    # Train both models (without grid search for speed)
    print("\n🌲 Training Random Forest...")
    classifier.train_random_forest(use_grid_search=False)
    
    print("\n🚀 Training XGBoost...")
    classifier.train_xgboost(use_grid_search=False)
    
    # Evaluate on test set
    print("\n📊 Evaluating models on test set...")
    classifier.evaluate_test_set()
    
    # Show feature importance
    print("\n🎯 Feature Importance Analysis:")
    if hasattr(classifier, 'rf_model') and classifier.rf_model:
        rf_importance = classifier.rf_model.feature_importances_
        print(f"Random Forest - Top 3 features: {np.argsort(rf_importance)[-3:][::-1]}")
    
    if hasattr(classifier, 'xgb_model') and classifier.xgb_model:
        xgb_importance = classifier.xgb_model.feature_importances_
        print(f"XGBoost - Top 3 features: {np.argsort(xgb_importance)[-3:][::-1]}")
    
    # Make predictions on new data
    print("\n🔮 Making predictions on new data...")
    X_new, _ = generate_sample_data(n_samples=5, n_features=10, n_informative=8)
    
    # Scale new data using the fitted scaler
    X_new_scaled = classifier.scaler.transform(X_new)
    
    # Make predictions with both models
    rf_pred = classifier.rf_model.predict(X_new_scaled)
    rf_pred_proba = classifier.rf_model.predict_proba(X_new_scaled)
    
    xgb_pred = classifier.xgb_model.predict(X_new_scaled)
    xgb_pred_proba = classifier.xgb_model.predict_proba(X_new_scaled)
    
    # Create predictions dataframe
    import pandas as pd
    predictions_df = pd.DataFrame({
        'Sample': range(1, len(X_new) + 1),
        'RF_Prediction': rf_pred,
        'RF_Probability_Class_1': rf_pred_proba[:, 1],
        'XGB_Prediction': xgb_pred,
        'XGB_Probability_Class_1': xgb_pred_proba[:, 1]
    })
    
    print("New data predictions:")
    print(predictions_df.round(4).to_string(index=False))
    
    print("\n✅ Demo completed successfully!")
    return classifier

if __name__ == "__main__":
    classifier = quick_demo()