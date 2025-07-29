#!/usr/bin/env python3
"""
Binary Classification ML Model using XGBoost and Random Forest

This script provides a comprehensive machine learning pipeline for binary classification
using both XGBoost and Random Forest algorithms with proper train/test/validation splits.

Author: AI Assistant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, accuracy_score, 
    precision_score, recall_score, f1_score
)
from sklearn.datasets import make_classification
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class BinaryClassifier:
    """
    A comprehensive binary classification model using XGBoost and Random Forest.
    
    Features:
    - Data preprocessing and validation
    - Train/Test/Validation split
    - Model training with hyperparameter tuning
    - Comprehensive evaluation metrics
    - Visualization of results
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the binary classifier.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Models
        self.rf_model = None
        self.xgb_model = None
        
        # Data splits
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Results
        self.results = {}
        
    def load_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Load and split data into train/validation/test sets.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of training data for validation
        """
        print("Loading and splitting data...")
        print(f"Original dataset shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Validation set shape: {self.X_val.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data preprocessing completed!")
        
    def train_random_forest(self, use_grid_search=True):
        """
        Train Random Forest model with optional hyperparameter tuning.
        
        Args:
            use_grid_search (bool): Whether to use grid search for hyperparameter tuning
        """
        print("\n" + "="*50)
        print("Training Random Forest Model")
        print("="*50)
        
        if use_grid_search:
            print("Performing grid search for hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf_base = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                rf_base, param_grid, cv=5, scoring='roc_auc', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            self.rf_model = grid_search.best_estimator_
            print(f"Best RF parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            self.rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            )
            self.rf_model.fit(self.X_train_scaled, self.y_train)
        
        # Validation predictions
        rf_val_pred = self.rf_model.predict(self.X_val_scaled)
        rf_val_pred_proba = self.rf_model.predict_proba(self.X_val_scaled)[:, 1]
        
        self.results['rf'] = {
            'val_accuracy': accuracy_score(self.y_val, rf_val_pred),
            'val_precision': precision_score(self.y_val, rf_val_pred),
            'val_recall': recall_score(self.y_val, rf_val_pred),
            'val_f1': f1_score(self.y_val, rf_val_pred),
            'val_auc': roc_auc_score(self.y_val, rf_val_pred_proba),
            'val_pred': rf_val_pred,
            'val_pred_proba': rf_val_pred_proba
        }
        
        print(f"Random Forest Validation Results:")
        print(f"Accuracy: {self.results['rf']['val_accuracy']:.4f}")
        print(f"Precision: {self.results['rf']['val_precision']:.4f}")
        print(f"Recall: {self.results['rf']['val_recall']:.4f}")
        print(f"F1-Score: {self.results['rf']['val_f1']:.4f}")
        print(f"AUC: {self.results['rf']['val_auc']:.4f}")
        
    def train_xgboost(self, use_grid_search=True):
        """
        Train XGBoost model with optional hyperparameter tuning.
        
        Args:
            use_grid_search (bool): Whether to use grid search for hyperparameter tuning
        """
        print("\n" + "="*50)
        print("Training XGBoost Model")
        print("="*50)
        
        if use_grid_search:
            print("Performing grid search for hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            xgb_base = xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss')
            grid_search = GridSearchCV(
                xgb_base, param_grid, cv=5, scoring='roc_auc', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            self.xgb_model = grid_search.best_estimator_
            print(f"Best XGB parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                random_state=self.random_state,
                eval_metric='logloss'
            )
            self.xgb_model.fit(self.X_train_scaled, self.y_train)
        
        # Validation predictions
        xgb_val_pred = self.xgb_model.predict(self.X_val_scaled)
        xgb_val_pred_proba = self.xgb_model.predict_proba(self.X_val_scaled)[:, 1]
        
        self.results['xgb'] = {
            'val_accuracy': accuracy_score(self.y_val, xgb_val_pred),
            'val_precision': precision_score(self.y_val, xgb_val_pred),
            'val_recall': recall_score(self.y_val, xgb_val_pred),
            'val_f1': f1_score(self.y_val, xgb_val_pred),
            'val_auc': roc_auc_score(self.y_val, xgb_val_pred_proba),
            'val_pred': xgb_val_pred,
            'val_pred_proba': xgb_val_pred_proba
        }
        
        print(f"XGBoost Validation Results:")
        print(f"Accuracy: {self.results['xgb']['val_accuracy']:.4f}")
        print(f"Precision: {self.results['xgb']['val_precision']:.4f}")
        print(f"Recall: {self.results['xgb']['val_recall']:.4f}")
        print(f"F1-Score: {self.results['xgb']['val_f1']:.4f}")
        print(f"AUC: {self.results['xgb']['val_auc']:.4f}")
        
    def evaluate_test_set(self):
        """
        Evaluate both models on the test set and compare performance.
        """
        print("\n" + "="*50)
        print("Final Test Set Evaluation")
        print("="*50)
        
        for model_name, model in [('Random Forest', self.rf_model), ('XGBoost', self.xgb_model)]:
            if model is None:
                continue
                
            # Test predictions
            test_pred = model.predict(self.X_test_scaled)
            test_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            test_accuracy = accuracy_score(self.y_test, test_pred)
            test_precision = precision_score(self.y_test, test_pred)
            test_recall = recall_score(self.y_test, test_pred)
            test_f1 = f1_score(self.y_test, test_pred)
            test_auc = roc_auc_score(self.y_test, test_pred_proba)
            
            model_key = 'rf' if model_name == 'Random Forest' else 'xgb'
            self.results[model_key].update({
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'test_pred': test_pred,
                'test_pred_proba': test_pred_proba
            })
            
            print(f"\n{model_name} Test Results:")
            print(f"Accuracy: {test_accuracy:.4f}")
            print(f"Precision: {test_precision:.4f}")
            print(f"Recall: {test_recall:.4f}")
            print(f"F1-Score: {test_f1:.4f}")
            print(f"AUC: {test_auc:.4f}")
            
            # Classification report
            print(f"\n{model_name} Classification Report:")
            print(classification_report(self.y_test, test_pred))
        
        # Model comparison
        self._compare_models()
        
    def _compare_models(self):
        """
        Compare the performance of both models.
        """
        print("\n" + "="*50)
        print("Model Comparison Summary")
        print("="*50)
        
        comparison_df = pd.DataFrame({
            'Metric': ['Validation AUC', 'Test AUC', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1'],
            'Random Forest': [
                self.results['rf']['val_auc'],
                self.results['rf']['test_auc'],
                self.results['rf']['test_accuracy'],
                self.results['rf']['test_precision'],
                self.results['rf']['test_recall'],
                self.results['rf']['test_f1']
            ],
            'XGBoost': [
                self.results['xgb']['val_auc'],
                self.results['xgb']['test_auc'],
                self.results['xgb']['test_accuracy'],
                self.results['xgb']['test_precision'],
                self.results['xgb']['test_recall'],
                self.results['xgb']['test_f1']
            ]
        })
        
        print(comparison_df.round(4))
        
        # Determine best model
        rf_test_auc = self.results['rf']['test_auc']
        xgb_test_auc = self.results['xgb']['test_auc']
        
        if rf_test_auc > xgb_test_auc:
            print(f"\nBest Model: Random Forest (Test AUC: {rf_test_auc:.4f})")
        elif xgb_test_auc > rf_test_auc:
            print(f"\nBest Model: XGBoost (Test AUC: {xgb_test_auc:.4f})")
        else:
            print(f"\nBoth models perform equally well (Test AUC: {rf_test_auc:.4f})")
    
    def plot_results(self, save_plots=True):
        """
        Create comprehensive visualization of model results.
        
        Args:
            save_plots (bool): Whether to save plots to files
        """
        print("\nGenerating visualization plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Binary Classification Model Comparison', fontsize=16, fontweight='bold')
        
        # 1. ROC Curves
        ax1 = axes[0, 0]
        for model_name, model_key in [('Random Forest', 'rf'), ('XGBoost', 'xgb')]:
            if model_key in self.results:
                fpr, tpr, _ = roc_curve(self.y_test, self.results[model_key]['test_pred_proba'])
                auc = self.results[model_key]['test_auc']
                ax1.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves - Test Set')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curves
        ax2 = axes[0, 1]
        for model_name, model_key in [('Random Forest', 'rf'), ('XGBoost', 'xgb')]:
            if model_key in self.results:
                precision, recall, _ = precision_recall_curve(self.y_test, self.results[model_key]['test_pred_proba'])
                ax2.plot(recall, precision, label=f'{model_name}')
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves - Test Set')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Confusion Matrices
        ax3 = axes[0, 2]
        rf_cm = confusion_matrix(self.y_test, self.results['rf']['test_pred'])
        sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title('Random Forest - Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        ax4 = axes[1, 0]
        xgb_cm = confusion_matrix(self.y_test, self.results['xgb']['test_pred'])
        sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Greens', ax=ax4)
        ax4.set_title('XGBoost - Confusion Matrix')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        
        # 4. Performance Metrics Comparison
        ax5 = axes[1, 1]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        rf_values = [
            self.results['rf']['test_accuracy'],
            self.results['rf']['test_precision'],
            self.results['rf']['test_recall'],
            self.results['rf']['test_f1'],
            self.results['rf']['test_auc']
        ]
        xgb_values = [
            self.results['xgb']['test_accuracy'],
            self.results['xgb']['test_precision'],
            self.results['xgb']['test_recall'],
            self.results['xgb']['test_f1'],
            self.results['xgb']['test_auc']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax5.bar(x - width/2, rf_values, width, label='Random Forest', alpha=0.8)
        ax5.bar(x + width/2, xgb_values, width, label='XGBoost', alpha=0.8)
        ax5.set_xlabel('Metrics')
        ax5.set_ylabel('Score')
        ax5.set_title('Performance Metrics Comparison')
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 5. Feature Importance (if available)
        ax6 = axes[1, 2]
        if hasattr(self.rf_model, 'feature_importances_'):
            n_features = min(10, len(self.rf_model.feature_importances_))
            rf_importance = self.rf_model.feature_importances_[:n_features]
            feature_names = [f'Feature_{i}' for i in range(n_features)]
            
            y_pos = np.arange(len(feature_names))
            ax6.barh(y_pos, rf_importance)
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(feature_names)
            ax6.set_xlabel('Importance')
            ax6.set_title('Random Forest - Top Feature Importances')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('ml_classification_results.png', dpi=300, bbox_inches='tight')
            print("Plots saved as 'ml_classification_results.png'")
        
        plt.show()
    
    def get_model_summary(self):
        """
        Get a summary of both trained models.
        
        Returns:
            dict: Summary of model performance
        """
        summary = {
            'data_splits': {
                'train_size': len(self.X_train),
                'validation_size': len(self.X_val),
                'test_size': len(self.X_test),
                'total_features': self.X_train.shape[1]
            },
            'random_forest': self.results.get('rf', {}),
            'xgboost': self.results.get('xgb', {})
        }
        
        return summary


def generate_sample_data(n_samples=1000, n_features=20, n_informative=10, random_state=42):
    """
    Generate sample binary classification dataset for demonstration.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        n_informative (int): Number of informative features
        random_state (int): Random state for reproducibility
    
    Returns:
        tuple: (X, y) feature matrix and target vector
    """
    print("Generating sample binary classification dataset...")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=random_state
    )
    
    print(f"Generated dataset with {n_samples} samples and {n_features} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y


def main():
    """
    Main function to demonstrate the binary classification pipeline.
    """
    print("="*60)
    print("BINARY CLASSIFICATION WITH XGBOOST AND RANDOM FOREST")
    print("="*60)
    
    # Generate sample data
    X, y = generate_sample_data(n_samples=2000, n_features=20, n_informative=15)
    
    # Create and configure classifier
    classifier = BinaryClassifier(random_state=42)
    
    # Load and split data
    classifier.load_data(X, y, test_size=0.2, val_size=0.2)
    
    # Train models
    classifier.train_random_forest(use_grid_search=False)  # Set to True for grid search
    classifier.train_xgboost(use_grid_search=False)        # Set to True for grid search
    
    # Evaluate on test set
    classifier.evaluate_test_set()
    
    # Generate plots
    classifier.plot_results(save_plots=True)
    
    # Get model summary
    summary = classifier.get_model_summary()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Models trained and evaluated with train/validation/test splits")
    print("Results visualization saved as 'ml_classification_results.png'")
    
    return classifier, summary


if __name__ == "__main__":
    # Run the main pipeline
    classifier, summary = main()