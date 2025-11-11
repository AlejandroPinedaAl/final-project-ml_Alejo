"""
model_training_evaluation.py
Entrenamiento y Evaluación de Modelos de Machine Learning.

Este módulo contiene todas las funciones necesarias para:
- Entrenar múltiples modelos de clasificación
- Evaluar modelos con métricas completas
- Comparar modelos y seleccionar el mejor
- Visualizar resultados (ROC, confusion matrix, feature importance)
- Guardar el modelo seleccionado

Autor: Alejandro Pineda Alvarez
Proyecto: Marketing Campaign Response Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, auc, confusion_matrix, 
    classification_report, precision_recall_curve, average_precision_score
)

# XGBoost y LightGBM
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost no disponible. Instalar con: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM no disponible. Instalar con: pip install lightgbm")


# ============================================================================
# 1. FUNCIÓN DE ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# ============================================================================

def build_model(model, X_train, y_train, X_test, y_test, model_name="Model",
                cv_folds=5, verbose=True):
    """
    Entrena y evalúa un modelo de clasificación.
    
    Args:
        model: Modelo de sklearn/xgboost/lightgbm a entrenar
        X_train: Features de entrenamiento (ya transformadas)
        y_train: Target de entrenamiento
        X_test: Features de prueba (ya transformadas)
        y_test: Target de prueba
        model_name: Nombre del modelo para identificación
        cv_folds: Número de folds para validación cruzada
        verbose: Si True, imprime información del proceso
        
    Returns:
        dict con modelo entrenado, predicciones y métricas
    """
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🤖 ENTRENANDO: {model_name}")
        print(f"{'='*80}")
    
    # 1. Entrenar modelo
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    if verbose:
        print(f"✅ Modelo entrenado en {training_time:.2f} segundos")
    
    # 2. Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilidades (si el modelo las soporta)
    try:
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_train_proba = None
        y_test_proba = None
        if verbose:
            print("⚠️ Modelo no soporta predict_proba")
    
    # 3. Métricas en train
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1': f1_score(y_train, y_train_pred, zero_division=0)
    }
    
    if y_train_proba is not None:
        train_metrics['roc_auc'] = roc_auc_score(y_train, y_train_proba)
    
    # 4. Métricas en test
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0)
    }
    
    if y_test_proba is not None:
        test_metrics['roc_auc'] = roc_auc_score(y_test, y_test_proba)
    
    # 5. Validación cruzada
    cv_scores = {}
    if cv_folds > 0:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = cross_validate(model, X_train, y_train, cv=cv, 
                                    scoring=scoring, n_jobs=-1)
        
        for metric in scoring:
            cv_scores[f'{metric}_mean'] = cv_results[f'test_{metric}'].mean()
            cv_scores[f'{metric}_std'] = cv_results[f'test_{metric}'].std()
    
    # 6. Overfitting check
    overfitting = {
        'accuracy_diff': train_metrics['accuracy'] - test_metrics['accuracy'],
        'f1_diff': train_metrics['f1'] - test_metrics['f1']
    }
    
    if verbose:
        print(f"\n📊 Métricas en Test:")
        print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall:    {test_metrics['recall']:.4f}")
        print(f"   F1-Score:  {test_metrics['f1']:.4f}")
        if 'roc_auc' in test_metrics:
            print(f"   ROC-AUC:   {test_metrics['roc_auc']:.4f}")
        
        if cv_folds > 0:
            print(f"\n📊 Validación Cruzada ({cv_folds}-fold):")
            print(f"   F1-Score: {cv_scores['f1_mean']:.4f} ± {cv_scores['f1_std']:.4f}")
            print(f"   ROC-AUC:  {cv_scores['roc_auc_mean']:.4f} ± {cv_scores['roc_auc_std']:.4f}")
        
        print(f"\n⚠️ Overfitting Check:")
        print(f"   Accuracy diff: {overfitting['accuracy_diff']:.4f}")
        print(f"   F1 diff:       {overfitting['f1_diff']:.4f}")
        
        if overfitting['f1_diff'] > 0.1:
            print(f"   ⚠️ Posible overfitting detectado")
        else:
            print(f"   ✅ Modelo generaliza bien")
    
    # 7. Retornar resultados
    return {
        'model': model,
        'model_name': model_name,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'y_train_proba': y_train_proba,
        'y_test_proba': y_test_proba,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'cv_scores': cv_scores,
        'overfitting': overfitting,
        'training_time': training_time
    }


# ============================================================================
# 2. FUNCIÓN DE RESUMEN DE MÉTRICAS
# ============================================================================

def summarize_classification(y_true, y_pred, y_pred_proba=None, model_name="Model",
                            show_confusion_matrix=True, show_classification_report=True):
    """
    Resume las métricas de clasificación de forma completa.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        y_pred_proba: Probabilidades predichas (opcional)
        model_name: Nombre del modelo
        show_confusion_matrix: Si True, muestra matriz de confusión
        show_classification_report: Si True, muestra reporte de clasificación
        
    Returns:
        dict con todas las métricas
    """
    
    print(f"\n{'='*80}")
    print(f"📊 RESUMEN DE CLASIFICACIÓN: {model_name}")
    print(f"{'='*80}")
    
    # Métricas básicas
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Métricas con probabilidades
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
    
    # Imprimir métricas
    print(f"\n📈 Métricas Principales:")
    print(f"   Accuracy:         {metrics['accuracy']:.4f}")
    print(f"   Precision:        {metrics['precision']:.4f}")
    print(f"   Recall:           {metrics['recall']:.4f}")
    print(f"   F1-Score:         {metrics['f1_score']:.4f}")
    
    if y_pred_proba is not None:
        print(f"   ROC-AUC:          {metrics['roc_auc']:.4f}")
        print(f"   Avg Precision:    {metrics['avg_precision']:.4f}")
    
    # Matriz de confusión
    if show_confusion_matrix:
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n🔢 Matriz de Confusión:")
        print(f"   TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
        print(f"   FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
        
        # Métricas derivadas
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n   Specificity:      {specificity:.4f}")
        print(f"   Sensitivity:      {sensitivity:.4f}")
        
        metrics['confusion_matrix'] = cm
        metrics['specificity'] = specificity
        metrics['sensitivity'] = sensitivity
    
    # Reporte de clasificación
    if show_classification_report:
        print(f"\n📋 Reporte de Clasificación:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['No acepta (0)', 'Acepta (1)']))
    
    print(f"{'='*80}")
    
    return metrics


# ============================================================================
# 3. COMPARACIÓN DE MODELOS
# ============================================================================

def compare_models(results_dict, metric='f1', show_plot=True):
    """
    Compara múltiples modelos y genera visualizaciones.
    
    Args:
        results_dict: Diccionario con resultados de cada modelo
                     {model_name: result_dict}
        metric: Métrica principal para comparación
        show_plot: Si True, muestra gráficos comparativos
        
    Returns:
        DataFrame comparativo ordenado por métrica
    """
    
    print(f"\n{'='*80}")
    print(f"🏆 COMPARACIÓN DE MODELOS")
    print(f"{'='*80}")
    
    # Crear DataFrame comparativo
    comparison_data = []
    
    for model_name, result in results_dict.items():
        test_metrics = result['test_metrics']
        cv_scores = result.get('cv_scores', {})
        
        row = {
            'Model': model_name,
            'Accuracy': test_metrics['accuracy'],
            'Precision': test_metrics['precision'],
            'Recall': test_metrics['recall'],
            'F1-Score': test_metrics['f1'],
            'ROC-AUC': test_metrics.get('roc_auc', np.nan),
            'CV_F1_mean': cv_scores.get('f1_mean', np.nan),
            'CV_F1_std': cv_scores.get('f1_std', np.nan),
            'Overfitting_F1': result['overfitting']['f1_diff'],
            'Training_Time': result['training_time']
        }
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values(metric.replace('-', '_').title(), ascending=False)
    
    # Mostrar tabla
    print(f"\n📊 Tabla Comparativa (ordenada por {metric}):\n")
    print(df_comparison.to_string(index=False))
    
    # Identificar mejor modelo
    best_model_name = df_comparison.iloc[0]['Model']
    print(f"\n🏆 Mejor Modelo: {best_model_name}")
    print(f"   {metric}: {df_comparison.iloc[0][metric.replace('-', '_').title()]:.4f}")
    
    # Visualizaciones
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Comparación de métricas principales
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        df_plot = df_comparison[['Model'] + metrics_to_plot].set_index('Model')
        
        df_plot.plot(kind='bar', ax=axes[0, 0], rot=45)
        axes[0, 0].set_title('Comparación de Métricas Principales', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # 2. ROC-AUC comparison
        df_comparison_sorted = df_comparison.sort_values('ROC-AUC', ascending=True)
        axes[0, 1].barh(df_comparison_sorted['Model'], df_comparison_sorted['ROC-AUC'], color='skyblue')
        axes[0, 1].set_title('Comparación de ROC-AUC', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('ROC-AUC Score')
        axes[0, 1].grid(axis='x', alpha=0.3)
        axes[0, 1].set_xlim([0, 1])
        
        # 3. Overfitting check
        axes[1, 0].bar(df_comparison['Model'], df_comparison['Overfitting_F1'], 
                      color=['red' if x > 0.1 else 'green' for x in df_comparison['Overfitting_F1']])
        axes[1, 0].set_title('Overfitting Check (F1 Train - Test)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Diferencia F1')
        axes[1, 0].axhline(y=0.1, color='orange', linestyle='--', label='Umbral (0.1)')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Training time
        axes[1, 1].bar(df_comparison['Model'], df_comparison['Training_Time'], color='coral')
        axes[1, 1].set_title('Tiempo de Entrenamiento', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Segundos')
        axes[1, 1].grid(axis='y', alpha=0.3)
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Gráfico guardado: model_comparison.png")
        plt.show()
    
    return df_comparison


# ============================================================================
# 4. VISUALIZACIONES
# ============================================================================

def plot_roc_curves(results_dict, y_test):
    """
    Grafica curvas ROC de múltiples modelos en un solo gráfico.
    
    Args:
        results_dict: Diccionario con resultados de cada modelo
        y_test: Valores reales de test
    """
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for idx, (model_name, result) in enumerate(results_dict.items()):
        if result['y_test_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['y_test_proba'])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Curvas ROC - Comparación de Modelos', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado: roc_curves_comparison.png")
    plt.show()


def plot_confusion_matrices(results_dict, y_test, figsize=(18, 10)):
    """
    Grafica matrices de confusión de múltiples modelos.
    
    Args:
        results_dict: Diccionario con resultados de cada modelo
        y_test: Valores reales de test
        figsize: Tamaño de la figura
    """
    
    n_models = len(results_dict)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel() if n_models > 1 else [axes]
    
    for idx, (model_name, result) in enumerate(results_dict.items()):
        cm = confusion_matrix(y_test, result['y_test_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar=False, square=True, linewidths=1, linecolor='black')
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Real')
        axes[idx].set_xlabel('Predicho')
        axes[idx].set_xticklabels(['No (0)', 'Sí (1)'])
        axes[idx].set_yticklabels(['No (0)', 'Sí (1)'])
    
    # Ocultar ejes sobrantes
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado: confusion_matrices.png")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, model_name="Model"):
    """
    Grafica la importancia de features para modelos que lo soporten.
    
    Args:
        model: Modelo entrenado
        feature_names: Lista de nombres de features
        top_n: Número de features más importantes a mostrar
        model_name: Nombre del modelo
    """
    
    # Obtener importancia de features
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print(f"⚠️ {model_name} no soporta feature importance")
        return
    
    # Crear DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Graficar
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'], color='teal')
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
    plt.xlabel('Importancia', fontsize=12)
    plt.title(f'Top {top_n} Features Más Importantes - {model_name}', 
             fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.replace(" ", "_")}.png', 
               dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado: feature_importance_{model_name.replace(' ', '_')}.png")
    plt.show()
    
    return feature_importance_df


# ============================================================================
# 5. GUARDAR MODELO
# ============================================================================

def save_best_model(model, model_name, metrics, preprocessor=None, 
                   feature_names=None, filename='best_model.pkl'):
    """
    Guarda el mejor modelo con metadata.
    
    Args:
        model: Modelo entrenado
        model_name: Nombre del modelo
        metrics: Diccionario con métricas
        preprocessor: Pipeline de preprocesamiento (opcional)
        feature_names: Lista de nombres de features (opcional)
        filename: Nombre del archivo de salida
    """
    
    model_package = {
        'model': model,
        'model_name': model_name,
        'metrics': metrics,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    joblib.dump(model_package, filename)
    print(f"\n✅ Modelo guardado: {filename}")
    print(f"   Modelo: {model_name}")
    print(f"   F1-Score: {metrics.get('f1', metrics.get('f1_score', 'N/A')):.4f}")
    print(f"   ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}" if 'roc_auc' in metrics else "")
    
    return filename


# ============================================================================
# 6. FUNCIÓN PRINCIPAL - ENTRENAR MÚLTIPLES MODELOS
# ============================================================================

def train_multiple_models(X_train, y_train, X_test, y_test, 
                         feature_names=None, cv_folds=5, verbose=True):
    """
    Entrena y evalúa múltiples modelos de clasificación.
    
    Args:
        X_train: Features de entrenamiento (ya transformadas)
        y_train: Target de entrenamiento
        X_test: Features de prueba (ya transformadas)
        y_test: Target de prueba
        feature_names: Lista de nombres de features
        cv_folds: Número de folds para validación cruzada
        verbose: Si True, imprime información detallada
        
    Returns:
        dict con resultados de todos los modelos
    """
    
    print(f"\n{'='*80}")
    print(f"🚀 ENTRENAMIENTO DE MÚLTIPLES MODELOS")
    print(f"{'='*80}")
    print(f"Train set: {X_train.shape}")
    print(f"Test set:  {X_test.shape}")
    print(f"CV folds:  {cv_folds}")
    
    # Definir modelos
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    }
    
    # Agregar XGBoost si está disponible
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, 
                                         scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                                         use_label_encoder=False, eval_metric='logloss')
    
    # Agregar LightGBM si está disponible
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMClassifier(n_estimators=100, random_state=42, 
                                           class_weight='balanced', verbose=-1)
    
    # Entrenar todos los modelos
    results = {}
    
    for model_name, model in models.items():
        try:
            result = build_model(
                model, X_train, y_train, X_test, y_test,
                model_name=model_name, cv_folds=cv_folds, verbose=verbose
            )
            results[model_name] = result
        except Exception as e:
            print(f"\n❌ Error entrenando {model_name}: {str(e)}")
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ Entrenamiento completado: {len(results)} modelos")
    print(f"{'='*80}")
    
    return results


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EJEMPLO DE USO - MODEL TRAINING & EVALUATION")
    print("="*80)
    
    # Este ejemplo requiere que primero se ejecute ft_engineering.py
    print("\n⚠️ Este script requiere datos preprocesados de ft_engineering.py")
    print("Ejecutar primero: python ft_engineering.py")
    
    # Ejemplo de cómo usar las funciones:
    """
    from ft_engineering import run_feature_engineering_pipeline
    from model_training_evaluation import train_multiple_models, compare_models
    
    # 1. Preparar datos
    results_fe = run_feature_engineering_pipeline(
        data_path='../../Base_de_datos.csv',
        verbose=True
    )
    
    # 2. Transformar datos
    preprocessor = results_fe['preprocessor']
    X_train_transformed = preprocessor.fit_transform(results_fe['X_train'])
    X_test_transformed = preprocessor.transform(results_fe['X_test'])
    
    # 3. Entrenar modelos
    results_models = train_multiple_models(
        X_train_transformed, results_fe['y_train'],
        X_test_transformed, results_fe['y_test'],
        cv_folds=5, verbose=True
    )
    
    # 4. Comparar modelos
    comparison = compare_models(results_models, metric='f1', show_plot=True)
    
    # 5. Visualizaciones
    plot_roc_curves(results_models, results_fe['y_test'])
    plot_confusion_matrices(results_models, results_fe['y_test'])
    
    # 6. Guardar mejor modelo
    best_model_name = comparison.iloc[0]['Model']
    best_result = results_models[best_model_name]
    save_best_model(
        best_result['model'], 
        best_model_name,
        best_result['test_metrics'],
        preprocessor=preprocessor,
        filename='best_model.pkl'
    )
    """

