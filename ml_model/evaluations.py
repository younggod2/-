import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix, average_precision_score, precision_recall_curve, f1_score,matthews_corrcoef
from joblib import load

def plot_classification_performance(model, X_test, y_test):
    # Предсказание
    y_pred = model.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Average precision
    average_precision = average_precision_score(y_test, y_pred)
    
    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    
    # F1 score
    f1 = f1_score(y_test, y_pred)
    
    # Matthews correlation coefficient
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Построение confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Истинные значения')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Построение precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Average Precision = {average_precision:.2f})')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()
    
    # Вывод метрик F1 и MCC
    print(f'F1 Score: {f1:.2f}')
    print(f'Matthews Correlation Coefficient: {mcc:.2f}')


def plot_regression_performance(model, X_test, y_test):
    # Предсказание
    y_pred = model.predict(X_test)
    
    # Вычисление метрик
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Вычисление коэффициента корреляции Пирсона
    pearson_corr = np.corrcoef(y_test, y_pred)[0, 1]
    
    # Создание DataFrame для Seaborn
    df = pd.DataFrame({'Истинные ∆∆G': y_test, 'Предсказанные ∆∆G': y_pred})
    
    # Установка стиля графика
    sns.set_style("darkgrid")
    
    # Построение графика
    sns.scatterplot(data=df, x='Истинные ∆∆G', y='Предсказанные ∆∆G')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.xlabel('Истинные ∆∆G')
    plt.ylabel('Предсказанные ∆∆G')
    plt.title(f'R^2 = {r2:.2f}, RMSE = {rmse:.2f}, Pearson = {pearson_corr:.2f}')
    plt.show()
