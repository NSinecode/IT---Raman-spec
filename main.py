import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from model.training import train_or_load_model
from preprocessing.datagather import get_train_df, get_new_df
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    TRAINING_PATH = input('Введите папку в которой находится обучающая выборка: ')
    NEW_PATH = input('Введите папку в которой находится тестовая выборка: ')

    # 1. Обучаем модель
    df_train = get_train_df(TRAINING_PATH)
    print(df_train.head())

    model, class_names = train_or_load_model(df_train)

    # 2. Загружаем новые данные (без class)
    df_new = get_new_df(NEW_PATH)

    intensity_cols = [col for col in df_new.columns if col.startswith('intensity')]
    categorical_cols = ['brain_region', 'wave_category']

    # 1. Признаки и целевая переменная
    X = df_new.drop(columns=["x", "y"], errors='ignore')
    
    # One-Hot Encoding для категориальных признаков
    cat_cols = [col for col in X.columns if col in ["brain_region", "wave_category"]]
    X = pd.get_dummies(X, columns=cat_cols)
    
    # Заполняем NaN средним
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_new = scaler.fit_transform(X)

    # 3. Предсказания
    predictions_idx = model.predict(X_new)
    predictions_labels = [class_names[i] for i in predictions_idx]

    probabilities = model.predict_proba(X_new)

    # 4. Таблица результатов
    results = pd.DataFrame({
        'Brain_Region': df_new['brain_region'],
        'Predicted_Class': predictions_labels,
        'Confidence': [max(p) for p in probabilities]
    })

    print("\nPredictions:")
    print(results)

    # 5. Если в новых данных есть class — считаем метрики
    if 'class' in df_new.columns:

        y_true = df_new['class']
        y_pred = predictions_labels

        print("\nAccuracy:")
        print(accuracy_score(y_true, y_pred))

        print("\nClassification Report:")
        report = classification_report(y_true, y_pred)
        print(report)

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)