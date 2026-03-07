import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model.training import train_or_load_rf_model # Исправил название функции согласно вашему файлу
from preprocessing.datagather import get_train_df, get_new_df

if __name__ == '__main__':
    TRAINING_PATH = input('Введите папку с обучающей выборкой: ')
    NEW_PATH = input('Введите папку с новыми данными: ')

    # 1. Обучаем модель (или загружаем)
    df_train = get_train_df(TRAINING_PATH)
    model, class_names = train_or_load_rf_model(df_train)
    
    # ВАЖНО: Для корректного препроцессинга нам нужны колонки из обучения
    # Получаем структуру X_train (без 'class', 'x', 'y')
    X_train_template = pd.get_dummies(df_train.drop(columns=["class", "x", "y"], errors='ignore'))
    train_columns = X_train_template.columns

    # 2. Загружаем новые данные
    df_new = get_new_df(NEW_PATH)

    # 3. ПРЕПРОЦЕССИНГ (идентичный тренировочному)
    # Убираем лишнее
    X_new = df_new.drop(columns=["x", "y"], errors='ignore')
    
    # One-Hot Encoding
    cat_cols = [col for col in X_new.columns if col in ["brain_region", "wave_category"]]
    X_new = pd.get_dummies(X_new, columns=cat_cols)
    
    # --- СИНХРОНИЗАЦИЯ КОЛОНОК ---
    # Добавляем недостающие колонки (которых нет в new, но были в train) заполняя нулями
    for col in train_columns:
        if col not in X_new.columns:
            X_new[col] = 0
            
    # Убираем лишние колонки (если в new появились новые категории, которых не было в train)
    # И выстраиваем колонки в том же порядке, что и в train
    X_new = X_new[train_columns]
    
    # Заполняем NaN средним (лучше использовать средние из train, но для простоты оставим так)
    X_new = X_new.fillna(X_new.mean())

    # 4. СТАНДАРТИЗАЦИЯ
    # В идеале Scaler нужно сохранять в pkl вместе с моделью. 
    # Если в training.py он не сохраняется отдельно, создаем новый и обучаем на TRAIN
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Обучаем на тренировочных (нужно повторить логику из training.py)
    X_train_data = pd.get_dummies(df_train.drop(columns=["class", "x", "y"], errors='ignore'))
    scaler.fit(X_train_data) 
    
    X_new_scaled = scaler.transform(X_new)

    # 5. ПРЕДСКАЗАНИЯ
    predictions_idx = model.predict(X_new_scaled)
    # Если в class_names индексы строк, используем их напрямую
    predictions_labels = predictions_idx 

    probabilities = model.predict_proba(X_new_scaled)

    # 6. РЕЗУЛЬТАТЫ
    results = pd.DataFrame({
        'Brain_Region': df_new['brain_region'] if 'brain_region' in df_new.columns else "N/A",
        'Predicted_Class': predictions_labels,
        'Confidence': [max(p) for p in probabilities]
    })

    print("\nPredictions:")
    print(results.head(20))

    # 7. Метрики (если 'class' случайно есть в new)
    if 'class' in df_new.columns:
        print("\n--- Metrics found in new data ---")
        y_true = df_new['class']
        print(f"Accuracy: {accuracy_score(y_true, predictions_labels):.4f}")
        print(classification_report(y_true, predictions_labels))