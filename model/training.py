# training.py
import os
import joblib
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_or_load_rf_model(df, model_path="rf_raman_model.pkl", force_train=False, n_trees=300):
    """
    Тренирует RandomForestClassifier с warm_start и прогресс-баром.
    Поддерживает категориальные признаки brain_region и wave_category.
    """
    if os.path.exists(model_path) and not force_train:
        print(f"✅ Найдена сохраненная модель: {model_path}. Загрузка...")
        data = joblib.load(model_path)
        return data['model'], data['class_names']

    print("🚀 Начинаем тренировку RandomForest...")
    
    # 1. Признаки и целевая переменная
    X = df.drop(columns=["class", "x", "y"], errors='ignore')
    y = df["class"]
    
    # One-Hot Encoding для категориальных признаков
    cat_cols = [col for col in X.columns if col in ["brain_region", "wave_category"]]
    X = pd.get_dummies(X, columns=cat_cols)
    
    # Заполняем NaN средним
    X = X.fillna(X.mean())
    
    # 2. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # 3. Стандартизация (необязательно для RandomForest, но оставим)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. RandomForest с warm_start
    clf = RandomForestClassifier(
        n_estimators=1,
        warm_start=True,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training RandomForest...")
    for i in tqdm(range(1, n_trees + 1), desc="Training trees"):
        clf.set_params(n_estimators=i)
        clf.fit(X_train_scaled, y_train)
    
    # 5. Предсказания
    y_pred = clf.predict(X_test_scaled)
    
    print("\n✅ Отчет по тестовой выборке:")
    print(classification_report(y_test, y_pred))
    
    # 6. Важность признаков
    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    print("\nTop 20 important features:")
    print(feat_imp.head(20))
    
    # 7. Сохранение модели
    joblib.dump({'model': clf, 'class_names': y.unique().tolist()}, model_path)
    print(f"\n💾 Модель сохранена в {model_path}")
    
    return clf, y.unique().tolist()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RandomForest on Raman spectra")
    parser.add_argument("data_file", help="Путь к DataFrame CSV/Feather с обработанными спектрами")
    parser.add_argument("--model_path", default="rf_raman_model.pkl", help="Куда сохранять модель")
    parser.add_argument("--force_train", action="store_true", help="Принудительно переобучить модель")
    parser.add_argument("--n_trees", type=int, default=300, help="Количество деревьев RandomForest")
    args = parser.parse_args()
    
    # Загружаем DataFrame
    if args.data_file.endswith(".csv"):
        df = pd.read_csv(args.data_file)
    else:
        df = pd.read_feather(args.data_file)
    
    model, classes = train_or_load_rf_model(
        df, 
        model_path=args.model_path, 
        force_train=args.force_train, 
        n_trees=args.n_trees
    )