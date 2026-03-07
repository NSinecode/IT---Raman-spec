# datagather.py
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.signal import savgol_filter

# --------------------------
# Функции обработки спектров
# --------------------------
def assign_wave_category(filename):
    """Определяем категорию по centerXXX в имени файла."""
    match = re.search(r"center(\d+)", filename)
    if match:
        val = int(match.group(1))
        if val == 1500:
            return "low"
        elif val == 2900:
            return "high"
    return "unknown"

def apply_preprocessing(intensity_array):
    """Очистка, сглаживание и нормализация спектра."""
    # Spike removal
    median_filtered = np.copy(intensity_array)
    std_diff = np.std(np.diff(intensity_array))
    for i in range(1, len(intensity_array)-1):
        if abs(intensity_array[i] - intensity_array[i-1]) > 5 * std_diff:
            median_filtered[i] = (median_filtered[i-1] + median_filtered[i+1]) / 2
    
    # Savgol фильтр
    smoothed = savgol_filter(median_filtered, window_length=21, polyorder=3)
    
    # Простое baseline correction (минимум по окну)
    window_size = 200
    base = np.zeros_like(smoothed)
    for i in range(len(smoothed)):
        start = max(0, i - window_size//2)
        end = min(len(smoothed), i + window_size//2)
        base[i] = np.min(smoothed[start:end])
    corrected = smoothed - base
    corrected = np.maximum(corrected, 0)
    
    # Нормализация 0-1
    c_min, c_max = corrected.min(), corrected.max()
    if c_max > c_min:
        corrected = (corrected - c_min) / (c_max - c_min)
    return corrected

def process_single_file(path, cls_name, wave_grid):
    """Обработка одного файла: чтение, интерполяция, предобработка."""
    filename = os.path.basename(path)
    brain_region = re.match(r"([a-zA-Z]+)_", filename)
    brain_region = brain_region.group(1).lower() if brain_region else "unknown"
    wave_category = assign_wave_category(filename)
    
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python", comment="#",
                         names=["x","y","wave","intensity"], on_bad_lines="skip")
        if df.empty:
            return None
        df = df.sort_values("wave").drop_duplicates(subset=["wave"])
        interp_intensity = np.interp(wave_grid, df["wave"], df["intensity"])
        processed_intensity = apply_preprocessing(interp_intensity)
        row = {f"wave{int(w)}": v for w,v in zip(wave_grid, processed_intensity)}
        row.update({
            "class": cls_name,
            "brain_region": brain_region,
            "wave_category": wave_category,
            "x": df["x"].iloc[0],
            "y": df["y"].iloc[0]
        })
        return row
    except Exception as e:
        print(f"❌ Ошибка в {filename}: {e}")
        return None

# --------------------------
# Главная функция загрузки
# --------------------------
def load_and_preprocess(root_directory, is_training=True, n_jobs=-1):
    """Загрузка и предобработка всех спектров в папках control/endo/exo."""
    output_dir = os.path.join(root_directory, "processed")
    os.makedirs(output_dir, exist_ok=True)
    cache_name = "train_cache.feather" if is_training else "new_data_cache.feather"
    cache_path = os.path.join(output_dir, cache_name)
    
    if os.path.exists(cache_path):
        print(f"✅ Найден кэш: {cache_path}. Загружаем мгновенно...")
        return pd.read_feather(cache_path)
    
    classes = ["control", "endo", "exo"]
    wave_grid = np.arange(900, 3501)
    all_files = []

    for cls in classes:
        cls_folder = os.path.join(root_directory, cls)
        if not os.path.exists(cls_folder):
            continue
        for root, _, files in os.walk(cls_folder):
            for f in files:
                if f.endswith(".txt"):
                    all_files.append((os.path.join(root,f), cls))
    
    if not all_files:
        print(f"⚠️ Файлы не найдены в {root_directory}")
        return pd.DataFrame()
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_file)(p, c, wave_grid) for p,c in tqdm(all_files, desc="Processing files")
    )
    
    final_df = pd.DataFrame([r for r in results if r is not None])
    if not final_df.empty:
        final_df.reset_index(drop=True, inplace=True)
        try:
            final_df.to_feather(cache_path)
            print(f"💾 Данные кэшированы: {cache_path}")
        except Exception as e:
            print(f"⚠️ Не удалось сохранить кэш: {e}")
    return final_df

# --------------------------
# Удобные функции для вызова
# --------------------------
def get_train_df(root_directory):
    return load_and_preprocess(root_directory, is_training=True)

def get_new_df(root_directory):
    return load_and_preprocess(root_directory, is_training=False)

# --------------------------
# Если запуск напрямую
# --------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and preprocess Raman spectra")
    parser.add_argument("data_dir", help="Путь к корневой папке с control/endo/exo")
    args = parser.parse_args()
    
    df = get_train_df(args.data_dir)
    print(df.head())