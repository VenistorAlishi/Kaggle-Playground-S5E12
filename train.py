"""
Kaggle Playground Series S5E12 - Diabetes Prediction
Решение с использованием LightGBM, CatBoost, XGBoost и ансамблирования
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import warnings
import time
import sys
import os
import gc
from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Отключаем буферизацию вывода для реального времени
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except:
        pass

# Константы
RANDOM_STATE = 42
N_FOLDS = 5  # Возвращаемся к 5 фолдам для баланса между стабильностью и временем

# Файлы для сохранения промежуточных результатов
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
LGB_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'lgb_predictions.npy')
CAT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'cat_predictions.npy')
XGB_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'xgb_predictions.npy')
TEST_IDS_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'test_ids.npy')

print("=" * 60, flush=True)
print("Kaggle Playground Series S5E12 - Diabetes Prediction", flush=True)
print("=" * 60, flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"Working directory: {os.getcwd()}", flush=True)
sys.stdout.flush()

# Загрузка данных с оптимизацией памяти
print("\n[1/6] Загрузка данных...", flush=True)
sys.stdout.flush()

print("Loading train.csv (79 MB, ~700K rows)...", flush=True)
sys.stdout.flush()

# Оптимизированная загрузка с указанием типов данных для экономии памяти
dtype_dict = {
    'id': 'int32',
    'age': 'int8',
    'alcohol_consumption_per_week': 'int8',
    'physical_activity_minutes_per_week': 'int16',
    'diet_score': 'float32',
    'sleep_hours_per_day': 'float32',
    'screen_time_hours_per_day': 'float32',
    'bmi': 'float32',
    'waist_to_hip_ratio': 'float32',
    'systolic_bp': 'int16',
    'diastolic_bp': 'int16',
    'heart_rate': 'int16',
    'cholesterol_total': 'int16',
    'hdl_cholesterol': 'int16',
    'ldl_cholesterol': 'int16',
    'triglycerides': 'int16',
    'family_history_diabetes': 'int8',
    'hypertension_history': 'int8',
    'cardiovascular_history': 'int8',
    'diagnosed_diabetes': 'float32'
}

try:
    train = pd.read_csv('train.csv', dtype=dtype_dict, low_memory=False, engine='c')
    print(f"Train loaded: {train.shape} | Memory: {train.memory_usage(deep=True).sum() / 1024**2:.1f} MB", flush=True)
    sys.stdout.flush()
except Exception as e:
    print(f"Error loading train.csv: {e}", flush=True)
    raise

# Для test убираем diagnosed_diabetes из dtype
test_dtype = {k: v for k, v in dtype_dict.items() if k != 'diagnosed_diabetes'}
print("Loading test.csv (33 MB, ~300K rows)...", flush=True)
sys.stdout.flush()

try:
    test = pd.read_csv('test.csv', dtype=test_dtype, low_memory=False, engine='c')
    print(f"Test loaded: {test.shape} | Memory: {test.memory_usage(deep=True).sum() / 1024**2:.1f} MB", flush=True)
    sys.stdout.flush()
except Exception as e:
    print(f"Error loading test.csv: {e}", flush=True)
    raise

# Разделение на features и target
target = 'diagnosed_diabetes'
X_train = train.drop([target, 'id'], axis=1)
y_train = train[target]
X_test = test.drop('id', axis=1)
test_ids = test['id']

print(f"\nTarget distribution:")
print(y_train.value_counts())
print(f"Target ratio: {y_train.mean():.4f}")

# Определение типов признаков
categorical_features = ['gender', 'ethnicity', 'education_level', 'income_level', 
                       'smoking_status', 'employment_status']
numerical_features = [col for col in X_train.columns if col not in categorical_features]

print(f"\nCategorical features: {len(categorical_features)}")
print(f"Numerical features: {len(numerical_features)}")

# Feature Engineering
print("\n[2/6] Feature Engineering...")

def create_features(df, train_stats=None, y_train=None):
    """Создание новых признаков с улучшенным feature engineering
    
    Args:
        df: DataFrame для обработки
        train_stats: словарь со статистиками train (mean, std) для нормализации
        y_train: целевая переменная для target encoding (только для train)
    """
    df = df.copy()
    
    # BMI категории (более детальные)
    df['bmi_category'] = pd.cut(df['bmi'], 
                                bins=[0, 18.5, 25, 30, 35, 100], 
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese1', 'Obese2'])
    df['bmi_category'] = df['bmi_category'].astype(str)
    
    # Возрастные группы (более детальные)
    df['age_group'] = pd.cut(df['age'], 
                             bins=[0, 25, 35, 45, 55, 65, 100], 
                             labels=['VeryYoung', 'Young', 'Middle', 'Senior', 'Elderly', 'VeryElderly'])
    df['age_group'] = df['age_group'].astype(str)
    
    # Комбинированные признаки здоровья
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1e-5)
    df['bp_mean'] = (df['systolic_bp'] + df['diastolic_bp']) / 2
    df['bp_pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['cholesterol_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)
    df['total_cholesterol_hdl'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
    df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)
    df['triglycerides_hdl'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-5)
    df['cholesterol_balance'] = df['hdl_cholesterol'] - df['ldl_cholesterol']
    
    # Индексы здоровья (улучшенные)
    df['health_score'] = (df['diet_score'] + 
                         (df['physical_activity_minutes_per_week'] / 100) + 
                         (df['sleep_hours_per_day'] / 8))
    
    df['lifestyle_score'] = (df['diet_score'] * 0.4 + 
                             (df['physical_activity_minutes_per_week'] / 200) * 0.3 +
                             (df['sleep_hours_per_day'] / 8) * 0.3)
    
    df['comprehensive_health_score'] = (
        df['diet_score'] * 0.3 +
        (df['physical_activity_minutes_per_week'] / 300) * 0.25 +
        (df['sleep_hours_per_day'] / 8) * 0.2 +
        (1 - df['screen_time_hours_per_day'] / 12) * 0.15 +
        (1 - df['alcohol_consumption_per_week'] / 14) * 0.1
    )
    
    df['risk_factors_count'] = (df['family_history_diabetes'] + 
                                df['hypertension_history'] + 
                                df['cardiovascular_history'])
    
    # Расширенные взаимодействия
    df['bmi_age'] = df['bmi'] * df['age']
    df['bmi_waist_hip'] = df['bmi'] * df['waist_to_hip_ratio']
    df['bmi_systolic'] = df['bmi'] * df['systolic_bp']
    df['bmi_diastolic'] = df['bmi'] * df['diastolic_bp']
    df['bmi_cholesterol'] = df['bmi'] * df['cholesterol_total']
    df['bmi_heart_rate'] = df['bmi'] * df['heart_rate']
    df['activity_sleep'] = df['physical_activity_minutes_per_week'] * df['sleep_hours_per_day']
    df['age_activity'] = df['age'] * df['physical_activity_minutes_per_week']
    df['age_bp'] = df['age'] * df['systolic_bp']
    df['age_cholesterol'] = df['age'] * df['cholesterol_total']
    df['waist_hip_age'] = df['waist_to_hip_ratio'] * df['age']
    df['diet_activity'] = df['diet_score'] * (df['physical_activity_minutes_per_week'] / 100)
    df['sleep_diet'] = df['sleep_hours_per_day'] * df['diet_score']
    
    # Логические признаки (расширенные)
    df['high_bp'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 85)).astype(int)
    df['stage1_hypertension'] = ((df['systolic_bp'] >= 130) & (df['systolic_bp'] < 140)).astype(int)
    df['stage2_hypertension'] = (df['systolic_bp'] >= 140).astype(int)
    df['high_cholesterol'] = (df['cholesterol_total'] >= 200).astype(int)
    df['very_high_cholesterol'] = (df['cholesterol_total'] >= 240).astype(int)
    df['high_ldl'] = (df['ldl_cholesterol'] >= 100).astype(int)
    df['very_high_ldl'] = (df['ldl_cholesterol'] >= 160).astype(int)
    df['low_hdl'] = (df['hdl_cholesterol'] < 40).astype(int)
    df['very_low_hdl'] = (df['hdl_cholesterol'] < 35).astype(int)
    df['high_triglycerides'] = (df['triglycerides'] >= 150).astype(int)
    df['very_high_triglycerides'] = (df['triglycerides'] >= 200).astype(int)
    df['obese'] = (df['bmi'] >= 30).astype(int)
    df['severely_obese'] = (df['bmi'] >= 35).astype(int)
    df['overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
    df['high_waist_hip'] = (df['waist_to_hip_ratio'] > 0.9).astype(int)
    df['very_high_waist_hip'] = (df['waist_to_hip_ratio'] > 0.95).astype(int)
    df['low_activity'] = (df['physical_activity_minutes_per_week'] < 150).astype(int)
    df['poor_sleep'] = ((df['sleep_hours_per_day'] < 6) | (df['sleep_hours_per_day'] > 9)).astype(int)
    df['high_screen_time'] = (df['screen_time_hours_per_day'] > 6).astype(int)
    df['high_alcohol'] = (df['alcohol_consumption_per_week'] > 7).astype(int)
    
    # Метаболические признаки (расширенные)
    df['metabolic_risk'] = (df['high_bp'].astype(int) + 
                            df['high_cholesterol'].astype(int) + 
                            df['high_triglycerides'].astype(int) + 
                            df['obese'].astype(int))
    
    df['cardiovascular_risk'] = (df['high_bp'].astype(int) + 
                                 df['high_cholesterol'].astype(int) + 
                                 df['high_ldl'].astype(int) + 
                                 df['low_hdl'].astype(int) + 
                                 df['obese'].astype(int))
    
    df['diabetes_risk_score'] = (
        df['obese'].astype(int) * 2 +
        df['overweight'].astype(int) * 1 +
        df['high_bp'].astype(int) * 1.5 +
        df['high_cholesterol'].astype(int) * 1 +
        df['low_hdl'].astype(int) * 1 +
        df['high_triglycerides'].astype(int) * 1 +
        df['family_history_diabetes'] * 2 +
        df['hypertension_history'] * 1.5 +
        df['cardiovascular_history'] * 1
    )
    
    # Нормализованные признаки (используем статистики train для test)
    if train_stats is None:
        # Вычисляем статистики для train
        age_mean, age_std = df['age'].mean(), df['age'].std()
        bmi_mean, bmi_std = df['bmi'].mean(), df['bmi'].std()
        bp_mean, bp_std = df['systolic_bp'].mean(), df['systolic_bp'].std()
        chol_mean, chol_std = df['cholesterol_total'].mean(), df['cholesterol_total'].std()
        train_stats = {
            'age': (age_mean, age_std), 
            'bmi': (bmi_mean, bmi_std),
            'bp': (bp_mean, bp_std),
            'cholesterol': (chol_mean, chol_std)
        }
    else:
        # Используем статистики train для test
        age_mean, age_std = train_stats['age']
        bmi_mean, bmi_std = train_stats['bmi']
        bp_mean, bp_std = train_stats['bp']
        chol_mean, chol_std = train_stats['cholesterol']
    
    df['age_normalized'] = (df['age'] - age_mean) / (age_std + 1e-5)
    df['bmi_normalized'] = (df['bmi'] - bmi_mean) / (bmi_std + 1e-5)
    df['bp_normalized'] = (df['systolic_bp'] - bp_mean) / (bp_std + 1e-5)
    df['cholesterol_normalized'] = (df['cholesterol_total'] - chol_mean) / (chol_std + 1e-5)
    
    # Полиномиальные признаки (расширенные)
    df['age_squared'] = df['age'] ** 2
    df['bmi_squared'] = df['bmi'] ** 2
    df['age_cubed'] = df['age'] ** 3
    df['bmi_cubed'] = df['bmi'] ** 3
    df['bp_squared'] = df['systolic_bp'] ** 2
    
    # Логарифмические признаки (для skewed distributions)
    df['log_bmi'] = np.log1p(df['bmi'])
    df['log_age'] = np.log1p(df['age'])
    df['log_cholesterol'] = np.log1p(df['cholesterol_total'])
    df['log_triglycerides'] = np.log1p(df['triglycerides'])
    df['log_activity'] = np.log1p(df['physical_activity_minutes_per_week'] + 1)
    
    # Квантильные признаки
    if train_stats is None:
        # Для train вычисляем квантили
        df['bmi_quantile'] = pd.qcut(df['bmi'], q=5, labels=False, duplicates='drop')
        df['age_quantile'] = pd.qcut(df['age'], q=5, labels=False, duplicates='drop')
        df['cholesterol_quantile'] = pd.qcut(df['cholesterol_total'], q=5, labels=False, duplicates='drop')
    else:
        # Для test используем квантили из train (если сохранены)
        if 'bmi_quantiles' in train_stats:
            df['bmi_quantile'] = pd.cut(df['bmi'], bins=train_stats['bmi_quantiles'], labels=False, include_lowest=True)
            df['age_quantile'] = pd.cut(df['age'], bins=train_stats['age_quantiles'], labels=False, include_lowest=True)
            df['cholesterol_quantile'] = pd.cut(df['cholesterol_total'], bins=train_stats['cholesterol_quantiles'], labels=False, include_lowest=True)
        else:
            df['bmi_quantile'] = 0
            df['age_quantile'] = 0
            df['cholesterol_quantile'] = 0
    
    # Сохраняем квантили для test
    if train_stats is not None and 'bmi_quantiles' not in train_stats:
        pass  # Квантили уже должны быть сохранены
    
    return df, train_stats

# Создаем признаки для train и сохраняем статистики
X_train, train_stats = create_features(X_train, y_train=y_train)
# Сохраняем квантили для использования в test
if 'bmi_quantiles' not in train_stats:
    train_stats['bmi_quantiles'] = pd.qcut(X_train['bmi'], q=5, retbins=True, duplicates='drop')[1]
    train_stats['age_quantiles'] = pd.qcut(X_train['age'], q=5, retbins=True, duplicates='drop')[1]
    train_stats['cholesterol_quantiles'] = pd.qcut(X_train['cholesterol_total'], q=5, retbins=True, duplicates='drop')[1]
# Создаем признаки для test используя статистики train
X_test, _ = create_features(X_test, train_stats=train_stats)

# Обновление списков признаков
categorical_features.extend(['bmi_category', 'age_group'])
new_numerical = [
    'bp_ratio', 'bp_mean', 'bp_pulse_pressure', 'cholesterol_ratio', 'total_cholesterol_hdl', 
    'non_hdl_cholesterol', 'ldl_hdl_ratio', 'triglycerides_hdl', 'cholesterol_balance',
    'health_score', 'lifestyle_score', 'comprehensive_health_score', 'risk_factors_count',
    'bmi_age', 'bmi_waist_hip', 'bmi_systolic', 'bmi_diastolic', 'bmi_cholesterol', 'bmi_heart_rate',
    'activity_sleep', 'age_activity', 'age_bp', 'age_cholesterol', 'waist_hip_age',
    'diet_activity', 'sleep_diet',
    'high_bp', 'stage1_hypertension', 'stage2_hypertension', 'high_cholesterol', 'very_high_cholesterol',
    'high_ldl', 'very_high_ldl', 'low_hdl', 'very_low_hdl', 'high_triglycerides', 'very_high_triglycerides',
    'obese', 'severely_obese', 'overweight', 'high_waist_hip', 'very_high_waist_hip',
    'low_activity', 'poor_sleep', 'high_screen_time', 'high_alcohol',
    'metabolic_risk', 'cardiovascular_risk', 'diabetes_risk_score',
    'age_normalized', 'bmi_normalized', 'bp_normalized', 'cholesterol_normalized',
    'age_squared', 'bmi_squared', 'age_cubed', 'bmi_cubed', 'bp_squared',
    'log_bmi', 'log_age', 'log_cholesterol', 'log_triglycerides', 'log_activity',
    'bmi_quantile', 'age_quantile', 'cholesterol_quantile'
]
numerical_features.extend(new_numerical)

print(f"Total features after engineering: {len(X_train.columns)}")

# Подготовка данных для моделей
print("\n[3/6] Подготовка данных...")

# Target Encoding для категориальных признаков (очень важно для улучшения результата!)
print("Applying Target Encoding to categorical features...", flush=True)
from sklearn.model_selection import KFold

def target_encode(train_df, test_df, cat_cols, target, n_splits=5, smooth=1.0):
    """Target Encoding с кросс-валидацией для предотвращения переобучения"""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    # Глобальное среднее
    global_mean = target.mean()
    
    # KFold для target encoding
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    for col in cat_cols:
        if col not in train_df.columns:
            continue
            
        # Для train используем кросс-валидацию
        train_encoded[f'{col}_target_enc'] = 0.0
        for train_idx, val_idx in kf.split(train_df):
            # Вычисляем среднее на train fold
            train_mean = target.iloc[train_idx].groupby(train_df[col].iloc[train_idx]).mean()
            train_count = train_df[col].iloc[train_idx].value_counts()
            
            # Применяем к validation fold
            val_values = train_df[col].iloc[val_idx]
            encoded_values = val_values.map(train_mean)
            
            # Smoothing
            counts = val_values.map(train_count)
            encoded_values = (encoded_values * counts + global_mean * smooth) / (counts + smooth)
            encoded_values = encoded_values.fillna(global_mean)
            
            train_encoded.loc[train_df.index[val_idx], f'{col}_target_enc'] = encoded_values
        
        # Для test используем все train данные
        test_mean = target.groupby(train_df[col]).mean()
        test_count = train_df[col].value_counts()
        test_values = test_df[col]
        encoded_test = test_values.map(test_mean)
        counts_test = test_values.map(test_count)
        encoded_test = (encoded_test * counts_test + global_mean * smooth) / (counts_test + smooth)
        encoded_test = encoded_test.fillna(global_mean)
        test_encoded[f'{col}_target_enc'] = encoded_test
    
    return train_encoded, test_encoded

# Применяем target encoding
categorical_features_filtered = [col for col in categorical_features if col in X_train.columns]
X_train_encoded, X_test_encoded = target_encode(
    X_train, X_test, categorical_features_filtered, y_train, n_splits=5, smooth=5.0
)
print(f"Target encoding applied to {len(categorical_features_filtered)} categorical features", flush=True)

# Также сохраняем категориальные признаки как category для LightGBM
for col in categorical_features_filtered:
    if col in X_train_encoded.columns:
        X_train_encoded[col] = X_train_encoded[col].astype('category')
        X_test_encoded[col] = X_test_encoded[col].astype('category')

# Feature Selection - удаляем наименее важные признаки
print("Applying Feature Selection...", flush=True)
from sklearn.ensemble import RandomForestClassifier

# Используем Random Forest для оценки важности признаков
print("  Computing feature importance with Random Forest...", flush=True)
rf_selector = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
rf_selector.fit(X_train_encoded.select_dtypes(include=[np.number]), y_train)

# Получаем важность признаков
feature_importance = pd.DataFrame({
    'feature': X_train_encoded.select_dtypes(include=[np.number]).columns,
    'importance': rf_selector.feature_importances_
}).sort_values('importance', ascending=False)

# Оставляем топ-150 признаков (или все, если меньше)
n_features_to_keep = min(150, len(feature_importance))
top_features = feature_importance.head(n_features_to_keep)['feature'].tolist()

# Добавляем категориальные признаки и target encoded признаки
all_features_to_keep = top_features + categorical_features_filtered + [f'{col}_target_enc' for col in categorical_features_filtered]
all_features_to_keep = [f for f in all_features_to_keep if f in X_train_encoded.columns]

# Фильтруем признаки
X_train_encoded = X_train_encoded[all_features_to_keep]
X_test_encoded = X_test_encoded[all_features_to_keep]

print(f"  Selected {len(all_features_to_keep)} features (from {len(X_train.columns)})", flush=True)
print(f"  Top 10 features: {', '.join(feature_importance.head(10)['feature'].tolist())}", flush=True)

# Кросс-валидация
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Хранение предсказаний
# Проверяем, есть ли сохраненные результаты
load_checkpoint = False
if os.path.exists(LGB_CHECKPOINT) and os.path.exists(TEST_IDS_CHECKPOINT):
    print(f"\n[INFO] Found checkpoint files. Loading previous results...", flush=True)
    try:
        test_predictions_lgb = np.load(LGB_CHECKPOINT)
        test_ids_array = np.load(TEST_IDS_CHECKPOINT)
        test_ids = pd.Series(test_ids_array)
        load_checkpoint = True
        print(f"[INFO] Loaded LightGBM predictions from checkpoint", flush=True)
    except Exception as e:
        print(f"[INFO] Failed to load checkpoint: {e}. Starting fresh.", flush=True)
        load_checkpoint = False

if not load_checkpoint:
    oof_predictions_lgb = np.zeros(len(X_train))
    test_predictions_lgb = np.zeros(len(X_test))
else:
    oof_predictions_lgb = np.zeros(len(X_train))  # OOF всегда пересчитываем

oof_predictions_cat = np.zeros(len(X_train))
oof_predictions_xgb = np.zeros(len(X_train))
test_predictions_cat = np.zeros(len(X_test))
test_predictions_xgb = np.zeros(len(X_test))

print("\n[4/6] Обучение моделей...")
start_time_total = time.time()

# LightGBM
print("\n" + "="*60, flush=True)
print("--- LightGBM ---", flush=True)
print("="*60, flush=True)
lgb_start = time.time()

# Проверяем, нужно ли пропустить LightGBM если есть checkpoint
skip_lgb = False
if load_checkpoint and os.path.exists(LGB_CHECKPOINT):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] LightGBM checkpoint found. Skipping LightGBM training...", flush=True)
    skip_lgb = True
    lgb_auc = 0.0  # Будет пересчитан позже если нужно
else:
    fold_splits = list(skf.split(X_train, y_train))

if not skip_lgb:
    for fold, (train_idx, val_idx) in enumerate(tqdm(fold_splits, desc="LightGBM Folds", ncols=100)):
        fold_start = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fold {fold + 1}/{N_FOLDS} - Training...", flush=True)
        
        X_tr, X_val = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=categorical_features_filtered)
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_features_filtered, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 95,  # Оптимальный баланс сложности и регуляризации
            'learning_rate': 0.015,  # Более консервативный learning rate
            'feature_fraction': 0.75,  # Более агрессивная регуляризация
            'bagging_fraction': 0.75,
            'bagging_freq': 5,
            'min_child_samples': 25,  # Увеличено для предотвращения переобучения
            'min_split_gain': 0.1,  # Увеличено для регуляризации
            'reg_alpha': 0.3,  # Сильная регуляризация
            'reg_lambda': 0.5,  # Сильная регуляризация
            'max_depth': 12,  # Добавлено ограничение глубины
            'min_data_in_leaf': 10,  # Добавлено для регуляризации
            'verbose': -1,
            'random_state': RANDOM_STATE
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2500,  # Оптимальный баланс итераций
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)]
        )
        
        best_iter = model.best_iteration if model.best_iteration is not None else model.num_trees()
        oof_predictions_lgb[val_idx] = model.predict(X_val, num_iteration=best_iter)
        test_predictions_lgb += model.predict(X_test_encoded, num_iteration=best_iter) / N_FOLDS
        
        val_auc = roc_auc_score(y_val, oof_predictions_lgb[val_idx])
        fold_time = time.time() - fold_start
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fold {fold + 1} AUC: {val_auc:.5f} | Time: {fold_time:.1f}s | Best iteration: {best_iter}", flush=True)
        del model, train_data, val_data
        gc.collect()

lgb_auc = roc_auc_score(y_train, oof_predictions_lgb)
lgb_time = time.time() - lgb_start
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] LightGBM OOF AUC: {lgb_auc:.5f} | Total time: {lgb_time/60:.1f} min", flush=True)

# Сохраняем промежуточные результаты LightGBM
print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving LightGBM checkpoint...", flush=True)
np.save(LGB_CHECKPOINT, test_predictions_lgb)
np.save(TEST_IDS_CHECKPOINT, test_ids.values)
print(f"[{datetime.now().strftime('%H:%M:%S')}] LightGBM checkpoint saved", flush=True)

# Создаем промежуточный submission только с LightGBM
try:
    temp_submission = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': test_predictions_lgb
    })
    temp_submission.to_csv('submission_lgb.csv', index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Intermediate submission (LightGBM only) saved to submission_lgb.csv", flush=True)
except Exception as e:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Could not save intermediate submission: {e}", flush=True)
sys.stdout.flush()

# CatBoost
print("\n" + "="*60)
print("--- CatBoost ---")
print("="*60)
cat_start = time.time()

# Проверяем, нужно ли пропустить CatBoost если есть checkpoint
skip_cat = False
if os.path.exists(CAT_CHECKPOINT):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] CatBoost checkpoint found. Loading...", flush=True)
    try:
        test_predictions_cat = np.load(CAT_CHECKPOINT)
        skip_cat = True
        print(f"[{datetime.now().strftime('%H:%M:%S')}] CatBoost predictions loaded from checkpoint", flush=True)
    except:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to load CatBoost checkpoint, will retrain", flush=True)
        skip_cat = False

if not skip_cat:
    if 'fold_splits' not in locals():
        fold_splits = list(skf.split(X_train, y_train))
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(fold_splits, desc="CatBoost Folds", ncols=100)):
        fold_start = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fold {fold + 1}/{N_FOLDS} - Training...", flush=True)
        sys.stdout.flush()
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data prepared. Train: {len(X_tr)}, Val: {len(X_val)}", flush=True)
        sys.stdout.flush()
        
        model = cb.CatBoostClassifier(
            iterations=2500,  # Оптимальный баланс
            learning_rate=0.015,  # Более консервативный learning rate
            depth=7,  # Оптимальная глубина
            l2_leaf_reg=5,  # Увеличено для регуляризации
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=RANDOM_STATE,
            verbose=100,
            early_stopping_rounds=150,  # Увеличено для более стабильного early stopping
            min_data_in_leaf=15,  # Снижено для большей детализации
            thread_count=-1,
            bootstrap_type='Bayesian',  # Улучшенный бутстрап
            bagging_temperature=0.8,  # Для Bayesian bootstrap
            random_strength=0.5,  # Добавлено для регуляризации
            border_count=128  # Увеличено для лучшей точности
        )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting CatBoost training...", flush=True)
        sys.stdout.flush()
        
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            cat_features=categorical_features_filtered,
            use_best_model=True
        )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed. Best iteration: {model.best_iteration_}", flush=True)
        sys.stdout.flush()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Making predictions on validation set ({len(X_val)} samples)...", flush=True)
        sys.stdout.flush()
        oof_predictions_cat[val_idx] = model.predict_proba(X_val)[:, 1]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Validation predictions completed.", flush=True)
        sys.stdout.flush()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Making predictions on test set ({len(X_test)} samples)...", flush=True)
        sys.stdout.flush()
        test_pred = model.predict_proba(X_test)[:, 1]
        test_predictions_cat += test_pred / N_FOLDS
        del test_pred  # Освобождаем память
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Test predictions completed.", flush=True)
        sys.stdout.flush()
        
        val_auc = roc_auc_score(y_val, oof_predictions_cat[val_idx])
        fold_time = time.time() - fold_start
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fold {fold + 1} AUC: {val_auc:.5f} | Time: {fold_time:.1f}s | Best iteration: {model.best_iteration_}", flush=True)
        del model
        gc.collect()

cat_auc = roc_auc_score(y_train, oof_predictions_cat)
cat_time = time.time() - cat_start
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] CatBoost OOF AUC: {cat_auc:.5f} | Total time: {cat_time/60:.1f} min")

# XGBoost
print("\n" + "="*60)
print("--- XGBoost ---")
print("="*60)
xgb_start = time.time()

# Проверяем, нужно ли пропустить XGBoost если есть checkpoint
skip_xgb = False
if os.path.exists(XGB_CHECKPOINT):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] XGBoost checkpoint found. Loading...", flush=True)
    try:
        test_predictions_xgb = np.load(XGB_CHECKPOINT)
        skip_xgb = True
        print(f"[{datetime.now().strftime('%H:%M:%S')}] XGBoost predictions loaded from checkpoint", flush=True)
    except:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to load XGBoost checkpoint, will retrain", flush=True)
        skip_xgb = False

if not skip_xgb:
    if 'fold_splits' not in locals():
        fold_splits = list(skf.split(X_train, y_train))
    
    # Для XGBoost нужно преобразовать категориальные признаки в числовые
    # Создаем копию данных для XGBoost с LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    X_train_xgb = X_train_encoded.copy()
    X_test_xgb = X_test_encoded.copy()
    
    # Кодируем категориальные признаки для XGBoost
    le_dict_xgb = {}
    for col in categorical_features_filtered:
        if col in X_train_xgb.columns:
            le = LabelEncoder()
            # Объединяем train и test для правильного кодирования
            combined = pd.concat([X_train_xgb[col].astype(str), X_test_xgb[col].astype(str)])
            le.fit(combined)
            X_train_xgb[col] = le.transform(X_train_xgb[col].astype(str))
            X_test_xgb[col] = le.transform(X_test_xgb[col].astype(str))
            le_dict_xgb[col] = le
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(fold_splits, desc="XGBoost Folds", ncols=100)):
        fold_start = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fold {fold + 1}/{N_FOLDS} - Training...", flush=True)
        
        X_tr, X_val = X_train_xgb.iloc[train_idx], X_train_xgb.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBClassifier(
            n_estimators=2500,  # Оптимальный баланс
            max_depth=7,  # Оптимальная глубина
            learning_rate=0.015,  # Более консервативный learning rate
            subsample=0.75,  # Более агрессивная регуляризация
            colsample_bytree=0.75,
            colsample_bylevel=0.75,  # Дополнительная регуляризация
            min_child_weight=5,  # Увеличено для предотвращения переобучения
            gamma=0.3,  # Сильная регуляризация
            reg_alpha=0.3,  # Сильная регуляризация
            reg_lambda=2.5,  # Сильная регуляризация
            random_state=RANDOM_STATE,
            eval_metric='auc',
            use_label_encoder=False,
            tree_method='hist',
            grow_policy='lossguide',
            max_leaves=127,  # Добавлено ограничение листьев
            early_stopping_rounds=150  # Увеличено для более стабильного early stopping
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        oof_predictions_xgb[val_idx] = model.predict_proba(X_val)[:, 1]
        test_predictions_xgb += model.predict_proba(X_test_xgb)[:, 1] / N_FOLDS
        
        val_auc = roc_auc_score(y_val, oof_predictions_xgb[val_idx])
        fold_time = time.time() - fold_start
        best_iter = model.get_booster().best_iteration if hasattr(model.get_booster(), 'best_iteration') else model.n_estimators
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fold {fold + 1} AUC: {val_auc:.5f} | Time: {fold_time:.1f}s | Best iteration: {best_iter}", flush=True)
        del model
        gc.collect()

xgb_auc = roc_auc_score(y_train, oof_predictions_xgb)
xgb_time = time.time() - xgb_start
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] XGBoost OOF AUC: {xgb_auc:.5f} | Total time: {xgb_time/60:.1f} min", flush=True)

# Сохраняем промежуточные результаты XGBoost
print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving XGBoost checkpoint...", flush=True)
np.save(XGB_CHECKPOINT, test_predictions_xgb)
print(f"[{datetime.now().strftime('%H:%M:%S')}] XGBoost checkpoint saved", flush=True)
sys.stdout.flush()

# Ансамблирование
print("\n" + "="*60)
print("[5/6] Ансамблирование...")
print("="*60)
ensemble_start = time.time()

# Stacking вместо простого ансамблирования
print(f"[{datetime.now().strftime('%H:%M:%S')}] Building Stacking Meta-Model...", flush=True)
from sklearn.linear_model import LogisticRegression

# Создаем мета-признаки из OOF предсказаний
meta_features_train = np.column_stack([
    oof_predictions_lgb,
    oof_predictions_cat,
    oof_predictions_xgb,
    oof_predictions_lgb ** 2,  # Квадраты для нелинейности
    oof_predictions_cat ** 2,
    oof_predictions_xgb ** 2,
    oof_predictions_lgb * oof_predictions_cat,  # Взаимодействия
    oof_predictions_lgb * oof_predictions_xgb,
    oof_predictions_cat * oof_predictions_xgb
])

# Обучаем мета-модель (Logistic Regression с регуляризацией)
meta_model = LogisticRegression(
    C=0.1,  # Сильная регуляризация
    max_iter=1000,
    random_state=RANDOM_STATE,
    solver='lbfgs'
)
meta_model.fit(meta_features_train, y_train)

# Предсказания мета-модели на train
oof_ensemble = meta_model.predict_proba(meta_features_train)[:, 1]
ensemble_auc = roc_auc_score(y_train, oof_ensemble)

# Для test создаем мета-признаки
meta_features_test = np.column_stack([
    test_predictions_lgb,
    test_predictions_cat,
    test_predictions_xgb,
    test_predictions_lgb ** 2,
    test_predictions_cat ** 2,
    test_predictions_xgb ** 2,
    test_predictions_lgb * test_predictions_cat,
    test_predictions_lgb * test_predictions_xgb,
    test_predictions_cat * test_predictions_xgb
])

# Финальные предсказания
test_predictions = meta_model.predict_proba(meta_features_test)[:, 1]

# Также пробуем простое взвешивание для сравнения
auc_scores = np.array([lgb_auc, cat_auc, xgb_auc])
weights_simple = auc_scores ** 3  # Кубическое взвешивание
weights_simple = weights_simple / weights_simple.sum()
simple_ensemble = (weights_simple[0] * test_predictions_lgb + 
                    weights_simple[1] * test_predictions_cat + 
                    weights_simple[2] * test_predictions_xgb)
simple_auc = roc_auc_score(y_train, 
    weights_simple[0] * oof_predictions_lgb + 
    weights_simple[1] * oof_predictions_cat + 
    weights_simple[2] * oof_predictions_xgb)

# Выбираем лучший метод
if ensemble_auc > simple_auc:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Using Stacking (better than weighted average)", flush=True)
    print(f"  Stacking AUC: {ensemble_auc:.5f} vs Weighted AUC: {simple_auc:.5f}", flush=True)
else:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Using Weighted Average (better than stacking)", flush=True)
    print(f"  Weighted AUC: {simple_auc:.5f} vs Stacking AUC: {ensemble_auc:.5f}", flush=True)
    test_predictions = simple_ensemble
    ensemble_auc = simple_auc

print(f"[{datetime.now().strftime('%H:%M:%S')}] Individual model AUCs:")
print(f"  LightGBM: {lgb_auc:.5f}")
print(f"  CatBoost: {cat_auc:.5f}")
print(f"  XGBoost:  {xgb_auc:.5f}")

ensemble_time = time.time() - ensemble_start
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Final Ensemble OOF AUC: {ensemble_auc:.5f} | Time: {ensemble_time:.1f}s")

# Создание submission файла
print("\n" + "="*60, flush=True)
print("[6/6] Создание submission файла...", flush=True)
print("="*60, flush=True)
submission_start = time.time()
sys.stdout.flush()

try:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating submission DataFrame...", flush=True)
    sys.stdout.flush()
    
    submission = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': test_predictions
    })
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Submission DataFrame created. Shape: {submission.shape}", flush=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving to submission.csv...", flush=True)
    sys.stdout.flush()
    
    submission.to_csv('submission.csv', index=False)
    
    submission_time = time.time() - submission_start
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Submission saved to submission.csv | Time: {submission_time:.1f}s", flush=True)
    sys.stdout.flush()
    
    # Проверяем, что файл создан
    if os.path.exists('submission.csv'):
        file_size = os.path.getsize('submission.csv')
        print(f"[{datetime.now().strftime('%H:%M:%S')}] File verified. Size: {file_size / 1024:.2f} KB", flush=True)
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: File was not created!", flush=True)
        
except Exception as e:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR creating submission file: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.stdout.flush()

# Финальная статистика
total_time = time.time() - start_time_total
print("\n" + "=" * 60)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print("=" * 60)
print(f"\nРезультаты моделей:")
print(f"  LightGBM:  {lgb_auc:.5f} ({lgb_time/60:.1f} min)")
print(f"  CatBoost:  {cat_auc:.5f} ({cat_time/60:.1f} min)")
print(f"  XGBoost:   {xgb_auc:.5f} ({xgb_time/60:.1f} min)")
print(f"  Ensemble:  {ensemble_auc:.5f} (лучший результат)")
print(f"\nОбщее время обучения: {total_time/60:.1f} минут ({total_time:.0f} секунд)")
print(f"Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

