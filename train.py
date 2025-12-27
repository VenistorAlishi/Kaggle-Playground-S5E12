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
N_FOLDS = 5

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

def create_features(df, train_stats=None):
    """Создание новых признаков
    
    Args:
        df: DataFrame для обработки
        train_stats: словарь со статистиками train (mean, std) для нормализации
    """
    df = df.copy()
    
    # BMI категории
    df['bmi_category'] = pd.cut(df['bmi'], 
                                bins=[0, 18.5, 25, 30, 100], 
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    df['bmi_category'] = df['bmi_category'].astype(str)
    
    # Возрастные группы
    df['age_group'] = pd.cut(df['age'], 
                             bins=[0, 30, 45, 60, 100], 
                             labels=['Young', 'Middle', 'Senior', 'Elderly'])
    df['age_group'] = df['age_group'].astype(str)
    
    # Комбинированные признаки здоровья
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1e-5)
    df['cholesterol_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)
    df['total_cholesterol_hdl'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
    df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)
    
    # Индексы здоровья
    df['health_score'] = (df['diet_score'] + 
                         (df['physical_activity_minutes_per_week'] / 100) + 
                         (df['sleep_hours_per_day'] / 8))
    
    df['lifestyle_score'] = (df['diet_score'] * 0.4 + 
                             (df['physical_activity_minutes_per_week'] / 200) * 0.3 +
                             (df['sleep_hours_per_day'] / 8) * 0.3)
    
    df['risk_factors_count'] = (df['family_history_diabetes'] + 
                                df['hypertension_history'] + 
                                df['cardiovascular_history'])
    
    # Взаимодействия
    df['bmi_age'] = df['bmi'] * df['age']
    df['bmi_waist_hip'] = df['bmi'] * df['waist_to_hip_ratio']
    df['activity_sleep'] = df['physical_activity_minutes_per_week'] * df['sleep_hours_per_day']
    df['age_activity'] = df['age'] * df['physical_activity_minutes_per_week']
    df['bmi_systolic'] = df['bmi'] * df['systolic_bp']
    
    # Логические признаки
    df['high_bp'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 85)).astype(int)
    df['high_cholesterol'] = (df['cholesterol_total'] >= 200).astype(int)
    df['high_ldl'] = (df['ldl_cholesterol'] >= 100).astype(int)
    df['low_hdl'] = (df['hdl_cholesterol'] < 40).astype(int)
    df['high_triglycerides'] = (df['triglycerides'] >= 150).astype(int)
    df['obese'] = (df['bmi'] >= 30).astype(int)
    df['overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
    df['high_waist_hip'] = (df['waist_to_hip_ratio'] > 0.9).astype(int)
    
    # Метаболические признаки
    df['metabolic_risk'] = (df['high_bp'].astype(int) + 
                            df['high_cholesterol'].astype(int) + 
                            df['high_triglycerides'].astype(int) + 
                            df['obese'].astype(int))
    
    # Нормализованные признаки (используем статистики train для test)
    if train_stats is None:
        # Вычисляем статистики для train
        age_mean, age_std = df['age'].mean(), df['age'].std()
        bmi_mean, bmi_std = df['bmi'].mean(), df['bmi'].std()
        train_stats = {'age': (age_mean, age_std), 'bmi': (bmi_mean, bmi_std)}
    else:
        # Используем статистики train для test
        age_mean, age_std = train_stats['age']
        bmi_mean, bmi_std = train_stats['bmi']
    
    df['age_normalized'] = (df['age'] - age_mean) / (age_std + 1e-5)
    df['bmi_normalized'] = (df['bmi'] - bmi_mean) / (bmi_std + 1e-5)
    
    # Полиномиальные признаки (важные взаимодействия)
    df['age_squared'] = df['age'] ** 2
    df['bmi_squared'] = df['bmi'] ** 2
    
    return df, train_stats

# Создаем признаки для train и сохраняем статистики
X_train, train_stats = create_features(X_train)
# Создаем признаки для test используя статистики train
X_test, _ = create_features(X_test, train_stats)

# Обновление списков признаков
categorical_features.extend(['bmi_category', 'age_group'])
new_numerical = ['bp_ratio', 'cholesterol_ratio', 'total_cholesterol_hdl', 'non_hdl_cholesterol',
                 'ldl_hdl_ratio', 'health_score', 'lifestyle_score', 'risk_factors_count',
                 'bmi_age', 'bmi_waist_hip', 'activity_sleep', 'age_activity', 'bmi_systolic',
                 'high_bp', 'high_cholesterol', 'high_ldl', 'low_hdl', 'high_triglycerides',
                 'obese', 'overweight', 'high_waist_hip', 'metabolic_risk',
                 'age_normalized', 'bmi_normalized', 'age_squared', 'bmi_squared']
numerical_features.extend(new_numerical)

print(f"Total features after engineering: {len(X_train.columns)}")

# Подготовка данных для моделей
print("\n[3/6] Подготовка данных...")

# Кодирование категориальных признаков для LightGBM и XGBoost
print("Preparing encoded datasets...", flush=True)
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()
print("Encoded datasets prepared.", flush=True)

# Фильтруем только существующие категориальные признаки
categorical_features_filtered = [col for col in categorical_features if col in X_train_encoded.columns]

for col in categorical_features_filtered:
    X_train_encoded[col] = X_train_encoded[col].astype('category')
    X_test_encoded[col] = X_test_encoded[col].astype('category')

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
            'num_leaves': 63,
            'learning_rate': 0.03,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_split_gain': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
            'random_state': RANDOM_STATE
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
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
            iterations=2000,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=5,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=RANDOM_STATE,
            verbose=100,  # Изменено: показываем прогресс каждые 100 итераций
            early_stopping_rounds=100,
            min_data_in_leaf=20,
            thread_count=-1  # Используем все доступные потоки
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
            n_estimators=2000,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=RANDOM_STATE,
            eval_metric='auc',
            use_label_encoder=False,
            tree_method='hist',
            grow_policy='lossguide',
            early_stopping_rounds=100  # Перемещено в конструктор для новых версий XGBoost
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

# Взвешенное усреднение на основе OOF AUC
weights = np.array([lgb_auc, cat_auc, xgb_auc])
weights = weights / weights.sum()

print(f"[{datetime.now().strftime('%H:%M:%S')}] Model weights:")
print(f"  LightGBM: {weights[0]:.4f} (AUC: {lgb_auc:.5f})")
print(f"  CatBoost:  {weights[1]:.4f} (AUC: {cat_auc:.5f})")
print(f"  XGBoost:   {weights[2]:.4f} (AUC: {xgb_auc:.5f})")

oof_ensemble = (weights[0] * oof_predictions_lgb + 
                weights[1] * oof_predictions_cat + 
                weights[2] * oof_predictions_xgb)

ensemble_auc = roc_auc_score(y_train, oof_ensemble)
ensemble_time = time.time() - ensemble_start
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Ensemble OOF AUC: {ensemble_auc:.5f} | Time: {ensemble_time:.1f}s")

test_predictions = (weights[0] * test_predictions_lgb + 
                   weights[1] * test_predictions_cat + 
                   weights[2] * test_predictions_xgb)

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

