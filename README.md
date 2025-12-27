# Kaggle Playground Series S5E12 - Diabetes Prediction

Решение для соревнования по предсказанию диабета с использованием ансамблирования градиентного бустинга.

**Competition:** [Kaggle Playground Series - Season 5, Episode 12](https://www.kaggle.com/competitions/playground-series-s5e12)

## Описание

Это решение использует три модели градиентного бустинга:
- **LightGBM** - быстрая и эффективная модель
- **CatBoost** - отлично работает с категориальными признаками
- **XGBoost** - проверенная временем модель

Модели объединяются в ансамбль с весами, основанными на их OOF (Out-of-Fold) AUC.

## Установка

```bash
pip install -r requirements.txt
```

Или используя uv:
```bash
uv pip install -r requirements.txt
```

## Использование

```bash
uv run python train.py
```

Или:
```bash
python train.py
```

## Особенности решения

1. **Feature Engineering**:
   - BMI категории
   - Возрастные группы
   - Комбинированные признаки здоровья (соотношения холестерина, давления)
   - Индексы здоровья
   - Логические признаки (высокое давление, высокий холестерин и т.д.)

2. **Кросс-валидация**:
   - 5-fold Stratified K-Fold для надежной оценки

3. **Ансамблирование**:
   - Взвешенное усреднение предсказаний на основе OOF AUC

4. **Оптимизация**:
   - Early stopping для предотвращения переобучения
   - Настроенные гиперпараметры для каждой модели

## Структура файлов

- `train.py` - основной скрипт обучения с системой checkpoint
- `requirements.txt` - зависимости проекта
- `data_analysis.ipynb` - Jupyter notebook с EDA и анализом данных
- `submission.csv` - финальный файл с предсказаниями (ансамбль всех моделей)
- `submission_lgb.csv` - промежуточный submission (только LightGBM)
- `submission_lgb_cat.csv` - промежуточный submission (LightGBM + CatBoost)
- `checkpoints/` - папка с сохраненными предсказаниями для возможности продолжить обучение

## Система Checkpoint

Скрипт автоматически сохраняет промежуточные результаты:
- После LightGBM → `checkpoints/lgb_predictions.npy`
- После CatBoost → `checkpoints/cat_predictions.npy`
- После XGBoost → `checkpoints/xgb_predictions.npy`

При перезапуске скрипт автоматически загружает сохраненные результаты и продолжает с того места, где остановился.

## Какой submission загружать?

**Рекомендуется использовать `submission.csv`** - это финальный файл с ансамблем всех трех моделей (LightGBM + CatBoost + XGBoost), который должен дать лучший результат.

Если финальный файл еще не создан, можно использовать:
- `submission_lgb_cat.csv` - ансамбль LightGBM + CatBoost (второй по качеству)
- `submission_lgb.csv` - только LightGBM (базовый вариант)

## Метрика

Соревнование использует **ROC AUC** в качестве метрики оценки.



