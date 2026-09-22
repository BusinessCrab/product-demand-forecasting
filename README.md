# Product Demand Forecasting

An end-to-end, leakage-aware time-series forecasting project for predicting daily unit sales for each `store_nbr` and product `family`. The main deliverable is the executed [Jupyter notebook](product_demand_forecasting.ipynb), which covers data-quality checks, exploratory and time-series analysis, feature engineering, chronological validation, baseline and XGBoost forecasts, representative SARIMA and Prophet models, error analysis, and a Kaggle-ready submission.

## Dataset Details

This project uses Kaggle's [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition data. The target is `sales`; the forecasting grain is `date`, `store_nbr`, and `family`. The competition test set covers a 16-day future horizon.

The notebook uses historical sales, `onpromotion`, store metadata, the holiday/event calendar, and daily oil prices where available. It intentionally does **not** create an advertising feature: the source data does not provide one. The `transactions.csv` file is inspected but is not used by the forecasting model because future transactions are not supplied for the complete test horizon.

## Forecasting Workflow

1. Data-quality analysis
2. Exploratory data analysis
3. Time-series characteristics
4. Feature engineering
5. Chronological validation
6. Seasonal-naive baseline forecasting
7. XGBoost global forecasting model
8. Representative-series SARIMA
9. Aggregated Prophet model
10. Evaluation and error analysis
11. Final 16-day forecast and `submission.csv`

## Methodology and Leakage Safeguards

- The final validation period is the latest 28 calendar days; there is no random train/test split.
- Lags are built independently for each `(store_nbr, family)` series.
- Rolling statistics are calculated from `sales.shift(1)`, so a row never uses its own target.
- Validation and test forecasts are generated recursively: after the forecast origin, only previous predictions—not hidden actual sales—can populate lag features.
- Calendar, holiday, store, promotion, and oil inputs are used only when supplied or knowable for the forecast rows.
- The model never uses `transactions` as a predictive feature because it is unavailable over the future horizon.
- The untouched validation targets are used only for final evaluation and error analysis.

## Models

- **Seasonal naive:** repeats the demand from seven days earlier. It is the practical benchmark for weekly retail seasonality.
- **XGBoost:** one global regressor trained with lag, shifted rolling, calendar, promotion, store, holiday, and oil features.
- **SARIMA:** fitted to an aggregated daily series to show a classical seasonal statistical model without fitting thousands of separate models.
- **Prophet:** fitted to the aggregated daily series with weekly/yearly seasonality and a known holiday calendar. This is likewise demonstrative rather than a per-series production model.

## Evaluation

The notebook reports MAE, RMSE, and RMSLE. RMSLE is computed after clipping predictions to zero because sales cannot be negative. MAPE is deliberately not a primary metric: valid zero-demand days make percentage errors unstable. XGBoost metrics are reported at the store-family-day level; SARIMA and Prophet metrics are reported separately at the aggregate daily level, so they are not presented as directly comparable.

## Limitations

- The test period is short and future demand can be disrupted by events not represented in the data.
- A global model may miss product- or store-specific behavior.
- Recursive multi-step forecasts can accumulate errors after the first week.
- Promotions are available in Kaggle's test rows, but future promotional plans may not be known in every real deployment.
- Oil is a coarse external signal and its relationship to individual product demand may change.
- The notebook's default full mode can require substantial memory and several minutes of CPU time. `FAST_MODE=1` is a clearly labelled development option; it evaluates a representative subset and falls back to seasonal-naive predictions for the remaining test series.

## Run

Download the competition files with Kaggle and extract them into `data/`:

```bash
python -m pip install kaggle
kaggle competitions download -c store-sales-time-series-forecasting -p data
python -m zipfile -e data/store-sales-time-series-forecasting.zip data
python -m pip install -r requirements.txt
jupyter notebook product_demand_forecasting.ipynb
```