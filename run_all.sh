#!/usr/bin/env bash
# Runs the whole project from raw data to final figures, in order.
set -e
python -m src.data_prep        # 1. clean the raw sensor log -> 30-minute grid + anomaly flags
python -m src.eda              # 2. EDA figures, heatmaps, zone clustering
python -m src.train            # 3. tune, train, stack and evaluate all models (the slow step)
python -m src.explain          # 4. SHAP explainability
python -m src.error_analysis   # 5. error analysis figures and tables
python -m src.export_forecasts # 6. batch-score the test period for the app
python -m pytest -q            # 7. sanity tests
echo "Done. Start the app with:  streamlit run app/streamlit_app.py"
