Title

A robust Ensemble Learning Approach for Early Paddy Rice Prediction

Overview

Predicting paddy (rice) yield in kilograms from farm management decisions and climate observations, using classical machine learning and a neural network. The best model is a Stacking ensemble that achieves R² = 0.9916 and RMSE = 826 kg on a held-out test set.
The repository covers the full workflow: data cleaning, feature engineering, exploratory data analysis, model training and comparison, explainable-AI diagnostics, and a working Gradio demo.

Objective

To develop and evaluate machine learning models that predict paddy rice yield (in kilograms) for a farm-season from pre-harvest information — farm attributes, applied inputs, and climate observation and to deploy the best-performing model as an interactive tool that farmers, co-operatives and extension officers can use to support storage, pricing and input-management decisions.

Dataset

Source	UCI Irvine ML repository(https://archive.ics.uci.edu/dataset/1186/paddy+dataset).
Rows	2,789 farm-seasons, Columns	68 raw variables, Target	Paddy yield (in Kg) — mean 22,518 kg, range 5,410 – 38,814 kg,
Categorical features	Agriblock (location), Variety, Soil Type, Nursery Type, Numeric features	Hectares, seed rate, DAP, urea, potash, micronutrients, pesticide, land prep, weed control
Climate features	Rain · Min temp · Max temp · Relative humidity, each measured across 4 windows: D1–30, D31–60, D61–90, D91–120

Feature Engineering

Raw input totals (e.g. total kilograms of urea) mostly track farm size. To let the model reason about how intensely a farm is managed, every input is re-expressed on a per-hectare basis. Three groups of engineered features are added to the raw columns:

1. Per-hectare input ratios (9 features) — seed rate/ha, DAP/ha, urea/ha, potash/ha, micronutrient/ha, pesticide/ha, land-prep/ha, nursery-prep/ha, weed-control/ha.
2. Composite intensity score — the mean of the z-scored ratios. Captures whether a farm is broadly under-applying, optimally applying, or over-applying inputs.
3. Climate aggregates — across the four growth windows: mean rain, total rain, rain variability (CV), mean min / max temperature, temperature range, mean relative humidity, RH range.

Categorical features (Agriblock, Variety, Soil Type, Nursery) are label-encoded.

The final feature matrix has the original raw columns plus these engineered features — giving tree-based models everything they need to capture both linear and non-linear effects.

Models Used
Random Forest, XAI, Neural Network(MLP), Gradient boosting, Extra trees, Ridge regression(baseline), voting, stack ensemble, 5-fold cross validation
Tools Used

pandas, numpy — data wrangling
matplotlib, seaborn — visualisation
scikit-learn — classical ML, ensembles, MLP, evaluation, permutation importance
Gradio — interactive demo
Google Colab — notebook environment

Author

Majwega Jackson — Final Project, Refactory
