# S&P 500 Stock Market Prediction

## Overview

This project uses historical S&P 500 index data to predict whether the index will increase or decrease on the next trading day. The prediction model employs a Random Forest Classifier, leveraging features such as rolling averages and trend indicators over various time horizons. The Jupyter notebook (`RFC.ipynb`) contains the complete workflow, including data retrieval, preprocessing, model training, and evaluation.

## Prerequisites

To run the notebook, you need the following Python packages:

- `yfinance`: To fetch historical S&P 500 data.
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `matplotlib`: For plotting (used for visualizing the S&P 500 index).
- `scikit-learn`: For the Random Forest Classifier and precision score calculation.

Install the dependencies using:

```bash
pip install yfinance pandas numpy matplotlib scikit-learn
```

## Project Structure

- **RFC.ipynb**: The main Jupyter notebook containing the code for data retrieval, preprocessing, model training, backtesting, and evaluation.
- **README.md**: This file, providing an overview and instructions for the project.

## Workflow

1. **Data Retrieval**:

   - The S&P 500 index data (`^GSPC`) is fetched using the `yfinance` library with the `history(period="max")` method to obtain the maximum available historical data.
   - The data includes columns: `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends`, and `Stock Splits`.

2. **Data Preprocessing**:

   - A `Tomorrow` column is created to store the next day's closing price.
   - A `Target` column is generated, where `1` indicates the next day's close is higher than the current day's close, and `0` otherwise.
   - New features are engineered:
     - **Close Ratios**: The ratio of the current day's closing price to the rolling mean over horizons of 2, 5, 60, 250, and 1000 days (`Close_Ratio_<horizon>`).
     - **Trends**: The sum of the `Target` values over the previous days for each horizon (`Trend_<horizon>`), shifted to avoid data leakage.
   - Rows with missing values (due to rolling calculations) are dropped.

3. **Model Training**:

   - A Random Forest Classifier is used with 200 estimators and a minimum split size of 50 (`random_state=1` for reproducibility).
   - The model is trained on the engineered features (`Close_Ratio_<horizon>` and `Trend_<horizon>` for each horizon).

4. **Prediction and Backtesting**:

   - A custom `predict` function trains the model on a training set and predicts on a test set.
   - Predictions are based on probabilities, with a threshold of 0.6 to classify a day as `1` (price increase) or `0` (price decrease or no increase).
   - A `backtest` function (not shown in the provided code but implied) performs rolling predictions to simulate real-world trading scenarios.

5. **Evaluation**:

   - The model's performance is evaluated using the precision score, which measures the proportion of correctly predicted price increases (`1`).
   - The distribution of predictions and actual targets is analyzed to understand model behavior.

## Results

- **Initial Model**:

  - Precision Score: \~52.89%
  - Prediction Distribution: 3502 (`0`), 2596 (`1`)
  - Target Distribution: \~53.48% (`1`), \~46.52% (`0`)

- **Improved Model** (with engineered features):

  - Precision Score: \~57.33%
  - Prediction Distribution: 4265 (`0`), 832 (`1`)
  - The improved model predicts fewer price increases but achieves higher precision, indicating better accuracy when predicting upward movements.

## Usage

1. Clone the repository or download the `RFC.ipynb` file.

2. Ensure all dependencies are installed.

3. Open the notebook in Jupyter:

   ```bash
   jupyter notebook RFC.ipynb
   ```

4. Run the cells sequentially to fetch data, preprocess it, train the model, and evaluate results.

5. Modify the `horizons`, model parameters (`n_estimators`, `min_samples_split`), or prediction threshold (`.6`) to experiment with different configurations.

## Notes

- The data is sourced from Yahoo Finance via `yfinance`, which may have limitations or delays in real-time data.
- The model uses a simple threshold-based prediction. Adjusting the threshold or exploring other algorithms (e.g., XGBoost, LSTM) could improve performance.
- The backtesting function assumes a rolling window approach, but its implementation is not fully shown in the provided code. Ensure it is defined or implement a custom backtesting loop if needed.
- The notebook includes a plot of the S&P 500 index, which can be used for visual analysis of trends.

## Future Improvements

- Incorporate additional features, such as volatility indicators or macroeconomic data.
- Experiment with different machine learning models or ensemble methods.
- Optimize hyperparameters using grid search or random search.
- Implement cross-validation to ensure robustness across different time periods.
- Add visualization of predictions vs. actual outcomes for better interpretability.

## License

This project is for educational purposes and does not include a specific license. Ensure compliance with the terms of use for `yfinance` data and any other dependencies.