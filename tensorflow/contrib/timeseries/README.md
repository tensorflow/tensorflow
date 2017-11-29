# TensorFlow Time Series

TensorFlow Time Series (TFTS) is a collection of ready-to-use classic models
(state space, autoregressive), and flexible infrastructure for building
high-performance time series models whatever the architecture. It includes tools
for chunking and batching a series, and for saving model state across chunks,
making use of parallel computation even when training sequential models on long
series (using truncated backpropagation).

To get started, take a look at the `examples/` directory, which includes:

 - Making probabilistic forecasts
 - Using exogenous features to train on data with known anomalies/changepoints
 - Learning correlations between series (multivariate forecasting/anomaly
   detection)

TFTS includes many other modeling tools, including non-linear autoregression
(see the `hidden_layer_sizes` argument to `ARRegressor` in `estimators.py`) and
a collection of components for linear state space modeling (level, trend,
period, vector autoregression, moving averages; see the
`StructuralEnsembleRegressor` in `estimators.py`). Both model classes support
heuristics for ignoring un-labeled anomalies in training data. Trained models
can be exported for inference/serving in
[SavedModel format](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md)
(see `examples/multivariate.py`).
