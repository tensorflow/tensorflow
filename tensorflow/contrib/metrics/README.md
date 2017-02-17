# TensorFlow evaluation metrics and summary statistics

## Evaluation metrics

Metrics are used in evaluation to assess the quality of a model. Most are
"streaming" ops, meaning they create variables to accumulate a running total,
and return an update tensor to update these variables, and a value tensor to 
read the accumulated value. Example:

value, update_op = metrics.streaming_mean_squared_error(
    predictions, targets, weight)

Most metric functions take a pair of tensors, `predictions` and ground truth
`targets` (`streaming_mean` is an exception, it takes a single value tensor,
usually a loss). It is assumed that the shape of both these tensors is of the
form `[batch_size, d1, ... dN]` where `batch_size` is the number of samples in
the batch and `d1` ... `dN` are the remaining dimensions.

The `weight` parameter can be used to adjust the relative weight of samples
within the batch. The result of each loss is a scalar average of all sample
losses with non-zero weights.

The result is 2 tensors that should be used like the following for each eval
run:

```python
predictions = ...
labels = ...
value, update_op = some_metric(predictions, labels)

for step_num in range(max_steps):
  update_op.run()

print "evaluation score: ", value.eval()
```
