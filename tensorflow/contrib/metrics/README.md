# TensorFlow evaluation metrics and summary statistics

## Evaluation metrics

Compare predictions and labels, producing an aggregate loss.  Typically produce
a `value` and an `update_op`.  The `update_op` is run with every batch to update
internal state (e.g. accumulated right/wrong predictions).
The `value` is extracted after all batches have been read (e.g. precision =
number correct / total).

```python
predictions = ...
labels = ...
value, update_op = some_metric(predictions, labels)

for step_num in range(max_steps):
  update_op.run()

print "evaluation score: ", value.eval()
```
