# Predictors

The `Predictor` classes provide a simple interface for performing repeated,
efficient inference. A `Predictor` can be constructed from a `SavedModel` on
disk, a `tf.Estimator` or a `tf.contrib.Estimator`.

To facilitate the examples below, let's define a trivial `Estimator` that just
calculates a sum:

```python
def model_fn(features, labels, mode):
  z = tf.add(features['x'], features['y'], name='z')
  return tf.contrib.learn.ModelFnOps(
      mode, {'z': z}, loss=tf.constant(0.0), train_op=tf.no_op())

estimator = tf.contrib.learn.Estimator(model_fn=model_fn)
```

We can then construct a `Predictor` in two different ways.

## `Predictor` from a `SavedModel`

Given a trained `Estimator`, we first export a `SavedModel`:

```python
def serving_input_fn():
  x = tf.placeholder(dtype=tf.float32, shape=[None], name='x')
  y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

  features = {'x': x, 'y': y}
  return tf.contrib.learn.utils.input_fn_utils.InputFnOps(
           features, None, default_inputs=features)

saved_model_dir = estimator.export_savedmodel(my_export_dir, serving_input_fn)
```

We can then construct a `Predictor` as follows:

```python
saved_model_predictor = predictor.from_saved_model(export_dir='test_export_dir')
output_dict = saved_model_predictor({'x': [1.0], 'y': [5.2]})
# output_dict == {'sum': [6.2]}
```

By specifying a signature definition, we can feed and fetch any `Tensor`s in
the `Graph`. In this example, we feed and fetch the same `Tensor`, `z`:

```python
inputs = outputs = {'z': tf.TensorInfo(
                        name='z:0',
                        dtype=types_pb2.DT_FLOAT,
                        tensor_shape=tensor_shape_pb2.TensorShapeProto())}

signature_def = tf.saved_model.signature_def_utils.build_signature_def(
          inputs=inputs,
          outputs=outputs,
          method_name='tensorflow/serving/regress')

trivial_predictor = predictor.from_saved_model(
          export_dir=saved_model_dir,
          signature_def=signature_def)

output_dict = trivial_predictor({'z': [32.]})
# output_dict == {'z': [32.]}
```

You can also specify input and output `Tensor`s by name using the `input_names`
and `output_names` keywords:

```python
saved_model_predictor = predictor.from_saved_model(
          export_dir=saved_model_dir,
          input_names={'x': 'x:0', 'y': 'y:0'},
          outputs={'z': 'z:0'})

output_dict = saved_model_predictor({'x': [6.], 'y': [11.]})
# output_dict == {'z': [17.]}
```

This functionality is particularly useful for performing encoding once, but
doing multiple decoding iterations with e.g. seq2seq models.

## `Predictor` from an `Estimator`

We can also construct a `Predictor` directly from an `Estimator`. Defining
`serving_input_fn` as above,

```python
estimator_predictor = predictor.from_contrib_estimator(
    estimator, serving_input_fn)
output_dict = sum_predictor({'x': [1., 2.], 'y': [3., 4.]})
# output_dict == {'z': [4., 6.]}
```

Construction from a `tf.Estimator` is almost identical.

