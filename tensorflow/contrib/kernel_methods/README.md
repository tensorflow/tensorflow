# TensorFlow contrib kernel_methods.

This module contains operations and estimators that enable the use of primal
(explicit) kernel methods in TensorFlow. See also the [tutorial](https://www.tensorflow.org/code/tensorflow/contrib/kernel_methods/g3doc/tutorial.md) on how to use this module to improve the quality of
classification or regression tasks.

## Kernel Mappers
Implement explicit kernel mapping Ops over tensors. Kernel mappers add
Tensor-In-Tensor-Out (TITO) Ops to the TensorFlow graph. They can be used in
conjunction with other layers or ML models.

Sample usage:

```python
kernel_mapper = tf.contrib.kernel_methods.SomeKernelMapper(...)
out_tensor = kernel_mapper.map(in_tensor)
...  # code that consumes out_tensor.
```

Currently, there is a [RandomFourierFeatureMapper](https://www.tensorflow.org/code/tensorflow/contrib/kernel_methods/python/mappers/random_fourier_features.py) implemented that maps dense input to dense
output. More mappers are on the way.

## Kernel-based Estimators

These estimators inherit from the
[`tf.contrib.learn.Estimator`](https://www.tensorflow.org/code/tensorflow/contrib/learn/python/learn/estimators/estimator.py)
class and use kernel mappers internally to discover non-linearities in the
data. These canned estimators map their input features using kernel mapper
Ops and then apply linear models to the mapped features. Combining kernel
mappers with linear models and different loss functions leads to a variety of
models: linear and non-linear SVMs, linear regression (with and without
kernels) and (multinomial) logistic regression (with and without kernels).

Currently there is a [KernelLinearClassifier](https://www.tensorflow.org/code/tensorflow/contrib/kernel_methods/python/kernel_estimators.py) implemented but more pre-packaged estimators
are on the way.

Sample usage:

```python
real_column_a = tf.contrib.layers.real_valued_column(name='real_column_a',...)
sparse_column_b = tf.contrib.layers.sparse_column_with_hash_bucket(...)
kernel_mappers = {real_column_a : [tf.contrib.kernel_methods.SomeKernelMapper(...)]}
optimizer = ...

kernel_classifier = tf.contrib.kernel_methods.KernelLinearClassifier(
    feature_columns=[real_column_a, sparse_column_b],
    model_dir=...,
    optimizer=optimizer,
    kernel_mappers=kernel_mappers)

# Construct input_fns
kernel_classifier.fit(...)
kernel_classifier.evaluate(...)
```

