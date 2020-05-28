# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Benchmarks for low-level eager execution primitives.

To run CPU benchmarks:
  bazel run -c opt benchmarks_test -- --benchmarks=.

To run GPU benchmarks:
  bazel run --config=cuda -c opt --copt="-mavx" benchmarks_test -- \
    --benchmarks=.

To run a subset of benchmarks using --benchmarks flag.
--benchmarks: the list of benchmarks to run. The specified value is interpreted
as a regular expression and any benchmark whose name contains a partial match
to the regular expression is executed.
e.g. --benchmarks=".*matmul*." will run all matmul related benchmarks.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.eager import profiler
from tensorflow.python.eager import test
from tensorflow.python.ops import random_ops
from tensorflow.python.training import gradient_descent


class SubclassedKerasModel(keras.Model):

  def __init__(self, initializer="ones"):
    super(SubclassedKerasModel, self).__init__()
    self.layer_a = keras.layers.Dense(
        64, kernel_initializer=initializer, bias_initializer="zeros")
    self.layer_b = keras.layers.Dense(
        128, kernel_initializer=initializer, bias_initializer="zeros")
    self.layer_c = keras.layers.Dense(
        256, kernel_initializer=initializer, bias_initializer="zeros")
    self.layer_d = keras.layers.Dense(
        256, kernel_initializer=initializer, bias_initializer="zeros")
    self.layer_e = keras.layers.Dense(
        10, kernel_initializer=initializer, bias_initializer="zeros")

  def call(self, x):
    x = self.layer_a(x)
    x = self.layer_b(x)
    x = self.layer_c(x)
    x = self.layer_d(x)
    return self.layer_e(x)


def make_keras_model(initializer="ones"):
  model_input = keras.Input(shape=(10,))
  x = keras.layers.Dense(
      64, kernel_initializer=initializer, bias_initializer="zeros")(model_input)
  x = keras.layers.Dense(
      128, kernel_initializer=initializer, bias_initializer="zeros")(x)
  x = keras.layers.Dense(
      256, kernel_initializer=initializer, bias_initializer="zeros")(x)
  x = keras.layers.Dense(
      256, kernel_initializer=initializer, bias_initializer="zeros")(x)
  x = keras.layers.Dense(
      10, kernel_initializer=initializer, bias_initializer="zeros")(x)
  return keras.Model(inputs=model_input, outputs=x)


def make_sequential_keras_model(initializer="ones"):
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(
      64, kernel_initializer=initializer, bias_initializer="zeros",
      input_shape=(10,)))
  model.add(keras.layers.Dense(
      128, kernel_initializer=initializer, bias_initializer="zeros"))
  model.add(keras.layers.Dense(
      256, kernel_initializer=initializer, bias_initializer="zeros"))
  model.add(keras.layers.Dense(
      256, kernel_initializer=initializer, bias_initializer="zeros"))
  model.add(keras.layers.Dense(
      10, kernel_initializer=initializer, bias_initializer="zeros"))
  return model


def run_benchmark(func, num_iters, execution_mode=None):
  ctx = context.context()
  with context.execution_mode(execution_mode):
    # call func to warm up
    func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    start = time.time()
    for _ in xrange(num_iters):
      func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    end = time.time()

    return end - start


class MicroBenchmarks(test.Benchmark):

  def _run(self, func, num_iters, execution_mode=None):
    total_time = run_benchmark(func, num_iters, execution_mode)
    mean_us = total_time * 1e6 / num_iters
    self.report_benchmark(
        iters=num_iters,
        wall_time=mean_us,
        extras={
            "examples_per_sec":
                float("{0:.3f}".format(num_iters / total_time)),
            "us_per_example":
                float("{0:.3f}".format(total_time * 1e6 / num_iters))
        })

  def benchmark_keras_model_subclassed(self):
    model = SubclassedKerasModel()
    data = random_ops.random_uniform((10, 10))

    func = lambda: model(data)
    # First call is more expensive (creates variables etc.), discount that.
    func()

    # The whole point of this test is to contrast subclassing with
    # the functional style of keras model building, so validate that
    # the models are equivalent.
    assert np.equal(func(), make_keras_model()(data)).all()

    self._run(func, 30000)

  def benchmark_keras_model_functional(self):
    model = make_keras_model()
    data = random_ops.random_uniform((10, 10))
    func = lambda: model(data)
    # Symmetry with benchmark_keras_model_subclassed
    func()
    assert np.equal(func(), SubclassedKerasModel()(data)).all()
    self._run(func, 30000)

  def benchmark_keras_model_sequential(self):
    model = make_sequential_keras_model()
    data = random_ops.random_uniform((10, 10))
    func = lambda: model(data)
    # Symmetry with benchmark_keras_model_functional
    func()
    assert np.equal(func(), make_keras_model()(data)).all()
    self._run(func, 30000)

  def _benchmark_keras_model_fit(self, model, run_eagerly=False):
    data = random_ops.random_uniform((10, 10), minval=-1, maxval=1)
    labels = random_ops.random_uniform((10, 10), minval=-1, maxval=1)
    dataset = dataset_ops.Dataset.from_tensors((data, labels)).repeat()
    model.compile(
        gradient_descent.GradientDescentOptimizer(learning_rate=0.001),
        loss="mse", run_eagerly=run_eagerly)
    func = lambda: model.fit(dataset, epochs=1, steps_per_epoch=1000, verbose=0)
    # First call is more expensive (creates variables etc.), discount that.
    model.fit(dataset, epochs=1, steps_per_epoch=1, verbose=0)

    self._run(func, 1)

  def _benchmark_keras_model_evaluate(self, model, run_eagerly=False):
    data = random_ops.random_uniform((10, 10), minval=-1, maxval=1)
    labels = random_ops.random_uniform((10, 10), minval=-1, maxval=1)
    dataset = dataset_ops.Dataset.from_tensors((data, labels)).repeat()
    model.compile(
        gradient_descent.GradientDescentOptimizer(learning_rate=0.001),
        loss="mse", run_eagerly=run_eagerly)
    func = lambda: model.evaluate(dataset, steps=1000, verbose=0)
    # First call is more expensive (creates variables etc.), discount that.
    model.evaluate(dataset, steps=1, verbose=0)

    self._run(func, 1)

  def _benchmark_keras_model_predict(self, model, run_eagerly=False):
    data = random_ops.random_uniform((10, 10), minval=-1, maxval=1)
    dataset = dataset_ops.Dataset.from_tensors(data).repeat()
    model.compile(
        gradient_descent.GradientDescentOptimizer(learning_rate=0.001),
        loss="mse", run_eagerly=run_eagerly)
    func = lambda: model.predict(dataset, steps=1000, verbose=0)
    # First call is more expensive (creates variables etc.), discount that.
    model.predict(dataset, steps=1, verbose=0)

    self._run(func, 1)

  def benchmark_keras_model_subclassed_fit(self):
    model = SubclassedKerasModel(initializer="glorot_uniform")
    self._benchmark_keras_model_fit(model)

  def benchmark_keras_model_subclassed_fit_graph_mode(self):
    with context.graph_mode():
      model = SubclassedKerasModel(initializer="glorot_uniform")
      self._benchmark_keras_model_fit(model)

  def benchmark_keras_model_subclassed_fit_run_model_eagerly(self):
    model = SubclassedKerasModel(initializer="glorot_uniform")
    self._benchmark_keras_model_fit(model, run_eagerly=True)

  def benchmark_keras_model_functional_fit(self):
    model = make_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_fit(model)

  def benchmark_keras_model_functional_fit_graph_mode(self):
    with context.graph_mode():
      model = make_keras_model(initializer="glorot_uniform")
      self._benchmark_keras_model_fit(model)

  def benchmark_keras_model_functional_fit_graph_mode_with_profiler(self):
    profiler.start()
    with context.graph_mode():
      model = make_keras_model(initializer="glorot_uniform")
      self._benchmark_keras_model_fit(model)
    result = profiler.stop()
    assert result is not None

  def benchmark_keras_model_functional_fit_run_model_eagerly(self):
    model = make_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_fit(model, run_eagerly=True)

  def benchmark_keras_model_functional_fit_run_model_eagerly_with_profiler(
      self):
    profiler.start()
    model = make_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_fit(model, run_eagerly=True)
    result = profiler.stop()
    assert result is not None

  def benchmark_keras_model_sequential_fit(self):
    model = make_sequential_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_fit(model)

  def benchmark_keras_model_sequential_fit_graph_mode(self):
    with context.graph_mode():
      model = make_sequential_keras_model(initializer="glorot_uniform")
      self._benchmark_keras_model_fit(model)

  def benchmark_keras_model_sequential_fit_run_model_eagerly(self):
    model = make_sequential_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_fit(model, run_eagerly=True)

  def benchmark_keras_model_subclassed_evaluate(self):
    model = SubclassedKerasModel(initializer="glorot_uniform")
    self._benchmark_keras_model_evaluate(model)

  def benchmark_keras_model_subclassed_evaluate_run_model_eagerly(self):
    model = SubclassedKerasModel(initializer="glorot_uniform")
    self._benchmark_keras_model_evaluate(model, run_eagerly=True)

  def benchmark_keras_model_functional_evaluate(self):
    model = make_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_evaluate(model)

  def benchmark_keras_model_functional_evaluate_run_model_eagerly(self):
    model = make_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_evaluate(model, run_eagerly=True)

  def benchmark_keras_model_sequential_evaluate(self):
    model = make_sequential_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_evaluate(model)

  def benchmark_keras_model_sequential_evaluate_run_model_eagerly(self):
    model = make_sequential_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_evaluate(model, run_eagerly=True)

  def benchmark_keras_model_subclassed_predict(self):
    model = SubclassedKerasModel(initializer="glorot_uniform")
    self._benchmark_keras_model_predict(model)

  def benchmark_keras_model_subclassed_predict_run_model_eagerly(self):
    model = SubclassedKerasModel(initializer="glorot_uniform")
    self._benchmark_keras_model_predict(model, run_eagerly=True)

  def benchmark_keras_model_functional_predict(self):
    model = make_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_predict(model)

  def benchmark_keras_model_functional_predict_run_model_eagerly(self):
    model = make_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_predict(model, run_eagerly=True)

  def benchmark_keras_model_sequential_predict(self):
    model = make_sequential_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_predict(model)

  def benchmark_keras_model_sequential_predict_run_model_eagerly(self):
    model = make_sequential_keras_model(initializer="glorot_uniform")
    self._benchmark_keras_model_predict(model, run_eagerly=True)


if __name__ == "__main__":
  test.main()
