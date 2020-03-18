# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Scikit-learn API wrapper."""

# build_fn's in this file have unused arguments to keep a single API
# pylint: disable=unused-argument
# allow use of "X" as a variable
# pylint: disable=invalid-name
# pylint does not like assignment of __call__
# pylint: disable=method-hidden

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_boston, load_digits, load_iris
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import r2_score as sklearn_r2_score
from sklearn.metrics import accuracy_score as sklearn_accuracy_score

from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python import keras
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.wrappers import scikit_learn
from tensorflow.python.platform import test
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
    Flatten,
    Input,
)
from tensorflow.python.keras.models import Model, Sequential, clone_model
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.wrappers.scikit_learn import (
    KerasClassifier,
    KerasRegressor,
    _r2_score,
    _accuracy_score,
)


INPUT_DIM = 5
HIDDEN_DIM = 5
TRAIN_SAMPLES = 10
TEST_SAMPLES = 5
NUM_CLASSES = 2
BATCH_SIZE = 5
EPOCHS = 1


def build_fn_clf(hidden_dim):
  """Builds a Sequential based classifier."""
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(INPUT_DIM, input_shape=(INPUT_DIM,)))
  model.add(keras.layers.Activation("relu"))
  model.add(keras.layers.Dense(hidden_dim))
  model.add(keras.layers.Activation("relu"))
  model.add(keras.layers.Dense(NUM_CLASSES))
  model.add(keras.layers.Activation("softmax"))
  model.compile(
      optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
  )
  return model


def assert_classification_works(clf):
  """Checks a classification task for errors."""
  np.random.seed(42)
  (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
      train_samples=TRAIN_SAMPLES,
      test_samples=TEST_SAMPLES,
      input_shape=(INPUT_DIM,),
      num_classes=NUM_CLASSES,
  )

  clf.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

  score = clf.score(x_train, y_train, batch_size=BATCH_SIZE)
  assert np.isscalar(score) and np.isfinite(score)

  preds = clf.predict(x_test, batch_size=BATCH_SIZE)
  assert preds.shape == (TEST_SAMPLES,)
  for prediction in np.unique(preds):
    assert prediction in range(NUM_CLASSES)

  proba = clf.predict_proba(x_test, batch_size=BATCH_SIZE)
  assert proba.shape == (TEST_SAMPLES, NUM_CLASSES)
  assert np.allclose(np.sum(proba, axis=1), np.ones(TEST_SAMPLES))


def build_fn_reg(hidden_dim):
  """Builds a Sequential based regressor."""
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(INPUT_DIM, input_shape=(INPUT_DIM,)))
  model.add(keras.layers.Activation("relu"))
  model.add(keras.layers.Dense(hidden_dim))
  model.add(keras.layers.Activation("relu"))
  model.add(keras.layers.Dense(1))
  model.add(keras.layers.Activation("linear"))
  model.compile(
      optimizer="sgd", loss="mean_absolute_error", metrics=["accuracy"]
  )
  return model


def assert_regression_works(reg):
  """Checks a regression task for errors."""
  np.random.seed(42)
  (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
      train_samples=TRAIN_SAMPLES,
      test_samples=TEST_SAMPLES,
      input_shape=(INPUT_DIM,),
      num_classes=NUM_CLASSES,
  )

  reg.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

  score = reg.score(x_train, y_train, batch_size=BATCH_SIZE)
  assert np.isscalar(score) and np.isfinite(score)

  preds = reg.predict(x_test, batch_size=BATCH_SIZE)
  assert preds.shape == (TEST_SAMPLES,)


class ScikitLearnAPIWrapperTest(test.TestCase):
  """Tests basic functionality."""

  def test_classify_build_fn(self):
    """Tests a classification task for errors."""
    with self.cached_session():
      clf = scikit_learn.KerasClassifier(
          build_fn=build_fn_clf,
          hidden_dim=HIDDEN_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
      )

      assert_classification_works(clf)

  def test_classify_class_build_fn(self):
    """Tests for errors using a class implementing __call__."""
    class ClassBuildFnClf(object):  # pylint:disable=useless-object-inheritance
      def __call__(self, hidden_dim):
        return build_fn_clf(hidden_dim)

    with self.cached_session():
      clf = scikit_learn.KerasClassifier(
          build_fn=ClassBuildFnClf(),
          hidden_dim=HIDDEN_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
      )

      assert_classification_works(clf)

  def test_classify_inherit_class_build_fn(self):
    """Tests for errors using an inherited class."""
    class InheritClassBuildFnClf(scikit_learn.KerasClassifier):
      def __call__(self, hidden_dim):
        return build_fn_clf(hidden_dim)

    with self.cached_session():
      clf = InheritClassBuildFnClf(
          build_fn=None,
          hidden_dim=HIDDEN_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
      )

      assert_classification_works(clf)

  def test_regression_build_fn(self):
    """Tests for errors using KerasRegressor."""
    with self.cached_session():
      reg = scikit_learn.KerasRegressor(
          build_fn=build_fn_reg,
          hidden_dim=HIDDEN_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
      )

      assert_regression_works(reg)

  def test_regression_class_build_fn(self):
    """Tests for errors using KerasRegressor implementing __call__."""
    class ClassBuildFnReg(object):  # pylint:disable=useless-object-inheritance
      def __call__(self, hidden_dim):
        return build_fn_reg(hidden_dim)

    with self.cached_session():
      reg = scikit_learn.KerasRegressor(
          build_fn=ClassBuildFnReg(),
          hidden_dim=HIDDEN_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
      )

      assert_regression_works(reg)

  def test_regression_inherit_class_build_fn(self):
    """Tests for errors using KerasRegressor inherited."""
    class InheritClassBuildFnReg(scikit_learn.KerasRegressor):
      def __call__(self, hidden_dim):
        return build_fn_reg(hidden_dim)

    with self.cached_session():
      reg = InheritClassBuildFnReg(
          build_fn=None,
          hidden_dim=HIDDEN_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
      )

      assert_regression_works(reg)


def load_digits8x8():
  """Load image 8x8 dataset."""
  data = load_digits()
  data.data = data.data.reshape([data.data.shape[0], 1, 8, 8]) / 16.0
  # Convert NCHW to NHWC
  # Convert back to numpy or sklearn funcs (GridSearchCV, etc.) WILL fail
  data.data = np.transpose(data.data, [0, 2, 3, 1])
  K.set_image_data_format("channels_last")
  return data


def check(estimator, loader):
  """Run basic checks (fit, score, pickle) on estimator."""
  data = loader()
  # limit to 100 data points to speed up testing
  X, y = data.data[:100], data.target[:100]  # pylint:disable=invalid-name
  estimator.fit(X, y)
  estimator.predict(X)
  score = estimator.score(X, y)
  serialized_estimator = pickle.dumps(estimator)
  deserialized_estimator = pickle.loads(serialized_estimator)
  deserialized_estimator.predict(X)
  score_new = deserialized_estimator.score(X, y)
  np.testing.assert_almost_equal(score, score_new)
  assert True


def build_fn_regs(X, n_outputs_, hidden_layer_sizes=None, n_classes_=None):
  """Dynamically build regressor."""
  if hidden_layer_sizes is None:
    hidden_layer_sizes = []
  model = Sequential()
  model.add(Dense(X.shape[1], activation="relu", input_shape=X.shape[1:]))
  for size in hidden_layer_sizes:
    model.add(Dense(size, activation="relu"))
  model.add(Dense(n_outputs_))
  model.compile("adam", loss="mean_squared_error")
  return model


def build_fn_clss(X, n_outputs_, hidden_layer_sizes=None, n_classes_=None):
  """Dynamically build classifier."""
  if hidden_layer_sizes is None:
    hidden_layer_sizes = []
  model = Sequential()
  model.add(Dense(X.shape[1], activation="relu", input_shape=X.shape[1:]))
  for size in hidden_layer_sizes:
    model.add(Dense(size, activation="relu"))
  model.add(Dense(1, activation="softmax"))
  model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
  return model


def build_fn_clscs(X, n_outputs_, hidden_layer_sizes=None, n_classes_=None):
  """Dynamically build functional API regressor."""
  if hidden_layer_sizes is None:
    hidden_layer_sizes = []
  model = Sequential()
  model.add(Conv2D(3, (3, 3), input_shape=X.shape[1:]))
  model.add(Flatten())
  for size in hidden_layer_sizes:
    model.add(Dense(size, activation="relu"))
  model.add(Dense(n_classes_, activation="softmax"))
  model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
  return model


def build_fn_clscf(X, n_outputs_, hidden_layer_sizes=None, n_classes_=None):
  """Dynamically build functional API classifier."""
  if hidden_layer_sizes is None:
    hidden_layer_sizes = []
  x = Input(shape=X.shape[1:])
  z = Conv2D(3, (3, 3))(x)
  z = Flatten()(z)
  for size in hidden_layer_sizes:
    z = Dense(size, activation="relu")(z)
  y = Dense(n_classes_, activation="softmax")(z)
  model = Model(inputs=x, outputs=y)
  model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
  return model


CONFIG = {
    "MLPRegressor": (
        load_boston,
        KerasRegressor,
        build_fn_regs,
        (BaggingRegressor, AdaBoostRegressor),
    ),
    "MLPClassifier": (
        load_iris,
        KerasClassifier,
        build_fn_clss,
        (BaggingClassifier, AdaBoostClassifier),
    ),
    "CNNClassifier": (
        load_digits8x8,
        KerasClassifier,
        build_fn_clscs,
        (BaggingClassifier, AdaBoostClassifier),
    ),
    "CNNClassifierF": (
        load_digits8x8,
        KerasClassifier,
        build_fn_clscf,
        (BaggingClassifier, AdaBoostClassifier),
    ),
}


class ScikitLearnAPIWrapperTestAdvFunc(test.TestCase):
  """Tests advanced features such as pipelines and hyperparameter tuning."""

  def test_standalone(self):
    """Tests standalone estimator."""
    for config in [
        "MLPRegressor",
        "MLPClassifier",
        "CNNClassifier",
        "CNNClassifierF",
    ]:
      loader, model, build_fn, _ = CONFIG[config]
      estimator = model(build_fn, epochs=1)
      check(estimator, loader)

  def test_pipeline(self):
    """Tests compatibility with Scikit-learn's pipeline."""
    for config in ["MLPRegressor", "MLPClassifier"]:
      loader, model, build_fn, _ = CONFIG[config]
      estimator = model(build_fn, epochs=1)
      estimator = Pipeline([("s", StandardScaler()), ("e", estimator)])
      check(estimator, loader)

  def test_searchcv(self):
    """Tests compatibility with Scikit-learn's hyperparameter search CV."""
    for config in [
        "MLPRegressor",
        "MLPClassifier",
        "CNNClassifier",
        "CNNClassifierF",
    ]:
      loader, model, build_fn, _ = CONFIG[config]
      estimator = model(
          build_fn, epochs=1, validation_split=0.1, hidden_layer_sizes=[]
      )
      check(GridSearchCV(estimator, {"hidden_layer_sizes": [[], [5]]}), loader)
      check(
          RandomizedSearchCV(
              estimator,
              {"epochs": np.random.randint(1, 5, 2),},
              n_iter=2
          ),
          loader,
      )

  def test_ensemble(self):
    """Tests compatibility with Scikit-learn's ensembles."""
    for config in ["MLPRegressor", "MLPClassifier"]:
      loader, model, build_fn, ensembles = CONFIG[config]
      base_estimator = model(build_fn, epochs=1)
      for ensemble in ensembles:
        estimator = ensemble(base_estimator=base_estimator, n_estimators=2)
        check(estimator, loader)

  def test_calibratedclassifiercv(self):
    """Tests compatibility with Scikit-learn's calibrated classifier CV."""
    for config in ["MLPClassifier"]:
      loader, _, build_fn, _ = CONFIG[config]
      base_estimator = KerasClassifier(build_fn, epochs=1)
      estimator = CalibratedClassifierCV(base_estimator=base_estimator, cv=5)
      check(estimator, loader)


class SentinalCallback(keras.callbacks.Callback):
  """
  Callback class that sets an internal value once it's been acessed.
  """

  called = 0

  def on_train_begin(self, logs=None):
    """Increments counter."""
    self.called += 1


class ClassWithCallback(scikit_learn.KerasClassifier):
  """Must be defined at top level to be picklable.
  """

  def __init__(self, **sk_params):
    self.callbacks = [SentinalCallback()]
    super(ClassWithCallback, self).__init__(**sk_params)

  def __call__(self, hidden_dim):
    return build_fn_clf(hidden_dim)


class ScikitLearnAPIWrapperTestCallbacks(test.TestCase):
  """Tests use of Callbacks."""

  def test_callbacks_passed_as_arg(self):
    """Tests estimators created passing a callback to __init__."""
    for config in [
        "MLPRegressor",
        "MLPClassifier",
        "CNNClassifier",
        "CNNClassifierF",
    ]:
      loader, model, build_fn, _ = CONFIG[config]
      callback = SentinalCallback()
      estimator = model(build_fn, epochs=1, callbacks=[callback])
      # check that callback did not break estimator
      check(estimator, loader)
      # check that callback is preserved after pickling
      data = loader()
      X, y = data.data[:100], data.target[:100]
      estimator.fit(X, y)
      assert estimator.callbacks[0].called != SentinalCallback.called
      old_callback = estimator.callbacks[0]
      deserialized_estimator = pickle.loads(pickle.dumps(estimator))
      assert deserialized_estimator.callbacks[0].called == old_callback.called

  def test_callbacks_inherit(self):
    """Test estimators that inherit from KerasClassifier and implement
    their own callbacks in their __init___.
    """
    with self.cached_session():
      clf = ClassWithCallback(
          hidden_dim=HIDDEN_DIM, batch_size=BATCH_SIZE, epochs=EPOCHS
      )

      assert_classification_works(clf)
      assert clf.callbacks[0].called != SentinalCallback.called
      serialized_estimator = pickle.dumps(clf)
      deserialized_estimator = pickle.loads(serialized_estimator)
      assert deserialized_estimator.callbacks[0].called == \
           clf.callbacks[0].called
      assert_classification_works(deserialized_estimator)


class ScikitLearnAPIWrapperTestSampleWeights(test.TestCase):
  """Tests involving the sample_weight parameter.
     TODO: fix warning regarding sample_weight shape coercing.
  """

  @staticmethod
  def check_sample_weights_work(estimator):
    """Checks that using the parameter sample_weight does not cause
    issues (it does not check if the parameter actually works as intended).
    """
    (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=NUM_CLASSES,
    )
    s_w_train = np.arange(TRAIN_SAMPLES, dtype=float)

    # check that we can train with sample weights
    # TODO: how do we reliably check the effect of training with sample_weights?
    estimator.fit(x_train, y_train, sample_weight=s_w_train)
    estimator.predict(x_test)

    # now train with no sample weights, test scoring
    estimator.fit(x_train, y_train, sample_weight=None)
    # re-use training data to try to get score > 0
    score_n_w = estimator.score(x_train, y_train)
    score_w = estimator.score(x_train, y_train, sample_weight=s_w_train)
    # check that sample weights did *something*
    try:
      np.testing.assert_array_almost_equal(score_n_w, score_w)
    except AssertionError:
      return

    raise AssertionError("`sample_weight` seemed to have no effect.")

  def test_classify_build_fn(self):
    with self.cached_session():
      clf = scikit_learn.KerasClassifier(
          build_fn=build_fn_clf,
          hidden_dim=HIDDEN_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
      )
      self.check_sample_weights_work(clf)

  def test_reg_build_fn(self):
    with self.cached_session():
      clf = scikit_learn.KerasRegressor(
          build_fn=build_fn_reg,
          hidden_dim=HIDDEN_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
      )
      self.check_sample_weights_work(clf)


def dynamic_classifier(X, y, n_classes_):
  """Creates a basic MLP classifier dynamically choosing binary/multiclass
  classification loss and ouput activations.
  """
  n_features = X.shape[1]
  if n_classes_ == 2:
    # binary
    loss = "binary_crossentropy"
    output_activation = "sigmoid"
    output_size = 1
  else:
    loss = "categorical_crossentropy"
    output_activation = "softmax"
    output_size = n_classes_
  n_features = X.shape[1]
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(n_features, input_shape=(n_features,)))
  model.add(keras.layers.Activation("relu"))
  model.add(keras.layers.Dense(100))
  model.add(keras.layers.Activation("relu"))
  model.add(keras.layers.Dense(output_size))
  model.add(keras.layers.Activation(output_activation))
  model.compile(optimizer="sgd", loss=loss, metrics=["accuracy"])
  return model


class ScikitLearnAPIWrapperTestYShapes(test.TestCase):
  """Tests that compare output shapes to `MLPClassifier` from sklearn to
     check that ouput shapes respect sklearn API.
  """

  @classmethod
  def setUpClass(cls):
    cls.keras_clf = KerasClassifier(build_fn=dynamic_classifier)
    cls.sklearn_clf = MLPClassifier()

  def test_1d_multiclass(self):
    """Compares KerasClassifier prediction output shape against
    sklearn.neural_net.MPLClassifier for 1D multi-class (n_samples,).
    """
    # crate 1D multiclass labels
    (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=4,
    )
    self.keras_clf.fit(X=x_train, y=y_train)
    self.sklearn_clf.fit(X=x_train, y=y_train)
    y_pred_keras = self.keras_clf.predict(X=x_test)
    y_pred_sklearn = self.sklearn_clf.predict(X=x_test)
    assert y_pred_keras.shape == y_pred_sklearn.shape
    y_pred_prob_keras = self.keras_clf.predict_proba(X=x_test)
    y_pred_prob_sklearn = self.sklearn_clf.predict_proba(X=x_test)
    assert y_pred_prob_keras.shape == y_pred_prob_sklearn.shape

  def test_2d_multiclass(self):
    """Compares KerasClassifier prediction output shape against
    sklearn.neural_net.MPLClassifier for 2D multi-class (n_samples,1).
    """
    # crate 2D multiclass labels
    (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=4,
    )
    y_train = y_train.reshape(-1, 1)
    self.keras_clf.fit(X=x_train, y=y_train)
    self.sklearn_clf.fit(X=x_train, y=y_train)
    y_pred_keras = self.keras_clf.predict(X=x_test)
    y_pred_sklearn = self.sklearn_clf.predict(X=x_test)
    assert y_pred_keras.shape == y_pred_sklearn.shape
    y_pred_prob_keras = self.keras_clf.predict_proba(X=x_test)
    y_pred_prob_sklearn = self.sklearn_clf.predict_proba(X=x_test)
    assert y_pred_prob_keras.shape == y_pred_prob_sklearn.shape

  def test_1d_binary(self):
    """Compares KerasClassifier prediction output shape against
    sklearn.neural_net.MPLClassifier for binary (n_samples,).
    """
    # create 1D binary labels
    (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=2,
    )
    self.keras_clf.fit(X=x_train, y=y_train)
    self.sklearn_clf.fit(X=x_train, y=y_train)
    y_pred_keras = self.keras_clf.predict(X=x_test)
    y_pred_sklearn = self.sklearn_clf.predict(X=x_test)
    assert y_pred_keras.shape == y_pred_sklearn.shape
    y_pred_prob_keras = self.keras_clf.predict_proba(X=x_test)
    y_pred_prob_sklearn = self.sklearn_clf.predict_proba(X=x_test)
    assert y_pred_prob_keras.shape == y_pred_prob_sklearn.shape

  def test_2d_binary(self):
    """Compares KerasClassifier prediction output shape against
    sklearn.neural_net.MPLClassifier for 2D binary (n_samples,1).
    """
    # create 2D binary labels
    (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=2,
    )
    y_train = y_train.reshape(-1, 1)
    self.keras_clf.fit(X=x_train, y=y_train)
    self.sklearn_clf.fit(X=x_train, y=y_train)
    y_pred_keras = self.keras_clf.predict(X=x_test)
    y_pred_sklearn = self.sklearn_clf.predict(X=x_test)
    assert y_pred_keras.shape == y_pred_sklearn.shape
    y_pred_prob_keras = self.keras_clf.predict_proba(X=x_test)
    y_pred_prob_sklearn = self.sklearn_clf.predict_proba(X=x_test)
    assert y_pred_prob_keras.shape == y_pred_prob_sklearn.shape


class ScikitLearnAPIWrapperTestPrebuiltModel(test.TestCase):
  """Tests using a prebuilt model instance."""

  def test_prebuilt_model(self):
    """Tests using a prebuilt model."""
    for config in [
        "MLPRegressor",
        "MLPClassifier",
        "CNNClassifier",
        "CNNClassifierF",
    ]:
      loader, model, build_fn, _ = CONFIG[config]
      data = loader()
      x_train, y_train = data.data[:100], data.target[:100]

      n_classes_ = np.unique(y_train).size
      # make y the same shape as will be used by .fit
      if config != "MLPRegressor":
        y_train = to_categorical(y_train)
        keras_model = build_fn(X=x_train, n_classes_=n_classes_, n_outputs_=1)
      else:
        keras_model = build_fn(X=x_train, n_outputs_=1)

      estimator = model(build_fn=keras_model)
      check(estimator, loader)

  def test_uncompiled_prebuilt_model_raises_error(self):
    """Tests that an uncompiled model cannot be used as build_fn param."""

    for config in [
        "MLPRegressor",
        "MLPClassifier",
        "CNNClassifier",
        "CNNClassifierF",
    ]:
      loader, model, build_fn, _ = CONFIG[config]
      data = loader()
      x_train, y_train = data.data[:100], data.target[:100]

      n_classes_ = np.unique(y_train).size
      # make y the same shape as will be used by .fit
      if config != "MLPRegressor":
        y_train = to_categorical(y_train)
        keras_model = build_fn(X=x_train, n_classes_=n_classes_, n_outputs_=1)
      else:
        keras_model = build_fn(X=x_train, n_outputs_=1)

      # clone to simulate uncompiled model
      keras_model = clone_model(keras_model)
      estimator = model(build_fn=keras_model)
      with self.assertRaises(ValueError):
        check(estimator, loader)

class FunctionAPIMultiInputMultiOutputClassifier(KerasClassifier):
  """Tests Functional API Classifier with 2 inputs and 2 outputs
  """

  def __call__(self, X, n_classes_):  # pylint:disable=invalid-name
    inp1 = Input((1,))
    inp2 = Input((3,))

    x1 = Dense(100)(inp1)
    x2 = Dense(100)(inp2)

    x3 = Concatenate(axis=-1)([x1, x2])

    binary_out = Dense(1, activation="sigmoid")(x3)
    cat_out = Dense(n_classes_[1], activation="softmax")(x3)

    model = Model([inp1, inp2], [binary_out, cat_out])
    losses = ["binary_crossentropy", "categorical_crossentropy"]
    model.compile(optimizer="adam", loss=losses, metrics=["accuracy"])

    return model

  @staticmethod
  def _pre_process_X(X):
    """To support multiple inputs, a custom method must be defined.
    """
    return [(X[:, 0], X[:, 1:4])], dict()


class FunctionAPIMultiLabelClassifier(KerasClassifier):
  """Tests Functional API Classifier with 2 inputs and 2 outputs
  """

  def __call__(self, X, n_outputs_):  # pylint:disable=invalid-name
    inp = Input((4,))

    x1 = Dense(100)(inp)

    outputs = []
    for _ in range(n_outputs_):
      # simulate multiple binary classification outputs
      # in reality, these would come from different nodes
      outputs.append(Dense(1, activation="sigmoid")(x1))

    model = Model(inp, outputs)
    losses = "binary_crossentropy"
    model.compile(optimizer="adam", loss=losses, metrics=["accuracy"])

    return model


class FunctionAPIMultiOutputRegressor(KerasRegressor):
  """Tests Functional API Regressor with multiple outputs.
  """

  def __call__(self, X, n_outputs_):  # pylint:disable=invalid-name
    inp = Input((INPUT_DIM,))

    x1 = Dense(100)(inp)

    outputs = []
    for _ in range(n_outputs_):
      # simulate multiple binary classification outputs
      # in reality, these would come from different nodes
      outputs.append(Dense(1)(x1))

    model = Model([inp], outputs)
    losses = "mean_squared_error"
    model.compile(optimizer="adam", loss=losses, metrics=["mse"])

    return model

@keras.utils.generic_utils.register_keras_serializable()
class CustomLoss(keras.losses.MeanSquaredError):
  """Dummy custom loss."""

@keras.utils.generic_utils.register_keras_serializable()
class CustomModelRegistered(Model):
  """Dummy custom Model subclass that is registered to be serializable."""

class CustomModelUnregistered(Model):
  """Dummy custom Model subclass that is not registered to be serializable."""

def build_fn_regs_custom_loss(X, n_outputs_, hidden_layer_sizes=None):
  """Build regressor with subclassed loss function."""
  if hidden_layer_sizes is None:
    hidden_layer_sizes = []
  model = Sequential()
  model.add(Dense(X.shape[1], activation="relu", input_shape=X.shape[1:]))
  for size in hidden_layer_sizes:
    model.add(Dense(size, activation="relu"))
  model.add(Dense(n_outputs_))
  model.compile("adam", loss=CustomLoss())
  return model

def build_fn_regs_custom_model_reg(X, n_outputs_, hidden_layer_sizes=None):
  """Build regressor with subclassed Model registered as serializable."""
  if hidden_layer_sizes is None:
    hidden_layer_sizes = []
  x = Input(shape=X.shape[1])
  z = Dense(X.shape[1], activation="relu")(x)
  for size in hidden_layer_sizes:
    z = Dense(size, activation="relu")(z)
  y = Dense(n_outputs_, activation="linear")(z)
  model = CustomModelRegistered(inputs=x, outputs=y)
  model.compile("adam", loss="mean_squared_error")
  return model

def build_fn_regs_custom_model_unreg(X, n_outputs_, hidden_layer_sizes=None):
  """Build regressor with subclassed Model not registered as serializable."""
  if hidden_layer_sizes is None:
    hidden_layer_sizes = []
  x = Input(shape=X.shape[1])
  z = Dense(X.shape[1], activation="relu")(x)
  for size in hidden_layer_sizes:
    z = Dense(size, activation="relu")(z)
  y = Dense(n_outputs_, activation="linear")(z)
  model = CustomModelUnregistered(inputs=x, outputs=y)
  model.compile("adam", loss="mean_squared_error")
  return model

class SerializeCustomLayers(test.TestCase):
  """Tests serializing custom layers."""

  def test_custom_loss_function(self):
    """Test that a registered subclassed Model can be serialized."""
    estimator = KerasRegressor(build_fn=build_fn_regs_custom_loss)
    check(estimator, load_boston)

  def test_custom_model_registered(self):
    """Test that a registered subclassed loss function can be serialized."""
    estimator = KerasRegressor(build_fn=build_fn_regs_custom_model_reg)
    check(estimator, load_boston)

  def test_custom_model_unregistered(self):
    """Test that an unregistered subclassed Model raises an error."""
    estimator = KerasRegressor(build_fn=build_fn_regs_custom_model_unreg)
    with self.assertRaises(ValueError):
      check(estimator, load_boston)

class ScikitLearnAPIWrapperScoring(test.TestCase):
  """Tests scoring methods.
  """

  def test_scoring_r2(self):
    """Test custom R^2 implementation against scikit-learn's."""
    n_samples = 50

    datasets = []
    y_true = np.arange(n_samples, dtype=float)
    y_pred = y_true + 1
    datasets.append((y_true.reshape(-1, 1), y_pred.reshape(-1, 1)))
    y_true = np.random.random_sample(size=y_true.shape)
    y_pred = np.random.random_sample(size=y_true.shape)
    datasets.append((y_true.reshape(-1, 1), y_pred.reshape(-1, 1)))

    def keras_backend_r2(y_true, y_pred):
      """Wrap Keras operations to numpy."""
      y_true = convert_to_tensor(y_true)
      y_pred = convert_to_tensor(y_pred)
      return KerasRegressor.root_mean_squared_error(y_true, y_pred).numpy()

    score_functions = (
        _r2_score,
        keras_backend_r2,
    )
    correct_scorer = sklearn_r2_score

    for (y_true, y_pred) in datasets:
      for f in score_functions:
        np.testing.assert_almost_equal(
            f(y_true, y_pred), correct_scorer(y_true, y_pred), decimal=5
        )

  def test_scoring_accuracy(self):
    """Test custom accuracy implementation against scikit-learn's."""

    n_samples = 50

    def make_multiclass(y):
      return y.reshape(-1, 1)

    datasets = []
    y_true = np.random.randint(low=0, high=n_samples, size=n_samples)
    y_pred = np.random.randint(low=0, high=n_samples, size=n_samples)
    datasets.append((make_multiclass(y_true), make_multiclass(y_pred)))
    score_functions = (_accuracy_score,)
    correct_scorer = sklearn_accuracy_score

    for (y_true, y_pred) in datasets:
      for f in score_functions:
        np.testing.assert_almost_equal(
            f(y_true, y_pred), correct_scorer(y_true, y_pred), decimal=5
        )

  def test_scoring_accuracy_multilabel(self):
    """Test that accuracy score function works for multilabel ouputs."""
    n_samples = 50
    n_outputs = 5
    n_classes = 2

    def make_multilabel(y):
      return np.repeat(y.reshape(-1, 1), n_outputs, axis=1)

    datasets = []
    y_true = np.random.randint(low=0, high=n_classes, size=n_samples)
    y_pred = np.random.randint(low=0, high=n_classes, size=n_samples)
    datasets.append((make_multilabel(y_true), make_multilabel(y_pred)))
    score_functions = (_accuracy_score,)
    correct_scorer = sklearn_accuracy_score

    for (y_true, y_pred) in datasets:
      for f in score_functions:
        np.testing.assert_almost_equal(
            f(y_true, y_pred), correct_scorer(y_true, y_pred), decimal=5
        )

  def test_scoring_accuracy_multiclass_multioutput(self):
    """Test that accuracy score function works for multioutput ouputs."""
    n_samples = 50
    n_outputs = 5
    n_classes = 10

    def make_multioutput(y):
      return np.repeat(y.reshape(-1, 1), n_outputs, axis=1)

    datasets = []
    y_true = np.random.randint(low=0, high=n_classes, size=n_samples)
    y_pred = np.random.randint(low=0, high=n_classes, size=n_samples)
    datasets.append((make_multioutput(y_true), make_multioutput(y_pred)))
    score_functions = (
        lambda y1, y2: _accuracy_score(
            y1, y2, cls_type="multiclass-multioutput"
        ),
    )
    correct_scorer = lambda y1, y2: np.mean(np.all(y1 == y2, axis=1))

    for (y_true, y_pred) in datasets:
      for f in score_functions:
        np.testing.assert_almost_equal(
            f(y_true, y_pred), correct_scorer(y_true, y_pred), decimal=5
        )


class ScikitLearnAPIWrapperTestMultiInputOutput(test.TestCase):
  """Tests involving multiple inputs / outputs.
  """

  def test_multi_input_and_output(self):
    """Compares to the scikit-learn RandomForestRegressor classifier.
    """
    clf = FunctionAPIMultiInputMultiOutputClassifier()
    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(4,),
        num_classes=3,
    )
    y_train = np.stack([y_train == 1, y_train], axis=1)  # simulate 2 outputs
    y_test = np.stack([y_test == 1, y_test], axis=1)  # simulate 2 outputs
    clf.fit(x_train, y_train)
    clf.predict(x_test)
    clf.score(x_train, y_train)

  def test_multi_label_clasification(self):
    """Compares to the scikit-learn RandomForestRegressor classifier.
    """
    clf_keras = FunctionAPIMultiLabelClassifier()
    clf_sklearn = RandomForestClassifier()
    # taken from https://scikit-learn.org/stable/modules/multiclass.html
    y = np.array([[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]])
    y = MultiLabelBinarizer().fit_transform(y)

    (x_train, _), (_, _) = testing_utils.get_test_data(
        train_samples=y.shape[0], test_samples=0,
        input_shape=(4,), num_classes=3,
    )

    clf_keras.fit(x_train, y)
    y_pred_keras = clf_keras.predict(x_train)
    clf_keras.score(x_train, y)

    clf_sklearn.fit(x_train, y)
    y_pred_sklearn = clf_sklearn.predict(x_train)
    clf_sklearn.score(x_train, y)

    assert y_pred_keras.shape == y_pred_sklearn.shape

  def test_multi_output_regression(self):
    """Compares to the scikit-learn RandomForestRegressor classifier.
    """
    reg_keras = FunctionAPIMultiOutputRegressor()
    reg_sklearn = RandomForestRegressor()
    # taken from https://scikit-learn.org/stable/modules/multiclass.html
    (X, _), (_, _) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=NUM_CLASSES,
    )
    y = np.random.random_sample(size=(TRAIN_SAMPLES, NUM_CLASSES))

    reg_keras.fit(X, y)
    y_pred_keras = reg_keras.predict(X)
    reg_keras.score(X, y)

    reg_sklearn.fit(X, y)
    y_pred_sklearn = reg_sklearn.predict(X)
    reg_sklearn.score(X, y)

    assert y_pred_keras.shape == y_pred_sklearn.shape

if __name__ == "__main__":
  test.main()
