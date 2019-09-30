# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Scikit-learn API wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
from scipy.stats import randint
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_boston, load_digits, load_iris
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              BaggingClassifier, BaggingRegressor)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tensorflow.python import keras
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test

from keras import backend as K
from keras.layers import Activation, Concatenate, Conv2D, Dense, Flatten, Input
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical

from keras.wrappers.scikit_learn import (BaseWrapper, KerasClassifier,
                                         KerasRegressor)

INPUT_DIM = 5
HIDDEN_DIM = 5
TRAIN_SAMPLES = 10
TEST_SAMPLES = 5
NUM_CLASSES = 2
BATCH_SIZE = 5
EPOCHS = 1


def build_fn_clf(input_shape, output_shape, hidden_dim):
  model = keras.models.Sequential()
  model.add(keras.layers.Activation('relu'))
  model.add(keras.layers.Dense(hidden_dim))
  model.add(keras.layers.Activation('relu'))
  model.add(keras.layers.Dense(NUM_CLASSES))
  model.add(keras.layers.Activation('softmax'))
  model.compile(
      optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
  return model


def assert_classification_works(clf):
  np.random.seed(42)
  (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
      train_samples=TRAIN_SAMPLES,
      test_samples=TEST_SAMPLES,
      input_shape=(INPUT_DIM,),
      num_classes=NUM_CLASSES)

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


def build_fn_reg(input_shape, output_shape, hidden_dim):
  model = keras.models.Sequential()
  model.add(keras.layers.Activation('relu'))
  model.add(keras.layers.Dense(hidden_dim))
  model.add(keras.layers.Activation('relu'))
  model.add(keras.layers.Dense(1))
  model.add(keras.layers.Activation('linear'))
  model.compile(
      optimizer='sgd', loss='mean_absolute_error', metrics=['accuracy'])
  return model


def assert_regression_works(reg):
  np.random.seed(42)
  (x_train, y_train), (x_test, _) = testing_utils.get_test_data(
      train_samples=TRAIN_SAMPLES,
      test_samples=TEST_SAMPLES,
      input_shape=(INPUT_DIM,),
      num_classes=NUM_CLASSES)

  reg.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

  score = reg.score(x_train, y_train, batch_size=BATCH_SIZE)
  assert np.isscalar(score) and np.isfinite(score)

  preds = reg.predict(x_test, batch_size=BATCH_SIZE)
  assert preds.shape == (TEST_SAMPLES,)


class ScikitLearnAPIWrapperTest(test.TestCase):

  def test_classify_build_fn(self):
    with self.cached_session():
      clf = KerasClassifier(
          build_fn=build_fn_clf,
          hidden_dim=HIDDEN_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS)

      assert_classification_works(clf)

  def test_classify_class_build_fn(self):

    class ClassBuildFnClf(object):

      def __call__(self, input_shape, output_shape, hidden_dim):
        return build_fn_clf(input_shape, output_shape, hidden_dim)

    with self.cached_session():
      clf = KerasClassifier(
          build_fn=ClassBuildFnClf(),
          hidden_dim=HIDDEN_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS)

      assert_classification_works(clf)

  def test_classify_inherit_class_build_fn(self):

    class InheritClassBuildFnClf(KerasClassifier):

      def __call__(self, input_shape, output_shape, hidden_dim):
        return build_fn_clf(input_shape, output_shape, hidden_dim)

    with self.cached_session():
      clf = InheritClassBuildFnClf(
          build_fn=None,
          hidden_dim=HIDDEN_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS)

      assert_classification_works(clf)

  def test_regression_build_fn(self):
    with self.cached_session():
      reg = KerasRegressor(
          build_fn=build_fn_reg,
          hidden_dim=HIDDEN_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS)

      assert_regression_works(reg)

  def test_regression_class_build_fn(self):

    class ClassBuildFnReg(object):

      def __call__(self, input_shape, output_shape, hidden_dim):
        return build_fn_reg(input_shape, output_shape, hidden_dim)

    with self.cached_session():
      reg = KerasRegressor(
          build_fn=ClassBuildFnReg(),
          hidden_dim=HIDDEN_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS)

      assert_regression_works(reg)

  def test_regression_inherit_class_build_fn(self):

    class InheritClassBuildFnReg(KerasRegressor):

      def __call__(self, input_shape, output_shape, hidden_dim):
        return build_fn_reg(input_shape, output_shape, hidden_dim)

    with self.cached_session():
      reg = InheritClassBuildFnReg(
          build_fn=None,
          hidden_dim=HIDDEN_DIM,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS)

      assert_regression_works(reg)


# Usage of sklearn's Pipelines, SearchCVs, Ensembles and CalibratedClassifierCVs


def load_digits8x8():
    data = load_digits()
    data.data = data.data.reshape([data.data.shape[0], 1, 8, 8]) / 16.0
    K.set_image_data_format('channels_first')
    return data


def check(estimator, loader):
    data = loader()
    estimator.fit(data.data, data.target)
    preds = estimator.predict(data.data)
    score = estimator.score(data.data, data.target)
    serialized_estimator = pickle.dumps(estimator)
    deserialized_estimator = pickle.loads(serialized_estimator)
    preds = deserialized_estimator.predict(data.data)
    score = deserialized_estimator.score(data.data, data.target)
    assert True


def build_fn_regs(input_shape, output_shape, hidden_layer_sizes=[]):
    model = Sequential()
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation='relu'))
    model.add(Dense(np.prod(output_shape, dtype=np.uint8)))
    model.compile('adam', loss='mean_squared_error')
    return model


def build_fn_clss(input_shape, output_shape, hidden_layer_sizes=[]):
    model = Sequential()
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation='relu'))
    model.add(Dense(np.prod(output_shape), activation='softmax'))
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_fn_clscs(input_shape, output_shape, hidden_layer_sizes=[]):
    model = Sequential()
    model.add(Conv2D(3, (3, 3)))
    model.add(Flatten())
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation='relu'))
    model.add(Dense(np.prod(output_shape), activation='softmax'))
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_fn_clscf(input_shape, output_shape, hidden_layer_sizes=[]):
    x = Input(shape=input_shape)
    z = Conv2D(3, (3, 3))(x)
    z = Flatten()(z)
    for size in hidden_layer_sizes:
        z = Dense(size, activation='relu')(z)
    y = Dense(np.prod(output_shape), activation='softmax')(z)
    model = Model(inputs=x, outputs=y)
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_fn_multi(input_shape, output_shape, hidden_layer_sizes=[]):
    features = Input(shape=input_shape['features'], name='features')
    class_in = Input(shape=input_shape['class_in'], name='class_in')
    z = Concatenate()([features, class_in])
    for size in hidden_layer_sizes:
        z = Dense(size, activation='relu')(z)
    onehot = Dense(np.prod(output_shape['onehot']), activation='softmax',
                   name='onehot')(z)
    class_out = Dense(np.prod(output_shape['class_out']), name='class_out')(z)
    model = Model(inputs=[features, class_in], outputs=[onehot, class_out])
    model.compile('adam',
                  loss={'onehot': 'categorical_crossentropy',
                        'class_out': 'mse'},
                  metrics={'onehot': 'accuracy'})
    return model


CONFIG = {'MLPRegressor': (load_boston, KerasRegressor, build_fn_regs,
                           (BaggingRegressor, AdaBoostRegressor)),
          'MLPClassifier': (load_iris, KerasClassifier, build_fn_clss,
                            (BaggingClassifier, AdaBoostClassifier)),
          'CNNClassifier': (load_digits8x8, KerasClassifier, build_fn_clscs,
                            (BaggingClassifier, AdaBoostClassifier)),
          'CNNClassifierF': (load_digits8x8, KerasClassifier, build_fn_clscf,
                             (BaggingClassifier, AdaBoostClassifier))}


def test_standalone():
    """Tests standalone estimator."""
    for config in ['MLPRegressor', 'MLPClassifier', 'CNNClassifier',
                   'CNNClassifierF']:
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, epochs=1)
        check(estimator, loader)


def test_pipeline():
    """Tests compatibility with Scikit-learn's pipeline."""
    for config in ['MLPRegressor', 'MLPClassifier']:
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, epochs=1)
        estimator = Pipeline([('s', StandardScaler()), ('e', estimator)])
        check(estimator, loader)


def test_searchcv():
    """Tests compatibility with Scikit-learn's hyperparameter search CV."""
    for config in ['MLPRegressor', 'MLPClassifier', 'CNNClassifier',
                   'CNNClassifierF']:
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, epochs=1, validation_split=0.1)
        check(GridSearchCV(estimator, {'hidden_layer_sizes': [[], [5]]}),
              loader)
        check(RandomizedSearchCV(estimator, {'epochs': randint(1, 5)},
                                 n_iter=2), loader)


def test_ensemble():
    """Tests compatibility with Scikit-learn's ensembles."""
    for config in ['MLPRegressor', 'MLPClassifier']:
        loader, model, build_fn, ensembles = CONFIG[config]
        base_estimator = model(build_fn, epochs=1)
        for ensemble in ensembles:
            estimator = ensemble(base_estimator=base_estimator, n_estimators=2)
            check(estimator, loader)


def test_calibratedclassifiercv():
    """Tests compatibility with Scikit-learn's calibrated classifier CV."""
    for config in ['MLPClassifier']:
        loader, _, build_fn, _ = CONFIG[config]
        base_estimator = KerasClassifier(build_fn, epochs=1)
        estimator = CalibratedClassifierCV(base_estimator=base_estimator)
        check(estimator, loader)


def test_transformedtargetregressor():
    """Tests compatibility with Scikit-learn's transformed target regressor."""
    for config in ['MLPRegressor']:
        loader, _, build_fn, _ = CONFIG[config]
        base_estimator = KerasCRegressor(build_fn, epochs=1)
        estimator = TransformedTargetRegressor(regressor=base_estimator,
                                               transformer=StandardScaler())
        check(estimator, loader)


def test_standalone_multi():
    """Tests standalone estimator with multiple inputs and outputs."""
    estimator = BaseWrapper(build_fn_multi, epochs=1)
    data = load_iris()
    features = data.data
    klass = data.target.reshape((-1, 1)).astype(np.float32)
    onehot = to_categorical(data.target)
    estimator.fit({'features': features, 'class_in': klass},
                  {'onehot': onehot, 'class_out': klass})
    preds = estimator.predict({'features': features, 'class_in': klass})
    score = estimator.score({'features': features, 'class_in': klass},
                            {'onehot': onehot, 'class_out': klass})
    serialized_estimator = pickle.dumps(estimator)
    deserialized_estimator = pickle.loads(serialized_estimator)
    preds = deserialized_estimator.predict({'features': features,
                                            'class_in': klass})
    score = deserialized_estimator.score({'features': features,
                                          'class_in': klass},
                                         {'onehot': onehot, 'class_out': klass})


if __name__ == '__main__':
  test.main()
