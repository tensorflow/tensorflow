from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import tempfile
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.saved_model.load import Loader
from tensorflow.python.keras.layers.octaveconvolution import OctaveConv1D,OctaveConvDual

from tensorflow.python.eager import backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


class TestConv1D(keras_parameterized.TestCase):

    @keras_parameterized.run_all_keras_modes
    def test_Test_Conv1D(self):
        if tf_test_util.is_gpu_available():
            self.skipTest('Only test embedding on CPU.')

        testing_utils.layer_test(
        keras.layers.OctaveConv1D,
        kwargs={'filters': 10,
                'kernel_size':3
                },
        input_shape=(3, 2),
        input_dtype='int32',
        expected_output_dtype='float32')

        testing_utils.layer_test(
        keras.layers.OctaveConv1D,
        kwargs={'filters': 10,
                'kernel_size':3,
                'octave':4
                },
        input_shape=(3, 2),
        input_dtype='int32',
        expected_output_dtype='float32')

        testing_utils.layer_test(
        keras.layers.OctaveConv1D,
        kwargs={'filters': 10,
                'kernel_size':3,
                'octave':4,
                'ratio_out':0.0
                },
        input_shape=(3, 2),
        input_dtype='int32',
        expected_output_dtype='float32')

    @keras_parameterized.run_all_keras_modes
    def _test_fit(self, model):
        data_size = 4096
        x = np.random.standard_normal((data_size, 32, 3))
        y = np.random.randint(0, 1, data_size)
        model.fit(x, y, epochs=3)
        model_path = os.path.join(tempfile.gettempdir(), 'test_octave_conv_%f.h5' % np.random.random())
        model.save(model_path)
        model = Loader.load(model_path)
        predicted = model.predict(x).argmax(axis=-1)
        diff = np.sum(np.abs(y - predicted))
        self.assertLess(diff, 100)

    def test_fit_default(self):
        inputs = Input(shape=(32, 3))
        high, low = OctaveConv1D(13, kernel_size=3)(inputs)
        high, low = MaxPool1D()(high), MaxPool1D()(low)
        high, low = OctaveConv1D(7, kernel_size=3)([high, low])
        high, low = MaxPool1D()(high), MaxPool1D()(low)
        conv = OctaveConv1D(5, kernel_size=3, ratio_out=0.0)([high, low])
        flatten = Flatten()(conv)
        outputs = Dense(units=2, activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_fit_octave(self):
        inputs = Input(shape=(32, 3))
        high, low = OctaveConv1D(13, kernel_size=3, octave=4)(inputs)
        high, low = MaxPool1D()(high), MaxPool1D()(low)
        conv = OctaveConv1D(5, kernel_size=3, octave=4, ratio_out=0.0)([high, low])
        flatten = Flatten()(conv)
        outputs = Dense(units=2, activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_fit_lower_output(self):
        inputs = Input(shape=(32, 3))
        high, low = OctaveConv1D(13, kernel_size=3)(inputs)
        high, low = MaxPool1D()(high), MaxPool1D()(low)
        high, low = OctaveConv1D(7, kernel_size=3)([high, low])
        high, low = MaxPool1D()(high), MaxPool1D()(low)
        conv = OctaveConv1D(5, kernel_size=3, ratio_out=1.0)([high, low])
        flatten = Flatten()(conv)
        outputs = Dense(units=2, activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_raise_dimension_specified(self):
        with self.assertRaises(ValueError):
            inputs = Input(shape=(32, None))
            outputs = OctaveConv1D(13, kernel_size=3, ratio_out=0.0)(inputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        with self.assertRaises(ValueError):
            inputs_high = Input(shape=(32, 3))
            inputs_low = Input(shape=(32, None))
            outputs = OctaveConv1D(13, kernel_size=3, ratio_out=0.0)([inputs_high, inputs_low])
            model = Model(inputs=[inputs_high, inputs_low], outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    def test_raise_octave_divisible(self):
        with self.assertRaises(ValueError):
            inputs = Input(shape=(32, 3))
            outputs = OctaveConv1D(13, kernel_size=3, octave=5, ratio_out=0.0)(inputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    def test_make_dual_layer(self):
        inputs = Input(shape=(32, 3))
        conv = OctaveConv1D(13, kernel_size=3)(inputs)
        pool = OctaveConvDual()(conv, MaxPool1D())
        conv = OctaveConv1D(7, kernel_size=3)(pool)
        pool = OctaveConvDual()(conv, MaxPool1D())
        conv = OctaveConv1D(5, kernel_size=3, ratio_out=0.0)(pool)
        flatten = OctaveConvDual()(conv, Flatten())
        outputs = Dense(units=2, activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

class TestConv2D(keras_parameterized.TestCase):

    @keras_parameterized.run_all_keras_modes
    def test_Test_Conv1D(self):
        if tf_test_util.is_gpu_available():
            self.skipTest('Only test embedding on CPU.')

        testing_utils.layer_test(
        keras.layers.OctaveConv1D,
        kwargs={'filters': 10,
                'kernel_size':3
                },
        input_shape=(32,32, 3),
        input_dtype='int32',
        expected_output_dtype='float32')

        testing_utils.layer_test(
        keras.layers.OctaveConv1D,
        kwargs={'filters': 10,
                'kernel_size':3,
                'octave':4
                },
        input_shape=(32,32, 3),
        input_dtype='int32',
        expected_output_dtype='float32')

        testing_utils.layer_test(
        keras.layers.OctaveConv1D,
        kwargs={'filters': 10,
                'kernel_size':3,
                'octave':4,
                'ratio_out':0.0
                },
        input_shape=(32,32, 3),
        input_dtype='int32',
        expected_output_dtype='float32')

    @keras_parameterized.run_all_keras_modes
    def _test_fit(self, model, data_format='channels_last'):
        data_size = 4096
        if data_format == 'channels_last':
            x = np.random.standard_normal((data_size, 32, 32, 3))
        else:
            x = np.random.standard_normal((data_size, 3, 32, 32))
        y = np.random.randint(0, 1, data_size)
        model.fit(x, y, epochs=3)
        model_path = os.path.join(tempfile.gettempdir(), 'test_octave_conv_%f.h5' % np.random.random())
        model.save(model_path)
        model = load_model(model_path, custom_objects={'OctaveConv2D': OctaveConv2D})
        predicted = model.predict(x).argmax(axis=-1)
        diff = np.sum(np.abs(y - predicted))
        self.assertLess(diff, 100)

        def test_fit_default(self):
        inputs = Input(shape=(32, 32, 3))
        high, low = OctaveConv2D(13, kernel_size=3)(inputs)
        high, low = MaxPool2D()(high), MaxPool2D()(low)
        high, low = OctaveConv2D(7, kernel_size=3)([high, low])
        high, low = MaxPool2D()(high), MaxPool2D()(low)
        conv = OctaveConv2D(5, kernel_size=3, ratio_out=0.0)([high, low])
        flatten = Flatten()(conv)
        outputs = Dense(units=2, activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_fit_channels_first(self):
        inputs = Input(shape=(3, 32, 32))
        high, low = OctaveConv2D(13, kernel_size=3, data_format='channels_first')(inputs)
        high, low = MaxPool2D(data_format='channels_first')(high), MaxPool2D(data_format='channels_first')(low)
        high, low = OctaveConv2D(7, kernel_size=3, data_format='channels_first')([high, low])
        high, low = MaxPool2D(data_format='channels_first')(high), MaxPool2D(data_format='channels_first')(low)
        conv = OctaveConv2D(5, kernel_size=3, ratio_out=0.0, data_format='channels_first')([high, low])
        flatten = Flatten()(conv)
        outputs = Dense(units=2, activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model, data_format='channels_first')

    def test_fit_octave(self):
        inputs = Input(shape=(32, 32, 3))
        high, low = OctaveConv2D(13, kernel_size=3, octave=4)(inputs)
        high, low = MaxPool2D()(high), MaxPool2D()(low)
        conv = OctaveConv2D(5, kernel_size=3, octave=4, ratio_out=0.0)([high, low])
        flatten = Flatten()(conv)
        outputs = Dense(units=2, activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_fit_lower_output(self):
        inputs = Input(shape=(32, 32, 3))
        high, low = OctaveConv2D(13, kernel_size=3)(inputs)
        high, low = MaxPool2D()(high), MaxPool2D()(low)
        high, low = OctaveConv2D(7, kernel_size=3)([high, low])
        high, low = MaxPool2D()(high), MaxPool2D()(low)
        conv = OctaveConv2D(5, kernel_size=3, ratio_out=1.0)([high, low])
        flatten = Flatten()(conv)
        outputs = Dense(units=2, activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_raise_dimension_specified(self):
        with self.assertRaises(ValueError):
            inputs = Input(shape=(32, 32, None))
            outputs = OctaveConv2D(13, kernel_size=3, ratio_out=0.0)(inputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        with self.assertRaises(ValueError):
            inputs_high = Input(shape=(32, 32, 3))
            inputs_low = Input(shape=(32, 32, None))
            outputs = OctaveConv2D(13, kernel_size=3, ratio_out=0.0)([inputs_high, inputs_low])
            model = Model(inputs=[inputs_high, inputs_low], outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    def test_raise_octave_divisible(self):
        with self.assertRaises(ValueError):
            inputs = Input(shape=(32, 32, 3))
            outputs = OctaveConv2D(13, kernel_size=3, octave=5, ratio_out=0.0)(inputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    def test_make_dual_lambda(self):
        inputs = Input(shape=(32, 32, 3))
        conv = OctaveConv2D(13, kernel_size=3)(inputs)
        pool = OctaveConvDual()(conv, lambda: MaxPool2D())
        conv = OctaveConv2D(7, kernel_size=3)(pool)
        pool = OctaveConvDual()(conv, lambda: MaxPool2D())
        conv = OctaveConv2D(5, kernel_size=3, ratio_out=0.0)(pool)
        flatten = Flatten()(conv)
        outputs = Dense(units=2, activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)
