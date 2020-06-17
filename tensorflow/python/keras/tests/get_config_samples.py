# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=protected-access
"""Sample `get_config` results for testing backwards compatibility."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# inputs = tf.keras.Input(10)
# x = tf.keras.layers.Dense(10, activation='relu')(inputs)
# outputs = tf.keras.layers.Dense(1)(x)
# model = tf.keras.Model(inputs, outputs)
FUNCTIONAL_DNN = {
    'input_layers': [['input_1', 0, 0]],
    'layers': [{
        'class_name': 'InputLayer',
        'config': {
            'batch_input_shape': (None, 10),
            'dtype': 'float32',
            'name': 'input_1',
            'ragged': False,
            'sparse': False
        },
        'inbound_nodes': [],
        'name': 'input_1'
    }, {
        'class_name': 'Dense',
        'config': {
            'activation': 'relu',
            'activity_regularizer': None,
            'bias_constraint': None,
            'bias_initializer': {
                'class_name': 'Zeros',
                'config': {}
            },
            'bias_regularizer': None,
            'dtype': 'float32',
            'kernel_constraint': None,
            'kernel_initializer': {
                'class_name': 'GlorotUniform',
                'config': {
                    'seed': None
                }
            },
            'kernel_regularizer': None,
            'name': 'dense',
            'trainable': True,
            'units': 10,
            'use_bias': True
        },
        'inbound_nodes': [[['input_1', 0, 0, {}]]],
        'name': 'dense'
    }, {
        'class_name': 'Dense',
        'config': {
            'activation': 'linear',
            'activity_regularizer': None,
            'bias_constraint': None,
            'bias_initializer': {
                'class_name': 'Zeros',
                'config': {}
            },
            'bias_regularizer': None,
            'dtype': 'float32',
            'kernel_constraint': None,
            'kernel_initializer': {
                'class_name': 'GlorotUniform',
                'config': {
                    'seed': None
                }
            },
            'kernel_regularizer': None,
            'name': 'dense_1',
            'trainable': True,
            'units': 1,
            'use_bias': True
        },
        'inbound_nodes': [[['dense', 0, 0, {}]]],
        'name': 'dense_1'
    }],
    'name': 'model',
    'output_layers': [['dense_1', 0, 0]]
}

# inputs = tf.keras.Input((256, 256, 3))
# x = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3))(inputs)
# x = tf.keras.layers.Flatten()(x)
# outputs = tf.keras.layers.Dense(1)(x)
# model = tf.keras.Model(inputs, outputs)
FUNCTIONAL_CNN = {
    'input_layers': [['input_2', 0, 0]],
    'layers': [{
        'class_name': 'InputLayer',
        'config': {
            'batch_input_shape': (None, 256, 256, 3),
            'dtype': 'float32',
            'name': 'input_2',
            'ragged': False,
            'sparse': False
        },
        'inbound_nodes': [],
        'name': 'input_2'
    }, {
        'class_name': 'Conv2D',
        'config': {
            'activation': 'linear',
            'activity_regularizer': None,
            'bias_constraint': None,
            'bias_initializer': {
                'class_name': 'Zeros',
                'config': {}
            },
            'bias_regularizer': None,
            'data_format': 'channels_last',
            'dilation_rate': (1, 1),
            'dtype': 'float32',
            'filters': 3,
            'kernel_constraint': None,
            'kernel_initializer': {
                'class_name': 'GlorotUniform',
                'config': {
                    'seed': None
                }
            },
            'kernel_regularizer': None,
            'kernel_size': (3, 3),
            'name': 'conv2d',
            'padding': 'valid',
            'strides': (1, 1),
            'trainable': True,
            'use_bias': True
        },
        'inbound_nodes': [[['input_2', 0, 0, {}]]],
        'name': 'conv2d'
    }, {
        'class_name': 'Flatten',
        'config': {
            'data_format': 'channels_last',
            'dtype': 'float32',
            'name': 'flatten',
            'trainable': True
        },
        'inbound_nodes': [[['conv2d', 0, 0, {}]]],
        'name': 'flatten'
    }, {
        'class_name': 'Dense',
        'config': {
            'activation': 'linear',
            'activity_regularizer': None,
            'bias_constraint': None,
            'bias_initializer': {
                'class_name': 'Zeros',
                'config': {}
            },
            'bias_regularizer': None,
            'dtype': 'float32',
            'kernel_constraint': None,
            'kernel_initializer': {
                'class_name': 'GlorotUniform',
                'config': {
                    'seed': None
                }
            },
            'kernel_regularizer': None,
            'name': 'dense_2',
            'trainable': True,
            'units': 1,
            'use_bias': True
        },
        'inbound_nodes': [[['flatten', 0, 0, {}]]],
        'name': 'dense_2'
    }],
    'name': 'model_1',
    'output_layers': [['dense_2', 0, 0]]
}

# inputs = tf.keras.Input((10, 3))
# x = tf.keras.layers.LSTM(10)(inputs)
# outputs = tf.keras.layers.Dense(1)(x)
# model = tf.keras.Model(inputs, outputs)
FUNCTIONAL_LSTM = {
    'input_layers': [['input_5', 0, 0]],
    'layers': [{
        'class_name': 'InputLayer',
        'config': {
            'batch_input_shape': (None, 10, 3),
            'dtype': 'float32',
            'name': 'input_5',
            'ragged': False,
            'sparse': False
        },
        'inbound_nodes': [],
        'name': 'input_5'
    }, {
        'class_name': 'LSTM',
        'config': {
            'activation': 'tanh',
            'activity_regularizer': None,
            'bias_constraint': None,
            'bias_initializer': {
                'class_name': 'Zeros',
                'config': {}
            },
            'bias_regularizer': None,
            'dropout': 0.0,
            'dtype': 'float32',
            'go_backwards': False,
            'implementation': 2,
            'kernel_constraint': None,
            'kernel_initializer': {
                'class_name': 'GlorotUniform',
                'config': {
                    'seed': None
                }
            },
            'kernel_regularizer': None,
            'name': 'lstm_2',
            'recurrent_activation': 'sigmoid',
            'recurrent_constraint': None,
            'recurrent_dropout': 0.0,
            'recurrent_initializer': {
                'class_name': 'Orthogonal',
                'config': {
                    'gain': 1.0,
                    'seed': None
                }
            },
            'recurrent_regularizer': None,
            'return_sequences': False,
            'return_state': False,
            'stateful': False,
            'time_major': False,
            'trainable': True,
            'unit_forget_bias': True,
            'units': 10,
            'unroll': False,
            'use_bias': True
        },
        'inbound_nodes': [[['input_5', 0, 0, {}]]],
        'name': 'lstm_2'
    }, {
        'class_name': 'Dense',
        'config': {
            'activation': 'linear',
            'activity_regularizer': None,
            'bias_constraint': None,
            'bias_initializer': {
                'class_name': 'Zeros',
                'config': {}
            },
            'bias_regularizer': None,
            'dtype': 'float32',
            'kernel_constraint': None,
            'kernel_initializer': {
                'class_name': 'GlorotUniform',
                'config': {
                    'seed': None
                }
            },
            'kernel_regularizer': None,
            'name': 'dense_4',
            'trainable': True,
            'units': 1,
            'use_bias': True
        },
        'inbound_nodes': [[['lstm_2', 0, 0, {}]]],
        'name': 'dense_4'
    }],
    'name': 'model_3',
    'output_layers': [['dense_4', 0, 0]]
}

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(10))
# model.add(tf.keras.layers.Dense(1))
SEQUENTIAL_DNN = {
    'layers': [{
        'class_name': 'Dense',
        'config': {
            'activation': 'linear',
            'activity_regularizer': None,
            'bias_constraint': None,
            'bias_initializer': {
                'class_name': 'Zeros',
                'config': {}
            },
            'bias_regularizer': None,
            'dtype': 'float32',
            'kernel_constraint': None,
            'kernel_initializer': {
                'class_name': 'GlorotUniform',
                'config': {
                    'seed': None
                }
            },
            'kernel_regularizer': None,
            'name': 'dense_2',
            'trainable': True,
            'units': 10,
            'use_bias': True
        }
    }, {
        'class_name': 'Dense',
        'config': {
            'activation': 'linear',
            'activity_regularizer': None,
            'bias_constraint': None,
            'bias_initializer': {
                'class_name': 'Zeros',
                'config': {}
            },
            'bias_regularizer': None,
            'dtype': 'float32',
            'kernel_constraint': None,
            'kernel_initializer': {
                'class_name': 'GlorotUniform',
                'config': {
                    'seed': None
                }
            },
            'kernel_regularizer': None,
            'name': 'dense_3',
            'trainable': True,
            'units': 1,
            'use_bias': True
        }
    }],
    'name': 'sequential_1'
}

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv2D(32, (3, 3)))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(1))
SEQUENTIAL_CNN = {
    'layers': [{
        'class_name': 'Conv2D',
        'config': {
            'activation': 'linear',
            'activity_regularizer': None,
            'bias_constraint': None,
            'bias_initializer': {
                'class_name': 'Zeros',
                'config': {}
            },
            'bias_regularizer': None,
            'data_format': 'channels_last',
            'dilation_rate': (1, 1),
            'dtype': 'float32',
            'filters': 32,
            'kernel_constraint': None,
            'kernel_initializer': {
                'class_name': 'GlorotUniform',
                'config': {
                    'seed': None
                }
            },
            'kernel_regularizer': None,
            'kernel_size': (3, 3),
            'name': 'conv2d_1',
            'padding': 'valid',
            'strides': (1, 1),
            'trainable': True,
            'use_bias': True
        }
    }, {
        'class_name': 'Flatten',
        'config': {
            'data_format': 'channels_last',
            'dtype': 'float32',
            'name': 'flatten_1',
            'trainable': True
        }
    }, {
        'class_name': 'Dense',
        'config': {
            'activation': 'linear',
            'activity_regularizer': None,
            'bias_constraint': None,
            'bias_initializer': {
                'class_name': 'Zeros',
                'config': {}
            },
            'bias_regularizer': None,
            'dtype': 'float32',
            'kernel_constraint': None,
            'kernel_initializer': {
                'class_name': 'GlorotUniform',
                'config': {
                    'seed': None
                }
            },
            'kernel_regularizer': None,
            'name': 'dense_6',
            'trainable': True,
            'units': 1,
            'use_bias': True
        }
    }],
    'name': 'sequential_4'
}

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.LSTM(10))
# model.add(tf.keras.layers.Dense(1))
SEQUENTIAL_LSTM = {
    'layers': [{
        'class_name': 'LSTM',
        'config': {
            'activation': 'tanh',
            'activity_regularizer': None,
            'bias_constraint': None,
            'bias_initializer': {
                'class_name': 'Zeros',
                'config': {}
            },
            'bias_regularizer': None,
            'dropout': 0.0,
            'dtype': 'float32',
            'go_backwards': False,
            'implementation': 2,
            'kernel_constraint': None,
            'kernel_initializer': {
                'class_name': 'GlorotUniform',
                'config': {
                    'seed': None
                }
            },
            'kernel_regularizer': None,
            'name': 'lstm',
            'recurrent_activation': 'sigmoid',
            'recurrent_constraint': None,
            'recurrent_dropout': 0.0,
            'recurrent_initializer': {
                'class_name': 'Orthogonal',
                'config': {
                    'gain': 1.0,
                    'seed': None
                }
            },
            'recurrent_regularizer': None,
            'return_sequences': False,
            'return_state': False,
            'stateful': False,
            'time_major': False,
            'trainable': True,
            'unit_forget_bias': True,
            'units': 10,
            'unroll': False,
            'use_bias': True
        }
    }, {
        'class_name': 'Dense',
        'config': {
            'activation': 'linear',
            'activity_regularizer': None,
            'bias_constraint': None,
            'bias_initializer': {
                'class_name': 'Zeros',
                'config': {}
            },
            'bias_regularizer': None,
            'dtype': 'float32',
            'kernel_constraint': None,
            'kernel_initializer': {
                'class_name': 'GlorotUniform',
                'config': {
                    'seed': None
                }
            },
            'kernel_regularizer': None,
            'name': 'dense_4',
            'trainable': True,
            'units': 1,
            'use_bias': True
        }
    }],
    'name': 'sequential_2'
}
