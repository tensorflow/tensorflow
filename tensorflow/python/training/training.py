# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Support for training models.

See the [Training](https://tensorflow.org/api_guides/python/train) guide.
"""

# Optimizers.
# pylint: disable=g-bad-import-order,unused-import
from tensorflow.python.ops.sdca_ops import sdca_optimizer
from tensorflow.python.ops.sdca_ops import sdca_fprint
from tensorflow.python.ops.sdca_ops import sdca_shrink_l1
from tensorflow.python.training.adadelta import AdadeltaOptimizer
from tensorflow.python.training.adagrad import AdagradOptimizer
from tensorflow.python.training.adagrad_da import AdagradDAOptimizer
from tensorflow.python.training.proximal_adagrad import ProximalAdagradOptimizer
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.experimental.loss_scale_optimizer import MixedPrecisionLossScaleOptimizer
from tensorflow.python.training.experimental.mixed_precision import enable_mixed_precision_graph_rewrite_v1
from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.training.moving_averages import ExponentialMovingAverage
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.python.training.proximal_gradient_descent import ProximalGradientDescentOptimizer
from tensorflow.python.training.sync_replicas_optimizer import SyncReplicasOptimizer

# Utility classes for training.
from tensorflow.python.training.coordinator import Coordinator
from tensorflow.python.training.coordinator import LooperThread
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.training.queue_runner import *

# For the module level doc.
from tensorflow.python.training import input as _input
from tensorflow.python.training.input import *  # pylint: disable=redefined-builtin
# pylint: enable=wildcard-import

from tensorflow.python.training.basic_session_run_hooks import get_or_create_steps_per_run_variable
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training.basic_session_run_hooks import LoggingTensorHook
from tensorflow.python.training.basic_session_run_hooks import StopAtStepHook
from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverHook
from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverListener
from tensorflow.python.training.basic_session_run_hooks import StepCounterHook
from tensorflow.python.training.basic_session_run_hooks import NanLossDuringTrainingError
from tensorflow.python.training.basic_session_run_hooks import NanTensorHook
from tensorflow.python.training.basic_session_run_hooks import SummarySaverHook
from tensorflow.python.training.basic_session_run_hooks import GlobalStepWaiterHook
from tensorflow.python.training.basic_session_run_hooks import FinalOpsHook
from tensorflow.python.training.basic_session_run_hooks import FeedFnHook
from tensorflow.python.training.basic_session_run_hooks import ProfilerHook
from tensorflow.python.training.basic_loops import basic_train_loop
from tensorflow.python.trackable.python_state import PythonState
from tensorflow.python.checkpoint.checkpoint import Checkpoint
from tensorflow.python.training.checkpoint_utils import init_from_checkpoint
from tensorflow.python.training.checkpoint_utils import list_variables
from tensorflow.python.training.checkpoint_utils import load_checkpoint
from tensorflow.python.training.checkpoint_utils import load_variable

from tensorflow.python.training.device_setter import replica_device_setter
from tensorflow.python.training.monitored_session import Scaffold
from tensorflow.python.training.monitored_session import MonitoredTrainingSession
from tensorflow.python.training.monitored_session import SessionCreator
from tensorflow.python.training.monitored_session import ChiefSessionCreator
from tensorflow.python.training.monitored_session import WorkerSessionCreator
from tensorflow.python.training.monitored_session import MonitoredSession
from tensorflow.python.training.monitored_session import SingularMonitoredSession
from tensorflow.python.training.saver import Saver
from tensorflow.python.checkpoint.checkpoint_management import checkpoint_exists
from tensorflow.python.checkpoint.checkpoint_management import generate_checkpoint_state_proto
from tensorflow.python.checkpoint.checkpoint_management import get_checkpoint_mtimes
from tensorflow.python.checkpoint.checkpoint_management import get_checkpoint_state
from tensorflow.python.checkpoint.checkpoint_management import latest_checkpoint
from tensorflow.python.checkpoint.checkpoint_management import update_checkpoint_state
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.training.saver import import_meta_graph
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.session_run_hook import SessionRunContext
from tensorflow.python.training.session_run_hook import SessionRunValues
from tensorflow.python.training.session_manager import SessionManager
from tensorflow.python.training.summary_io import summary_iterator
from tensorflow.python.training.supervisor import Supervisor
from tensorflow.python.training.training_util import write_graph
from tensorflow.python.training.training_util import global_step
from tensorflow.python.training.training_util import get_global_step
from tensorflow.python.training.training_util import assert_global_step
from tensorflow.python.training.training_util import create_global_step
from tensorflow.python.training.training_util import get_or_create_global_step
from tensorflow.python.training.warm_starting_util import VocabInfo
from tensorflow.python.training.warm_starting_util import warm_start
from tensorflow.python.training.py_checkpoint_reader import NewCheckpointReader
from tensorflow.python.util.tf_export import tf_export

# pylint: disable=wildcard-import
# Training data protos.
from tensorflow.core.example.example_pb2 import *
from tensorflow.core.example.feature_pb2 import *
from tensorflow.core.protobuf.saver_pb2 import *

# Utility op.  Open Source. TODO(touts): move to nn?
from tensorflow.python.training.learning_rate_decay import *
# pylint: enable=wildcard-import

# Distributed computing support.
from tensorflow.core.protobuf.cluster_pb2 import ClusterDef
from tensorflow.core.protobuf.cluster_pb2 import JobDef
from tensorflow.core.protobuf.tensorflow_server_pb2 import ServerDef
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.training.server_lib import Server

# pylint: disable=undefined-variable
tf_export("train.BytesList")(BytesList)
tf_export("train.ClusterDef")(ClusterDef)
tf_export("train.Example")(Example)
tf_export("train.Feature")(Feature)
tf_export("train.Features")(Features)
tf_export("train.FeatureList")(FeatureList)
tf_export("train.FeatureLists")(FeatureLists)
tf_export("train.FloatList")(FloatList)
tf_export("train.Int64List")(Int64List)
tf_export("train.JobDef")(JobDef)
tf_export(v1=["train.SaverDef"])(SaverDef)
tf_export("train.SequenceExample")(SequenceExample)
tf_export("train.ServerDef")(ServerDef)

BytesList.__doc__ = """\
Used in `tf.train.Example` protos. Holds a list of byte-strings.

An `Example` proto is a representation of the following python type:

```
Dict[str,
     Union[List[bytes],
           List[int64],
           List[float]]]
```

This proto implements the `List[bytes]` portion.

>>> from google.protobuf import text_format
>>> example = text_format.Parse('''
...   features {
...     feature {key: "my_feature"
...              value {bytes_list {value: ['abc', '12345' ]}}}
...   }''',
...   tf.train.Example())
>>>
>>> example.features.feature['my_feature'].bytes_list.value
["abc", "12345"]

Use `tf.io.parse_example` to extract tensors from a serialized `Example` proto:

>>> tf.io.parse_example(
...     example.SerializeToString(),
...     features = {'my_feature': tf.io.RaggedFeature(dtype=tf.string)})
{'my_feature': <tf.Tensor: shape=(2,), dtype=string,
                           numpy=array([b'abc', b'12345'], dtype=object)>}


See the [`tf.train.Example`](https://www.tensorflow.org/tutorials/load_data/tfrecord#tftrainexample)
guide for usage details.
"""

FloatList.__doc__ = """\
Used in `tf.train.Example` protos. Holds a list of floats.

An `Example` proto is a representation of the following python type:

```
Dict[str,
     Union[List[bytes],
           List[int64],
           List[float]]]
```

This proto implements the `List[float]` portion.

>>> from google.protobuf import text_format
>>> example = text_format.Parse('''
...   features {
...     feature {key: "my_feature"
...              value {float_list {value: [1., 2., 3., 4. ]}}}
...   }''',
...   tf.train.Example())
>>>
>>> example.features.feature['my_feature'].float_list.value
[1.0, 2.0, 3.0, 4.0]

Use `tf.io.parse_example` to extract tensors from a serialized `Example` proto:

>>> tf.io.parse_example(
...     example.SerializeToString(),
...     features = {'my_feature': tf.io.RaggedFeature(dtype=tf.float32)})
{'my_feature': <tf.Tensor: shape=(4,), dtype=float32,
                           numpy=array([1., 2., 3., 4.], dtype=float32)>}

See the [`tf.train.Example`](https://www.tensorflow.org/tutorials/load_data/tfrecord#tftrainexample)
guide for usage details.
"""

Int64List.__doc__ = """\
Used in `tf.train.Example` protos. Holds a list of Int64s.

An `Example` proto is a representation of the following python type:

```
Dict[str,
     Union[List[bytes],
           List[int64],
           List[float]]]
```

This proto implements the `List[int64]` portion.

>>> from google.protobuf import text_format
>>> example = text_format.Parse('''
...   features {
...     feature {key: "my_feature"
...              value {int64_list {value: [1, 2, 3, 4]}}}
...   }''',
...   tf.train.Example())
>>>
>>> example.features.feature['my_feature'].int64_list.value
[1, 2, 3, 4]

Use `tf.io.parse_example` to extract tensors from a serialized `Example` proto:

>>> tf.io.parse_example(
...     example.SerializeToString(),
...     features = {'my_feature': tf.io.RaggedFeature(dtype=tf.int64)})
{'my_feature': <tf.Tensor: shape=(4,), dtype=float32,
                           numpy=array([1, 2, 3, 4], dtype=int64)>}

See the [`tf.train.Example`](https://www.tensorflow.org/tutorials/load_data/tfrecord#tftrainexample)
guide for usage details.
"""

Feature.__doc__ = """\
Used in `tf.train.Example` protos. Contains a list of values.

An `Example` proto is a representation of the following python type:

```
Dict[str,
     Union[List[bytes],
           List[int64],
           List[float]]]
```

This proto implements the `Union`.

The contained list can be one of three types:

  - `tf.train.BytesList`
  - `tf.train.FloatList`
  - `tf.train.Int64List`

>>> int_feature = tf.train.Feature(
...     int64_list=tf.train.Int64List(value=[1, 2, 3, 4]))
>>> float_feature = tf.train.Feature(
...     float_list=tf.train.FloatList(value=[1., 2., 3., 4.]))
>>> bytes_feature = tf.train.Feature(
...     bytes_list=tf.train.BytesList(value=[b"abc", b"1234"]))
>>>
>>> example = tf.train.Example(
...     features=tf.train.Features(feature={
...         'my_ints': int_feature,
...         'my_floats': float_feature,
...         'my_bytes': bytes_feature,
...     }))

Use `tf.io.parse_example` to extract tensors from a serialized `Example` proto:

>>> tf.io.parse_example(
...     example.SerializeToString(),
...     features = {
...         'my_ints': tf.io.RaggedFeature(dtype=tf.int64),
...         'my_floats': tf.io.RaggedFeature(dtype=tf.float32),
...         'my_bytes': tf.io.RaggedFeature(dtype=tf.string)})
{'my_bytes': <tf.Tensor: shape=(2,), dtype=string,
                         numpy=array([b'abc', b'1234'], dtype=object)>,
 'my_floats': <tf.Tensor: shape=(4,), dtype=float32,
                          numpy=array([1., 2., 3., 4.], dtype=float32)>,
 'my_ints': <tf.Tensor: shape=(4,), dtype=int64,
                        numpy=array([1, 2, 3, 4])>}

"""

Features.__doc__ = """\
Used in `tf.train.Example` protos. Contains the mapping from keys to `Feature`.

An `Example` proto is a representation of the following python type:

```
Dict[str,
     Union[List[bytes],
           List[int64],
           List[float]]]
```

This proto implements the `Dict`.

>>> int_feature = tf.train.Feature(
...     int64_list=tf.train.Int64List(value=[1, 2, 3, 4]))
>>> float_feature = tf.train.Feature(
...     float_list=tf.train.FloatList(value=[1., 2., 3., 4.]))
>>> bytes_feature = tf.train.Feature(
...     bytes_list=tf.train.BytesList(value=[b"abc", b"1234"]))
>>>
>>> example = tf.train.Example(
...     features=tf.train.Features(feature={
...         'my_ints': int_feature,
...         'my_floats': float_feature,
...         'my_bytes': bytes_feature,
...     }))

Use `tf.io.parse_example` to extract tensors from a serialized `Example` proto:

>>> tf.io.parse_example(
...     example.SerializeToString(),
...     features = {
...         'my_ints': tf.io.RaggedFeature(dtype=tf.int64),
...         'my_floats': tf.io.RaggedFeature(dtype=tf.float32),
...         'my_bytes': tf.io.RaggedFeature(dtype=tf.string)})
{'my_bytes': <tf.Tensor: shape=(2,), dtype=string,
                         numpy=array([b'abc', b'1234'], dtype=object)>,
 'my_floats': <tf.Tensor: shape=(4,), dtype=float32,
                          numpy=array([1., 2., 3., 4.], dtype=float32)>,
 'my_ints': <tf.Tensor: shape=(4,), dtype=int64,
                        numpy=array([1, 2, 3, 4])>}

"""

FeatureList.__doc__ = """\
Mainly used as part of a `tf.train.SequenceExample`.

Contains a list of `tf.train.Feature`s.

The `tf.train.SequenceExample` proto can be thought of as a
proto implementation of the following python type:

```
# tf.train.Feature
Feature = Union[List[bytes],
                List[int64],
                List[float]]

# tf.train.FeatureList
FeatureList = List[Feature]

# tf.train.FeatureLists
FeatureLists = Dict[str, FeatureList]

class SequenceExample(typing.NamedTuple):
  context: Dict[str, Feature]
  feature_lists: FeatureLists
```

This proto implements the `List[Feature]` portion.

"""

FeatureLists.__doc__ = """\
Mainly used as part of a `tf.train.SequenceExample`.

Contains a list of `tf.train.Feature`s.

The `tf.train.SequenceExample` proto can be thought of as a
proto implementation of the following python type:

```
# tf.train.Feature
Feature = Union[List[bytes],
                List[int64],
                List[float]]

# tf.train.FeatureList
FeatureList = List[Feature]

# tf.train.FeatureLists
FeatureLists = Dict[str, FeatureList]

class SequenceExample(typing.NamedTuple):
  context: Dict[str, Feature]
  feature_lists: FeatureLists
```

This proto implements the `Dict[str, FeatureList]` portion.
"""


Example.__doc__ = """\
An `Example` is a standard proto storing data for training and inference.

An `Example` proto is a representation of the following python type:

```
Dict[str,
     Union[List[bytes],
           List[int64],
           List[float]]]
```

It contains a key-value store `Example.features` where each key (string) maps
to a `tf.train.Feature` message which contains a fixed-type list. This flexible
and compact format allows the storage of large amounts of typed data, but
requires that the data shape and use be determined by the configuration files
and parsers that are used to read and write this format (refer to
`tf.io.parse_example` for details).

>>> from google.protobuf import text_format
>>> example = text_format.Parse('''
...   features {
...     feature {key: "my_feature"
...              value {int64_list {value: [1, 2, 3, 4]}}}
...   }''',
...   tf.train.Example())

Use `tf.io.parse_example` to extract tensors from a serialized `Example` proto:

>>> tf.io.parse_example(
...     example.SerializeToString(),
...     features = {'my_feature': tf.io.RaggedFeature(dtype=tf.int64)})
{'my_feature': <tf.Tensor: shape=(4,), dtype=float32,
                           numpy=array([1, 2, 3, 4], dtype=int64)>}

While the list of keys, and the contents of each key _could_ be different for
every `Example`, TensorFlow expects a fixed list of keys, each with a fixed
`tf.dtype`. A conformant `Example` dataset obeys the following conventions:

  - If a Feature `K` exists in one example with data type `T`, it must be of
      type `T` in all other examples when present. It may be omitted.
  - The number of instances of Feature `K` list data may vary across examples,
      depending on the requirements of the model.
  - If a Feature `K` doesn't exist in an example, a `K`-specific default will be
      used, if configured.
  - If a Feature `K` exists in an example but contains no items, the intent
      is considered to be an empty tensor and no default will be used.

"""

SequenceExample.__doc__ = """\
A `SequenceExample` represents a sequence of features and some context.

It can be thought of as a proto-implementation of the following python type:

```
Feature = Union[List[bytes],
                List[int64],
                List[float]]

class SequenceExample(typing.NamedTuple):
  context: Dict[str, Feature]
  feature_lists: Dict[str, List[Feature]]
```

To implement this as protos it's broken up into sub-messages as follows:

```
# tf.train.Feature
Feature = Union[List[bytes],
                List[int64],
                List[float]]

# tf.train.FeatureList
FeatureList = List[Feature]

# tf.train.FeatureLists
FeatureLists = Dict[str, FeatureList]

# tf.train.SequenceExample
class SequenceExample(typing.NamedTuple):
  context: Dict[str, Feature]
  feature_lists: FeatureLists
```

To parse a `SequenceExample` in TensorFlow refer to the
`tf.io.parse_sequence_example` function.

The `context` contains features which apply to the entire
example. The `feature_lists` contain a key, value map where each key is
associated with a repeated set of `tf.train.Features` (a `tf.train.FeatureList`).
A `FeatureList` represents the values of a feature identified by its key
over time / frames.

Below is a `SequenceExample` for a movie recommendation application recording a
sequence of ratings by a user. The time-independent features ("locale",
"age", "favorites") describing the user are part of the context. The sequence
of movies the user rated are part of the feature_lists. For each movie in the
sequence we have information on its name and actors and the user's rating.
This information is recorded in three separate `feature_list`s.
In the example below there are only two movies. All three `feature_list`s,
namely "movie_ratings", "movie_names", and "actors" have a feature value for
both movies. Note, that "actors" is itself a `bytes_list` with multiple
strings per movie.

```
  context: {
    feature: {
      key  : "locale"
      value: {
        bytes_list: {
          value: [ "pt_BR" ]
        }
      }
    }
    feature: {
      key  : "age"
      value: {
        float_list: {
          value: [ 19.0 ]
        }
      }
    }
    feature: {
      key  : "favorites"
      value: {
        bytes_list: {
          value: [ "Majesty Rose", "Savannah Outen", "One Direction" ]
        }
      }
    }
  }
  feature_lists: {
    feature_list: {
      key  : "movie_ratings"
      value: {
        feature: {
          float_list: {
            value: [ 4.5 ]
          }
        }
        feature: {
          float_list: {
            value: [ 5.0 ]
          }
        }
      }
    }
    feature_list: {
      key  : "movie_names"
      value: {
        feature: {
          bytes_list: {
            value: [ "The Shawshank Redemption" ]
          }
        }
        feature: {
          bytes_list: {
            value: [ "Fight Club" ]
          }
        }
      }
    }
    feature_list: {
      key  : "actors"
      value: {
        feature: {
          bytes_list: {
            value: [ "Tim Robbins", "Morgan Freeman" ]
          }
        }
        feature: {
          bytes_list: {
            value: [ "Brad Pitt", "Edward Norton", "Helena Bonham Carter" ]
          }
        }
      }
    }
  }
```

A conformant `SequenceExample` data set obeys the following conventions:

`context`:

  - All conformant context features `K` must obey the same conventions as
    a conformant Example's features (see above).

`feature_lists`:

  - A `FeatureList L` may be missing in an example; it is up to the
    parser configuration to determine if this is allowed or considered
    an empty list (zero length).
  - If a `FeatureList L` exists, it may be empty (zero length).
  - If a `FeatureList L` is non-empty, all features within the `FeatureList`
    must have the same data type `T`. Even across `SequenceExample`s, the type `T`
    of the `FeatureList` identified by the same key must be the same. An entry
    without any values may serve as an empty feature.
  - If a `FeatureList L` is non-empty, it is up to the parser configuration
    to determine if all features within the `FeatureList` must
    have the same size.  The same holds for this `FeatureList` across multiple
    examples.
  - For sequence modeling ([example](https://github.com/tensorflow/nmt)), the
    feature lists represent a sequence of frames. In this scenario, all
    `FeatureList`s in a `SequenceExample` have the same number of `Feature`
    messages, so that the i-th element in each `FeatureList` is part of the
    i-th frame (or time step).

**Examples of conformant and non-conformant examples' `FeatureLists`:**

Conformant `FeatureLists`:

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0 ] } } }
    } }
```

Non-conformant `FeatureLists` (mismatched types):

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { int64_list: { value: [ 5 ] } } }
    } }
```

Conditionally conformant `FeatureLists`, the parser configuration determines
if the feature sizes must match:

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0, 6.0 ] } } }
    } }
```

**Examples of conformant and non-conformant `SequenceExample`s:**

Conformant pair of SequenceExample:

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0 ] } } }
     } }

    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0 ] } }
               feature: { float_list: { value: [ 2.0 ] } } }
     } }
```

Conformant pair of `SequenceExample`s:

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0 ] } } }
     } }

    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { }
     } }
```

Conditionally conformant pair of `SequenceExample`s, the parser configuration
determines if the second `feature_lists` is consistent (zero-length) or
invalid (missing "movie_ratings"):

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0 ] } } }
     } }

   feature_lists: { }
```

Non-conformant pair of `SequenceExample`s (mismatched types):

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0 ] } } }
     } }

    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { int64_list: { value: [ 4 ] } }
               feature: { int64_list: { value: [ 5 ] } }
               feature: { int64_list: { value: [ 2 ] } } }
     } }
```

Conditionally conformant pair of `SequenceExample`s; the parser configuration
determines if the feature sizes must match:

```
    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.5 ] } }
               feature: { float_list: { value: [ 5.0 ] } } }
    } }

    feature_lists: { feature_list: {
      key: "movie_ratings"
      value: { feature: { float_list: { value: [ 4.0 ] } }
              feature: { float_list: { value: [ 5.0, 3.0 ] } }
    } }
```
"""
