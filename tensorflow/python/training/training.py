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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from tensorflow.python.training.experimental.mixed_precision import enable_mixed_precision_graph_rewrite
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
from tensorflow.python.training.tracking.python_state import PythonState
from tensorflow.python.training.tracking.util import Checkpoint
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
from tensorflow.python.training.checkpoint_management import checkpoint_exists
from tensorflow.python.training.checkpoint_management import generate_checkpoint_state_proto
from tensorflow.python.training.checkpoint_management import get_checkpoint_mtimes
from tensorflow.python.training.checkpoint_management import get_checkpoint_state
from tensorflow.python.training.checkpoint_management import latest_checkpoint
from tensorflow.python.training.checkpoint_management import update_checkpoint_state
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.training.saver import import_meta_graph
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

# Docstring definitions for protos.

# LINT.IfChange
BytesList.__doc__ = """\
Container that holds repeated fundamental values of byte type in the `tf.train.Feature` message.

See the [`tf.train.Example`](https://www.tensorflow.org/tutorials/load_data/tfrecord#tftrainexample)
guide for usage details.
"""

FloatList.__doc__ = """\
Container that holds repeated fundamental values of float type in the `tf.train.Feature` message.

See the [`tf.train.Example`](https://www.tensorflow.org/tutorials/load_data/tfrecord#tftrainexample)
guide for usage details.
"""

Int64List.__doc__ = """\
Container that holds repeated fundamental value of int64 type in the `tf.train.Feature` message.

See the [`tf.train.Example`](https://www.tensorflow.org/tutorials/load_data/tfrecord#tftrainexample)
guide for usage details.
"""

Feature.__doc__ = """\
A `Feature` is a list which may hold zero or more values.

There are three base `Feature` types:

  - `tf.train.BytesList`
  - `tf.train.FloatList`
  - `tf.train.Int64List`
"""

Features.__doc__ = """\
Protocol message for describing the `features` of a `tf.train.Example`.

`Features` are organized into categories by name.  The `Features` message
contains the mapping from name to `tf.train.Feature`.

One item value of `Features` for a movie recommendation application:

```
    feature {
      key: "age"
      value { float_list {
        value: 29.0
      }}
    }
    feature {
      key: "movie"
      value { bytes_list {
        value: "The Shawshank Redemption"
        value: "Fight Club"
      }}
    }
    feature {
      key: "movie_ratings"
      value { float_list {
        value: 9.0
        value: 9.7
      }}
    }
    feature {
      key: "suggestion"
      value { bytes_list {
        value: "Inception"
      }}
    }
    feature {
      key: "suggestion_purchased"
      value { int64_list {
        value: 1
      }}
    }
    feature {
      key: "purchase_price"
      value { float_list {
        value: 9.99
      }}
    }
```
"""

FeatureList.__doc__ = "Contains zero or more values of `tf.train.Feature`s."

FeatureLists.__doc__ = ("Contains the mapping from name to "
                        "`tf.train.FeatureList`.")

# LINT.ThenChange(
#     https://www.tensorflow.org/code/tensorflow/core/example/feature.proto)

# LINT.IfChange
Example.__doc__ = """\
An `Example` is a mostly-normalized data format for storing data for training and inference.

It contains a key-value store `features` where each key (string) maps to a
`tf.train.Feature` message. This flexible and compact format allows the
storage of large amounts of typed data, but requires that the data shape
and use be determined by the configuration files and parsers that are used to
read and write this format.

In TensorFlow, `Example`s are read in row-major
format, so any configuration that describes data with rank-2 or above
should keep this in mind. For example, to store an `M x N` matrix of bytes,
the `tf.train.BytesList` must contain M*N bytes, with `M` rows of `N` contiguous values
each. That is, the `BytesList` value must store the matrix as:

```.... row 0 .... // .... row 1 .... // ...........  // ... row M-1 ....```

An `Example` for a movie recommendation application:

```
    features {
      feature {
        key: "age"
        value { float_list {
          value: 29.0
        }}
      }
      feature {
        key: "movie"
        value { bytes_list {
          value: "The Shawshank Redemption"
          value: "Fight Club"
        }}
      }
      feature {
        key: "movie_ratings"
        value { float_list {
          value: 9.0
          value: 9.7
        }}
      }
      feature {
        key: "suggestion"
        value { bytes_list {
          value: "Inception"
        }}
      }
      # Note that this feature exists to be used as a label in training.
      # E.g., if training a logistic regression model to predict purchase
      # probability in our learning tool we would set the label feature to
      # "suggestion_purchased".
      feature {
        key: "suggestion_purchased"
        value { float_list {
          value: 1.0
        }}
      }
      # Similar to "suggestion_purchased" above this feature exists to be used
      # as a label in training.
      # E.g., if training a linear regression model to predict purchase
      # price in our learning tool we would set the label feature to
      # "purchase_price".
      feature {
        key: "purchase_price"
        value { float_list {
          value: 9.99
        }}
      }
    }
```
A conformant `Example` dataset obeys the following conventions:

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
A `SequenceExample` is a format for representing one or more sequences and some context.

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
# pylint: enable=undefined-variable
# LINT.ThenChange(
#     https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
