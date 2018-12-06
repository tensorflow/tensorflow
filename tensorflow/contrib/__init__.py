# pylint: disable=g-import-not-at-top
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
"""contrib module containing volatile or experimental code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import platform

# Add projects here, they will show up under tf.contrib.
from tensorflow.contrib import autograph
from tensorflow.contrib import batching
from tensorflow.contrib import bayesflow
from tensorflow.contrib import checkpoint
if os.name != "nt" and platform.machine() != "s390x":
  from tensorflow.contrib import cloud
from tensorflow.contrib import cluster_resolver
from tensorflow.contrib import coder
from tensorflow.contrib import compiler
from tensorflow.contrib import constrained_optimization
from tensorflow.contrib import copy_graph
from tensorflow.contrib import crf
from tensorflow.contrib import cudnn_rnn
from tensorflow.contrib import data
from tensorflow.contrib import deprecated
from tensorflow.contrib import distribute
from tensorflow.contrib import distributions
from tensorflow.contrib import estimator
from tensorflow.contrib import factorization
from tensorflow.contrib import feature_column
from tensorflow.contrib import framework
from tensorflow.contrib import gan
from tensorflow.contrib import graph_editor
from tensorflow.contrib import grid_rnn
from tensorflow.contrib import image
from tensorflow.contrib import input_pipeline
from tensorflow.contrib import integrate
from tensorflow.contrib import keras
from tensorflow.contrib import kernel_methods
from tensorflow.contrib import labeled_tensor
from tensorflow.contrib import layers
from tensorflow.contrib import learn
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib import linear_optimizer
from tensorflow.contrib import lookup
from tensorflow.contrib import losses
from tensorflow.contrib import memory_stats
from tensorflow.contrib import metrics
from tensorflow.contrib import mixed_precision
from tensorflow.contrib import model_pruning
from tensorflow.contrib import nn
from tensorflow.contrib import opt
from tensorflow.contrib import periodic_resample
from tensorflow.contrib import predictor
from tensorflow.contrib import proto
from tensorflow.contrib import quantization
from tensorflow.contrib import quantize
from tensorflow.contrib import reduce_slice_ops
from tensorflow.contrib import resampler
from tensorflow.contrib import rnn
from tensorflow.contrib import rpc
from tensorflow.contrib import saved_model
from tensorflow.contrib import seq2seq
from tensorflow.contrib import signal
from tensorflow.contrib import slim
from tensorflow.contrib import solvers
from tensorflow.contrib import sparsemax
from tensorflow.contrib import staging
from tensorflow.contrib import stat_summarizer
from tensorflow.contrib import stateless
from tensorflow.contrib import tensor_forest
from tensorflow.contrib import tensorboard
from tensorflow.contrib import testing
from tensorflow.contrib import tfprof
from tensorflow.contrib import timeseries
from tensorflow.contrib import tpu
from tensorflow.contrib import training
from tensorflow.contrib import util
from tensorflow.contrib.eager.python import tfe as eager
from tensorflow.contrib.lite.python import lite
from tensorflow.contrib.optimizer_v2 import optimizer_v2_symbols as optimizer_v2
from tensorflow.contrib.receptive_field import receptive_field_api as receptive_field
from tensorflow.contrib.recurrent.python import recurrent_api as recurrent
from tensorflow.contrib.remote_fused_graph import pylib as remote_fused_graph
from tensorflow.contrib.specs import python as specs
from tensorflow.contrib.summary import summary

from tensorflow.python.util.lazy_loader import LazyLoader
ffmpeg = LazyLoader("ffmpeg", globals(),
                    "tensorflow.contrib.ffmpeg")
del os
del LazyLoader

del absolute_import
del division
del print_function
