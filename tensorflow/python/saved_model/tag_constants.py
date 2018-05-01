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
"""Common tags used for graphs in SavedModel.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.tf_export import tf_export


# Tag for the `serving` graph.
SERVING = "serve"
tf_export("saved_model.tag_constants.SERVING").export_constant(
    __name__, "SERVING")

# Tag for the `training` graph.
TRAINING = "train"
tf_export("saved_model.tag_constants.TRAINING").export_constant(
    __name__, "TRAINING")

# Tag for the `gpu` graph.
GPU = "gpu"
tf_export("saved_model.tag_constants.GPU").export_constant(__name__, "GPU")

# Tag for the `tpu` graph.
TPU = "tpu"
tf_export("saved_model.tag_constants.TPU").export_constant(__name__, "TPU")
