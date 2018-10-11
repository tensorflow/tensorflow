# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Import Estimator APIs.

Note: This file is imported by the create_estimator_api genrule. It must
transitively import all Estimator modules/packages for their @estimator_export
annotations to generate the public Estimator python API.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.estimator.estimator_lib
