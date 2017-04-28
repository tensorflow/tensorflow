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
"""Linear algebra libraries.

See the @{$python/contrib.linalg} guide.

@@LinearOperator
@@LinearOperatorDiag
@@LinearOperatorIdentity
@@LinearOperatorScaledIdentity
@@LinearOperatorFullMatrix
@@LinearOperatorTriL
@@LinearOperatorUDVHUpdate
@@LinearOperatorComposition
@@add_operators

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import,line-too-long,g-importing-member

from tensorflow.contrib.linalg.python.ops.linear_operator import *
from tensorflow.contrib.linalg.python.ops.linear_operator_addition import *
from tensorflow.contrib.linalg.python.ops.linear_operator_composition import *
from tensorflow.contrib.linalg.python.ops.linear_operator_diag import *
from tensorflow.contrib.linalg.python.ops.linear_operator_full_matrix import *
from tensorflow.contrib.linalg.python.ops.linear_operator_identity import *
from tensorflow.contrib.linalg.python.ops.linear_operator_tril import *
from tensorflow.contrib.linalg.python.ops.linear_operator_udvh_update import *

# pylint: enable=unused-import,wildcard-import,line-too-long,g-importing-member

from tensorflow.python.util.all_util import remove_undocumented
remove_undocumented(__name__)
