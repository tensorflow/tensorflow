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
"""SignatureDef utility functions implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_signature_def_by_key(meta_graph_def, signature_def_key):
  """Utility function to get a SignatureDef protocol buffer by its key.

  Args:
    meta_graph_def: MetaGraphDef protocol buffer with the SignatureDefMap to
      look up.
    signature_def_key: Key of the SignatureDef protocol buffer to find in the
      SignatureDefMap.

  Returns:
    A SignatureDef protocol buffer corresponding to the supplied key, if it
    exists.

  Raises:
    ValueError: If no entry corresponding to the supplied key is found in the
    SignatureDefMap of the MetaGraphDef.
  """
  if signature_def_key not in meta_graph_def.signature_def:
    raise ValueError("No SignatureDef with key '%s' found in MetaGraphDef." %
                     signature_def_key)
  return meta_graph_def.signature_def[signature_def_key]
