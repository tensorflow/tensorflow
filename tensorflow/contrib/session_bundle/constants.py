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

"""Constants for export/import."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


VERSION_FORMAT_SPECIFIER = "%08d"
ASSETS_DIRECTORY = "assets"
EXPORT_BASE_NAME = "export"
EXPORT_SUFFIX_NAME = "meta"
META_GRAPH_DEF_FILENAME = "export.meta"
VARIABLES_FILENAME = "export"
VARIABLES_FILENAME_PATTERN = "export-?????-of-?????"
VARIABLES_FILENAME_PATTERN_V2 = "export.data-?????-of-?????"
VARIABLES_INDEX_FILENAME_V2 = "export.index"
INIT_OP_KEY = "serving_init_op"
SIGNATURES_KEY = "serving_signatures"
ASSETS_KEY = "serving_assets"
GRAPH_KEY = "serving_graph"
