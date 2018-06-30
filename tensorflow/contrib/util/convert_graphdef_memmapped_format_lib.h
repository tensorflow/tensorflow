/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CONTRIB_UTIL_CONVERT_GRAPHDEF_MEMMAPPED_FORMAT_LIB_H_
#define TENSORFLOW_CONTRIB_UTIL_CONVERT_GRAPHDEF_MEMMAPPED_FORMAT_LIB_H_

#include <string>

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Converts a "frozen" inference graph (output from the freeze_graph utility)
// into a format in which large Const ops are converted to ImmutableConst ops
// which are memmapped when the graph is executed by TensorFlow.
Status ConvertConstantsToImmutable(const string& in_graph_filename,
                                   const string& out_graph_filename,
                                   int min_conversion_size_bytes);

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_UTIL_CONVERT_GRAPHDEF_MEMMAPPED_FORMAT_LIB_H_
