/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Utility functions in support of the XRT API.

#ifndef TENSORFLOW_COMPILER_XRT_XRT_UTIL_H_
#define TENSORFLOW_COMPILER_XRT_XRT_UTIL_H_

#include "tensorflow/compiler/xla/xla.pb.h"

namespace tensorflow {

// Filters the debug options provided as argument according to the value of the
// TF_XLA_DEBUG_OPTIONS_PASSTHROUGH environment variable. If such variable is
// set to "1" or "true", the debug options will be returned as is. Otherwise
// only a subset of them will be set in the returned ones, and all the paths
// contained in it, will be limited to gs:// and bigstore:// ones.
xla::DebugOptions BuildXlaDebugOptions(const xla::DebugOptions& ref_options);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_XRT_UTIL_H_
