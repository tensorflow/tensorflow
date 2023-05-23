/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_TF_OP_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_TF_OP_UTILS_H_

#include <string>
#include <vector>

#include "tensorflow/tsl/profiler/utils/tf_op_utils.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::Category;                // NOLINT
using tsl::profiler::DatasetOpEventName;      // NOLINT
using tsl::profiler::IsDatasetOp;             // NOLINT
using tsl::profiler::IsEmbeddingOp;           // NOLINT
using tsl::profiler::IsInfeedEnqueueOp;       // NOLINT
using tsl::profiler::IsJaxOpNameAndType;      // NOLINT
using tsl::profiler::IsJaxOpType;             // NOLINT
using tsl::profiler::IsMemcpyDToDOp;          // NOLINT
using tsl::profiler::IsMemcpyDToHOp;          // NOLINT
using tsl::profiler::IsMemcpyHToDOp;          // NOLINT
using tsl::profiler::IsMemcpyHToHOp;          // NOLINT
using tsl::profiler::IsOutsideCompilationOp;  // NOLINT
using tsl::profiler::IsTfOpName;              // NOLINT
using tsl::profiler::IsTfOpType;              // NOLINT
using tsl::profiler::IteratorName;            // NOLINT
using tsl::profiler::kDatasetOp;              // NOLINT
using tsl::profiler::kMemcpyDToDOp;           // NOLINT
using tsl::profiler::kMemcpyDToHOp;           // NOLINT
using tsl::profiler::kMemcpyHToDOp;           // NOLINT
using tsl::profiler::kMemcpyHToHOp;           // NOLINT
using tsl::profiler::kUnknownOp;              // NOLINT
using tsl::profiler::ParseTensorShapes;       // NOLINT
using tsl::profiler::ParseTfNameScopes;       // NOLINT
using tsl::profiler::ParseTfOpFullname;       // NOLINT
using tsl::profiler::TfOp;                    // NOLINT
using tsl::profiler::TfOpEventName;           // NOLINT

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_TF_OP_UTILS_H_
