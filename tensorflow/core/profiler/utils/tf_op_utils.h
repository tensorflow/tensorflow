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

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace profiler {

// Special op types.
TF_CONST_INIT extern const absl::string_view kUnknownOp;
TF_CONST_INIT extern const absl::string_view kDatasetOp;
TF_CONST_INIT extern const absl::string_view kMemcpyHToDOp;
TF_CONST_INIT extern const absl::string_view kMemcpyDToHOp;

enum class Category {
  kTensorFlow,
  kJax,
  kTfData,
  kMemcpyHToD,
  kMemcpyDToH,
  kUnknown,
};

// Breaks a TensorFlow op fullname into name and type.
struct TfOp {
  Category category;
  absl::string_view name;
  absl::string_view type;
};
TfOp ParseTfOpFullname(absl::string_view tf_op_fullname);

// Returns a vector of TF name scopes extracted from tf_op_full_name.
std::vector<absl::string_view> ParseTfNameScopes(const TfOp& tf_op);

// Trace event name for TF ops is the op type so they have the same color in
// trace viewer.
std::string TfOpEventName(const TfOp& tf_op);
std::string TfOpEventName(absl::string_view tf_op_fullname);

// Trace event name for dataset ops.
std::string DatasetOpEventName(absl::string_view full_name);

// Returns the iterator name without prefix and parent iterator names.
std::string IteratorName(absl::string_view full_name);

// Returns true if the given name is a TensorFlow Dataset Op.
inline bool IsDatasetOp(absl::string_view tf_op_type) {
  return tf_op_type == kDatasetOp;
}
inline bool IsDatasetOp(const TfOp& tf_op) {
  return tf_op.category == Category::kTfData;
}

// Returns true if the given name is a TensorFlow Infeed Enqueue Op.
inline bool IsInfeedEnqueueOp(absl::string_view tf_op_type) {
  return tf_op_type == "InfeedEnqueue" || tf_op_type == "InfeedEnqueueTuple";
}

// Returns true if the given op is for outside compilation.
inline bool IsOutsideCompilationOp(absl::string_view tf_op_fullname,
                                   absl::string_view hlo_expression) {
  if (absl::EndsWith(tf_op_fullname, ":XlaSendToHost")) return true;
  if (absl::StrContains(hlo_expression, "send-done") &&
      absl::StrContains(hlo_expression, "is_host_transfer=true"))
    return true;
  return false;
}

// Returns true if the given name is a TensorFlow embedding op.
inline bool IsEmbeddingOp(absl::string_view tf_op_fullname) {
  return absl::StrContains(tf_op_fullname, "Embedding");
}

// Returns true if the given op is for copying data from host to device.
inline bool IsMemcpyHToDOp(absl::string_view tf_op_type) {
  return tf_op_type == kMemcpyHToDOp;
}

// Returns true if the given op is for copying data from device to host.
inline bool IsMemcpyDToHOp(absl::string_view tf_op_type) {
  return tf_op_type == kMemcpyDToHOp;
}

// Splits a string of tensor shapes in "(shape1;shape2;...)" format, i.e.,
// delimited by '(' and ')' and separated by ';', into the individual shapes.
std::vector<absl::string_view> ParseTensorShapes(
    absl::string_view tensor_shapes);

// Returns true if the given string matches OpDef.name pattern.
bool IsTfOpName(absl::string_view op_name);

// Returns true if the given string matches NodeDef.name pattern.
bool IsTfOpType(absl::string_view op_type);

// Returns true if the given string matches JAX pattern.
bool IsJaxOpType(absl::string_view op_type);

// Returns true if the given strings match JAX pattern.
bool IsJaxOpNameAndType(absl::string_view op_name, absl::string_view op_type);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_TF_OP_UTILS_H_
