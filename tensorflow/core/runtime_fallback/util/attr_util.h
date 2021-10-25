/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_ATTR_UTIL_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_ATTR_UTIL_H_

#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tfrt/bef/bef_encoding.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attr_type.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

// Converts a TFRT string_view to the Abseil version.
inline absl::string_view ToAbslStringView(tfrt::string_view sv) {
  return absl::string_view(sv.data(), sv.size());
}

// Parses the string representation of the DataType in `dtype` into `data_type`.
// Aborts the program for unsupported dtypes.
tensorflow::Status ParseTfDataType(absl::string_view dtype,
                                   DataType* data_type);

// The following 2 functions convert between Tensorflow DataTypes and
// OpAttrTypes. The mapping between OpAttrType and DataType is defined in
// attr_type.def. Aborts on unsupported types.
DataType ConvertToTfDataType(tfrt::OpAttrType op_attr_type);
tfrt::OpAttrType ConvertFromTfDataType(DataType data_type);

// The following 2 functions convert between BEF attribute types and Tensorflow
// DataTypes. Aborts on unsupported datatypes.
DataType ConvertBefAttrTypeToTfDataType(tfrt::DType attr_type);
tfrt::DType ConvertTfDataTypeToBefAttrType(DataType data_type);

// Parses the tensor valued `attr_value` and constructs the tensor with its
// contents in `tensor`. Returns OK status on success, INVALID_ARGUMENT on
// failure.
tensorflow::Status ParseTensorAttrValue(absl::string_view attr_value,
                                        tensorflow::Tensor* tensor);

// Parses a string of the form "[1,2,3,...]" in `attr_value` and returns the
// constituent dimension sizes (shape) in `int_list_val`. Returns
// INVALID_ARGUMENT on invalid input.
tensorflow::Status ParseTensorShapeAttrValue(absl::string_view attr_value,
                                             std::vector<int64_t>* shape_val);

// Parses a boolean from `attr_value` into `bool_val` and returns OK status on
// success. Returns INVALID_ARGUMENT on invalid input.
tensorflow::Status ParseBoolAttrValue(absl::string_view attr_value,
                                      bool* bool_val);

// Parses an int64_t from `attr_value` into `int_val` and returns OK status on
// success. Returns INVLAID_ARGUMENT on invalid input.
tensorflow::Status ParseIntAttrValue(absl::string_view attr_value,
                                     int64_t* int_val);

inline std::vector<absl::string_view> AttrValueSplit(absl::string_view str) {
  return absl::StrSplit(str, absl::MaxSplits('$', 1));
}

// Returns true if `attr_name` is an attribute that is not required by TFRT
// (usually added by stages higher in the lowering process)
bool IsUnusedAttribute(absl::string_view attr_name);

// Fills in the passed in AttrValueMap `attr_value_map` with attributes from
// `attrs`.
llvm::Error FillAttrValueMap(const tfrt::OpAttrsRef& attrs,
                             tfrt::HostContext* host,
                             AttrValueMap* attr_value_map);

// Fills in the passed in AttrValueMap `attr_value_map`.
tensorflow::Status SetUpAttrValueMap(tfrt::AggregateAttr op_attr_array,
                                     tfrt::AggregateAttr op_func_attr_array,
                                     tensorflow::AttrValueMap* attr_value_map);

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_ATTR_UTIL_H_
