/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/cc/small_constant_optimization.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/ctstring_internal.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {

namespace {

constexpr TF_DataType kAllowedDataType[] = {TF_INT32, TF_INT64, TF_FLOAT,
                                            TF_STRING};

void AppendIntValues(const int num_of_elements, const int* int_values,
                     TensorProto* proto) {
  for (int i = 0; i < num_of_elements; ++i) {
    proto->add_int_val(int_values[i]);
  }
}

void AppendInt64Values(const int num_of_elements, const int64_t* int64_values,
                       TensorProto* proto) {
  for (int i = 0; i < num_of_elements; ++i) {
    proto->add_int64_val(int64_values[i]);
  }
}

void AppendStringValues(const int num_of_elements,
                        const TF_TString* string_values, TensorProto* proto) {
  for (int i = 0; i < num_of_elements; ++i) {
    proto->add_string_val(
        std::string(TF_TString_GetDataPointer(&string_values[i]),
                    TF_TString_GetSize(&string_values[i])));
  }
}
void AppendFloatValues(const int num_of_elements, const float* float_values,
                       TensorProto* proto) {
  for (int i = 0; i < num_of_elements; ++i) {
    proto->add_float_val(float_values[i]);
  }
}

}  // namespace

std::optional<NodeDef> ExtractSmallTensorValue(TFE_Context* context,
                                               TFE_TensorHandle* tensor,
                                               const Layout& layout,
                                               TF_Status* status) {
  if (!layout.IsFullyReplicated()) return std::nullopt;
  auto num_elements = TFE_TensorHandleNumElements(tensor, status);
  if (TF_GetCode(status) != TF_OK) return std::nullopt;

  if (num_elements >= kSmallTensorThreshold) return std::nullopt;

  // Check the DType before attempting to resolve the tensor so we don't try to
  // copy resource-dtype tensors off the DTensor device. Currently we only
  // extract small int32/int64_t tensors, primarily to catch shapes and axes,
  // and tf_string tensors that are mostly used in save/restore ops.
  const auto& dtype = TFE_TensorHandleDataType(tensor);
  if (absl::c_find(kAllowedDataType, dtype) == std::end(kAllowedDataType)) {
    return std::nullopt;
  }

  // This is the enum from protobuf, or the following AddNodeAttr will always
  // set the integer field.
  const auto& datatype = static_cast<DataType>(dtype);
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> value_tensor(
      TFE_TensorHandleResolve(tensor, status), TF_DeleteTensor);
  if (TF_GetCode(status) != TF_OK) return std::nullopt;

  NodeDef node_def;
  node_def.set_op("Const");
  AddNodeAttr("dtype", datatype, &node_def);

  TensorProto tensor_proto;
  tensor_proto.set_dtype(datatype);
  switch (dtype) {
    case TF_INT32:
      AppendIntValues(num_elements,
                      static_cast<int*>(TF_TensorData(value_tensor.get())),
                      &tensor_proto);
      break;
    case TF_INT64:
      AppendInt64Values(
          num_elements,
          static_cast<const int64_t*>(TF_TensorData(value_tensor.get())),
          &tensor_proto);
      break;
    case TF_STRING:
      AppendStringValues(
          num_elements,
          static_cast<const TF_TString*>(TF_TensorData(value_tensor.get())),
          &tensor_proto);
      break;
    case TF_FLOAT:
      AppendFloatValues(
          num_elements,
          static_cast<const float*>(TF_TensorData(value_tensor.get())),
          &tensor_proto);
      break;
    default:
      TF_SetStatus(status, TF_INTERNAL,
                   absl::StrCat("dtype: ", dtype,
                                " fell through the supported extraction list. "
                                "This should not happen.")
                       .c_str());
      return std::nullopt;
  }

  std::vector<int64_t> dim_list;
  int num_dims = value_tensor->tensor->NumDims();
  dim_list.reserve(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    dim_list.push_back(value_tensor->tensor->Dim(i));
  }

  TensorShape shape(std::move(dim_list));
  shape.AsProto(tensor_proto.mutable_tensor_shape());
  AddNodeAttr("value", tensor_proto, &node_def);

  AddNodeAttr(kLayoutAttr, {layout.ToString()}, &node_def);
  AddNodeAttr(kMeshAttr, layout.mesh().ToString(), &node_def);
  return node_def;
}

bool ShouldFoldInputArgument(absl::string_view operation_name,
                             int input_index) {
  // Fold if we are in a function or if a special eager op.
  // TODO(xiejw,power): Think about how to generalize this so it does not depend
  // on operation_name. For example, we can check the max abs value of the
  // tensor value.
  if (operation_name == absl::string_view("StatelessRandomUniform") ||
      operation_name == absl::string_view("StatelessRandomUniformFullInt") ||
      operation_name == absl::string_view("StatelessRandomNormal") ||
      operation_name == absl::string_view("StatelessTruncatedNormal")) {
    // For all stateless rng ops, we avoid fold seed (input_index==1) in graph.
    // This is an important optimization to avoid unnecessary MLIR SPMD lowering
    // and TPU compilation during model parameters initialization process.
    // which typically have the same shape for rng ops but different seeds.
    return input_index != 1;
  }

  return true;
}

bool NodeDefsHaveDifferentTensorProto(const NodeDef& a, const NodeDef& b) {
  const TensorProto* tensor_proto_a;
  bool read_a_tensor_proto = TryGetNodeAttr(a, "value", &tensor_proto_a);
  if (!read_a_tensor_proto) return true;

  const TensorProto* tensor_proto_b;
  bool read_b_tensor_proto = TryGetNodeAttr(b, "value", &tensor_proto_b);
  if (!read_b_tensor_proto) return true;
  return !protobuf::util::MessageDifferencer::Equals(*tensor_proto_a,
                                                     *tensor_proto_b);
}

}  // namespace dtensor
}  // namespace tensorflow
