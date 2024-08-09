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
#include "tensorflow/compiler/mlir/tfrt/translate/mlrt/test_utils.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/tfrt/mlrt/attribute/attribute.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/kernel.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/interpreter_testutil.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace mlrt {
namespace testing {

absl::StatusOr<std::string> EncodeAttribute(const tensorflow::AttrValue& attr) {
  if (attr.has_b()) {
    std::string result;
    result.resize(sizeof(uint8_t));
    uint8_t v = attr.b();
    std::memcpy(result.data(), &v, sizeof(v));
    return result;
  }

  if (attr.has_i()) {
    std::string result;
    result.resize(sizeof(int64_t));
    int64_t v = attr.i();
    std::memcpy(result.data(), &v, sizeof(v));
    return result;
  }

  if (attr.has_f()) {
    std::string result;
    result.resize(sizeof(float));
    float v = attr.f();
    std::memcpy(result.data(), &v, sizeof(v));
    return result;
  }

  if (attr.has_s()) {
    return attr.s();
  }

  if (attr.has_list()) {
    if (attr.list().s_size() > 0) {
      mlrt::bc::Buffer buffer;
      mlrt::bc::Allocator allocator(&buffer);
      auto ctor = mlrt::bc::New<mlrt::bc::Vector<mlrt::bc::String>>(
          &allocator, attr.list().s_size());

      for (int i = 0; i < attr.list().s_size(); ++i) {
        ctor.ConstructAt(i, attr.list().s(i));
      }

      return std::string(buffer.data(), buffer.size());
    }
  }

  if (attr.has_tensor()) {
    mlrt::bc::Buffer buffer;
    mlrt::bc::Allocator allocator(&buffer);

    tensorflow::Tensor tensor;
    if (!tensor.FromProto(attr.tensor())) {
      return absl::InvalidArgumentError("Invalid tensor proto.");
    }

    auto tensor_attr_ctor = mlrt::bc::New<tensorflow::tf_mlrt::TensorAttr>(
        &allocator, tensor.dtype());

    auto shape = tensor.shape().dim_sizes();

    tensor_attr_ctor.construct_shape(shape.size())
        .Assign(shape.begin(), shape.end());

    auto tensor_data = tensor.tensor_data();
    tensor_attr_ctor.construct_data(tensor_data.size())
        .Place(tensor_data.data(), tensor_data.size());

    return std::string(buffer.data(), buffer.size());
  }

  // TODO(chky,rohitju): Add more attribute support.

  return absl::InvalidArgumentError("Unsupported attribute.");
}

namespace {

bool CanBeInlined(const tensorflow::AttrValue& attr) {
  return attr.has_b() || attr.has_f();
}

}  // namespace

absl::Status EncodeAttributes(AttributeTable& attributes,
                              const tensorflow::AttrValueMap& attr_map) {
  std::vector<std::pair<std::string, tensorflow::AttrValue>> attrs(
      attr_map.begin(), attr_map.end());
  std::sort(attrs.begin(), attrs.end(),
            [](const auto& x, const auto& y) { return x.first < y.first; });

  for (int i = 0; i < attrs.size(); ++i) {
    const tensorflow::AttrValue& attr = attrs[i].second;
    TF_ASSIGN_OR_RETURN(auto attr_str, EncodeAttribute(attr));
    if (CanBeInlined(attr)) {
      attributes.AddInline(absl::StrCat(i), attr_str);
    } else {
      attributes.Add(absl::StrCat(i), attr_str);
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::pair<mlrt::bc::Kernel, mlrt::bc::Vector<mlrt::bc::String>>>
CreateKernelAndAttrs(int num_inputs, int num_outputs,
                     mlrt::ExecutionContext& exec_ctx, mlrt::bc::Buffer* buffer,
                     const tensorflow::AttrValueMap& attrs) {
  mlrt::bc::Allocator allocator(buffer);
  auto attributes_ctor = mlrt::bc::New<mlrt::bc::Vector<mlrt::bc::String>>(
      &allocator, attrs.size());
  AttributeTable attribute_table(attributes_ctor);
  TF_RETURN_IF_ERROR(EncodeAttributes(attribute_table, attrs));

  auto kernel_ctor = mlrt::bc::New<mlrt::bc::Kernel>(&allocator);
  kernel_ctor.set_code(0);

  std::vector<int> input_indices(num_inputs);
  std::iota(input_indices.begin(), input_indices.end(), 0);
  kernel_ctor.construct_arguments(input_indices.size())
      .Assign(input_indices.begin(), input_indices.end());

  std::vector<int> output_indices(num_outputs);
  std::iota(output_indices.begin(), output_indices.end(), num_inputs);
  kernel_ctor.construct_results(output_indices.size())
      .Assign(output_indices.begin(), output_indices.end());

  std::vector<uint32_t> attr_indices;
  attr_indices.reserve(attrs.size());
  for (int i = 0; i < attrs.size(); ++i) {
    attr_indices.push_back(attribute_table.GetHandle(absl::StrCat(i)));
  }

  kernel_ctor.construct_attributes(attr_indices.size())
      .Assign(attr_indices.begin(), attr_indices.end());

  mlrt::bc::Vector<mlrt::bc::String> attributes(
      buffer->Get(attributes_ctor.address()));
  mlrt::bc::Kernel kernel(buffer->Get(kernel_ctor.address()));

  return std::make_pair(kernel, attributes);
}

}  // namespace testing
}  // namespace mlrt
