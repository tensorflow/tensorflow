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

#include "tensorflow/lite/delegates/gpu/metal/kernels/elementwise.h"

#include <cstddef>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {

namespace {

std::string OneInputFunctor(OperationType op_type, const std::string& value) {
  const absl::flat_hash_map<OperationType, std::string> functors{
      {OperationType::ABS, "abs($0)"},
      {OperationType::SIN, "sin($0)"},
      {OperationType::HARD_SWISH,
       "$0 * clamp($0 / 6.0f + FLT4(0.5f), FLT4(0.0f), FLT4(1.0f))"},
      {OperationType::COS, "cos($0)"},
      {OperationType::ELU,
       "FLT4($0.x < FLT(0.0f) ? exp($0.x) - FLT(1.0f) : $0.x,"
       "$0.y < FLT(0.0f) ? exp($0.y) - FLT(1.0f) : $0.y,"
       "$0.z < FLT(0.0f) ? exp($0.z) - FLT(1.0f) : $0.z,"
       "$0.w < FLT(0.0f) ? exp($0.w) - FLT(1.0f) : $0.w)"},
      {OperationType::EXP, "exp($0)"},
      {OperationType::LOG, "log($0)"},
      {OperationType::NEG, "-($0)"},
      {OperationType::SQRT, "sqrt($0)"},
      {OperationType::RSQRT, "1.0 / sqrt($0)"},
      {OperationType::SQUARE, "$0 * $0"},
      {OperationType::SIGMOID, "1.0 / (1.0 + exp(-1.0 * $0))"},
      {OperationType::TANH, "tanh($0)"},
      {OperationType::COPY, "$0"},
  };

  if (functors.find(op_type) == functors.end()) {
    return "Error, unknown op";
  }

  return absl::Substitute(functors.at(op_type), value);
}

std::string TwoInputFunctor(OperationType op_type, const std::string& value0,
                            const std::string& value1) {
  const absl::flat_hash_map<OperationType, std::string> functors{
      {OperationType::ADD, "$0 + $1"},
      {OperationType::DIV, "$0 / $1"},
      {OperationType::MAXIMUM, "max($0, $1)"},
      {OperationType::MINIMUM, "min($0, $1)"},
      {OperationType::MUL, "$0 * $1"},
      {OperationType::POW, "pow($0, $1)"},
      {OperationType::SQUARED_DIFF, "($0 - $1) * ($0 - $1)"},
      {OperationType::SUB, "$0 - $1"},
  };

  if (functors.find(op_type) == functors.end()) {
    return "Error, unknown op";
  }

  return absl::Substitute(functors.at(op_type), value0, value1);
}

}  // namespace

ComputeTaskDescriptor ElementwiseWithTwoInputs(const OperationDef& definition,
                                               const BHWC& second_shape,
                                               OperationType op_type) {
  ComputeTaskDescriptor desc(definition);
  desc.is_linkable = true;
  const std::string x_coord = second_shape.w == 1 ? "0" : "gid.x";
  const std::string y_coord = second_shape.h == 1 ? "0" : "gid.y";
  const std::string s_coord = second_shape.c == 1 ? "0" : "gid.z";
  std::string code;
  code = "  FLT4 src_1 = args.second_tensor.Read(" + x_coord + ", " + y_coord +
         ", " + s_coord + ");\n";
  if (second_shape.c == 1) {
    code += "  src_1.y = src_1.x;\n";
    code += "  src_1.z = src_1.x;\n";
    code += "  src_1.w = src_1.x;\n";
  }
  code += "  value = " + TwoInputFunctor(op_type, "value", "src_1") + ";\n";

  desc.shader_source = code;

  desc.AddSrcTensor("second_tensor", definition.src_tensors[1]);
  return desc;
}

ComputeTaskDescriptor ElementwiseWithOneInput(const OperationDef& definition,
                                              OperationType op_type) {
  ComputeTaskDescriptor desc(definition);
  desc.is_linkable = true;
  desc.shader_source =
      "    value = " + OneInputFunctor(op_type, "value") + ";\n";
  return desc;
}

ComputeTaskDescriptor ElementwiseWithOneInputAndConstantArguent(
    const OperationDef& definition, OperationType op_type,
    const TensorOrScalar& attr) {
  auto scalar = absl::get_if<float>(&attr);
  auto linear_buf = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr);
  auto hwc_buf = absl::get_if<Tensor<HWC, DataType::FLOAT32>>(&attr);
  ComputeTaskDescriptor desc(definition);
  desc.is_linkable = true;
  if (scalar) {
    desc.args.AddFloat("scalar_val", *scalar);
    desc.shader_source += "  FLT4 second_arg = FLT4(args.scalar_val);\n";
  } else if (linear_buf) {
    auto data_type = DeduceDataTypeFromPrecision(definition.precision);
    const int dst_channels_aligned = AlignByN(linear_buf->shape.v, 4);
    BufferDescriptor linear_desc;
    linear_desc.element_type = data_type;
    linear_desc.element_size = 4;
    linear_desc.data = GetByteBufferConvertedResized(
        linear_buf->data, data_type, dst_channels_aligned);
    linear_desc.size = linear_desc.data.size();
    desc.args.AddObject(
        "linear", absl::make_unique<BufferDescriptor>(std::move(linear_desc)));
    desc.shader_source += "  FLT4 second_arg = args.linear.Read(gid.z);\n";
  } else if (hwc_buf) {
    TensorDescriptor hwc_desc{definition.GetDataType(),
                              TensorStorageType::BUFFER, Layout::HWC};
    hwc_desc.UploadData(*hwc_buf);
    desc.args.AddObject(
        "hwc", absl::make_unique<TensorDescriptor>(std::move(hwc_desc)));

    const std::string x_coord = hwc_buf->shape.w == 1 ? "0" : "gid.x";
    const std::string y_coord = hwc_buf->shape.h == 1 ? "0" : "gid.y";
    const std::string s_coord = hwc_buf->shape.c == 1 ? "0" : "gid.z";
    desc.shader_source += "  FLT4 second_arg = args.hwc.Read(" + x_coord +
                          ", " + y_coord + ", " + s_coord + ");\n";
    if (hwc_buf->shape.c == 1) {
      desc.shader_source += "  second_arg.y = second_arg.x;\n";
      desc.shader_source += "  second_arg.z = second_arg.x;\n";
      desc.shader_source += "  second_arg.w = second_arg.x;\n";
    }
  }
  desc.shader_source +=
      "  value = " + TwoInputFunctor(op_type, "value", "second_arg") + ";\n";
  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
