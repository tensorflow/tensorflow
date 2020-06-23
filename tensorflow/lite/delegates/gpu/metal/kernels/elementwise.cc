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
#include <unordered_map>
#include <vector>

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
  const std::unordered_map<OperationType, std::string> functors{
      {OperationType::ABS, "abs($0)"},
      {OperationType::SIN, "sin($0)"},
      {OperationType::HARD_SWISH,
       "$0 * clamp($0 / 6.0f + FLT4(0.5f), FLT4(0.0f), FLT4(1.0f))"},
      {OperationType::COS, "cos($0)"},
      {OperationType::EXP, "exp($0)"},
      {OperationType::LOG, "log($0)"},
      {OperationType::SQRT, "sqrt($0)"},
      {OperationType::RSQRT, "1.0 / sqrt($0)"},
      {OperationType::SQUARE, "$0 * $0"},
      {OperationType::SIGMOID, "1.0 / (1.0 + exp(-1.0 * $0))"},
      {OperationType::TANH, "tanh($0)"},
  };

  if (functors.find(op_type) == functors.end()) {
    return "Error, unknown op";
  }

  return absl::Substitute(functors.at(op_type), value);
}

std::string TwoInputFunctor(OperationType op_type, const std::string& value0,
                            const std::string& value1) {
  const std::unordered_map<OperationType, std::string> functors{
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

std::vector<ComputeTaskDescriptorPtr> ElementwiseWithTwoInputs(
    int id, std::vector<ValueId> input_ids, ValueId output_id,
    const BHWC& second_shape, OperationType op_type) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = true;
  const std::string x_coord = second_shape.w == 1 ? "0" : "int(gid.x)";
  const std::string y_coord = second_shape.h == 1 ? "0" : "int(gid.y)";
  const std::string s_coord = second_shape.c == 1 ? "0" : "int(gid.z)";
  std::string code =
      "FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid, device FLT4* "
      "const second_tensor, int2 second_size) {\n";
  code += "  int second_index = (" + s_coord + " * second_size.y + " + y_coord +
          ") * second_size.x + " + x_coord + ";\n";
  code += "  FLT4 src_1 = second_tensor[second_index];\n";
  if (second_shape.c == 1) {
    code += "  src_1.y = src_1.x;\n";
    code += "  src_1.z = src_1.x;\n";
    code += "  src_1.w = src_1.x;\n";
  }
  code += "  return " + TwoInputFunctor(op_type, "value", "src_1") + ";\n";
  code += "}\n";

  desc->shader_source = code;

  desc->input_buffers = {
      {input_ids[0], "device FLT4* const"},
      {input_ids[1], "device FLT4* const"},
  };
  desc->output_buffer = {output_id};

  desc->uniform_buffers = {
      {"constant int2&",
       [input_ids, output_id](const std::map<ValueId, BHWC>& buffers) {
         const auto& input_dim_1 = buffers.find(input_ids[1])->second;
         std::vector<int> uniform_params{
             input_dim_1.w,
             input_dim_1.h,
         };
         return GetByteBuffer(uniform_params);
       }},
  };
  return {desc};
}

std::vector<ComputeTaskDescriptorPtr> ElementwiseWithOneInput(
    int id, ValueId input_id, ValueId output_id, OperationType op_type) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = true;
  desc->shader_source =
      "FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid) {\n";
  desc->shader_source +=
      "    return " + OneInputFunctor(op_type, "value") + ";\n";
  desc->shader_source += "  }";

  desc->input_buffers = {{input_id}};
  desc->output_buffer = {output_id};
  return {desc};
}

std::vector<ComputeTaskDescriptorPtr> ElementwiseWithOneInputAndConstantArguent(
    int id, ValueId input_id, ValueId output_id, const RuntimeOptions& options,
    OperationType op_type, const TensorOrScalar& attr) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = true;
  auto scalar = absl::get_if<float>(&attr);
  auto linear_buf = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr);
  auto hwc_buf = absl::get_if<Tensor<HWC, DataType::FLOAT32>>(&attr);
  std::string param_desc;
  if (scalar) {
    param_desc += ", float scalar_val";
  }
  if (linear_buf) {
    param_desc += ", device FLT4* const linear_buf";
  }
  if (hwc_buf) {
    param_desc += ", device FLT4* const hwc_buf, int2 hwc_size";
  }
  desc->shader_source =
      "FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid" + param_desc +
      ") {\n";
  if (scalar) {
    desc->shader_source += "     FLT4 second_arg = FLT4(scalar_val);\n";
  } else if (linear_buf) {
    desc->shader_source += "     FLT4 second_arg = linear_buf[gid.z];\n";
  } else if (hwc_buf) {
    const std::string x_coord = hwc_buf->shape.w == 1 ? "0" : "int(gid.x)";
    const std::string y_coord = hwc_buf->shape.h == 1 ? "0" : "int(gid.y)";
    const std::string s_coord = hwc_buf->shape.c == 1 ? "0" : "int(gid.z)";
    std::string index = "(" + s_coord + " * hwc_size.y + " + y_coord +
                        ") * hwc_size.x + " + x_coord;
    desc->shader_source += "  FLT4 second_arg = hwc_buf[" + index + "];\n";
    if (hwc_buf->shape.c == 1) {
      desc->shader_source += "  second_arg.y = second_arg.x;\n";
      desc->shader_source += "  second_arg.z = second_arg.x;\n";
      desc->shader_source += "  second_arg.w = second_arg.x;\n";
    }
  }
  desc->shader_source +=
      "    return " + TwoInputFunctor(op_type, "value", "second_arg") + ";\n";
  desc->shader_source += "  }";

  desc->input_buffers = {{input_id}};
  desc->output_buffer = {output_id};
  if (scalar) {
    std::vector<uint8_t> scalar_bits =
        GetByteBuffer(std::vector<float>{*scalar});
    desc->uniform_buffers = {
        {"constant float&",
         [scalar_bits](const std::map<ValueId, BHWC>& buffers) {
           return scalar_bits;
         }},
    };
  } else if (linear_buf) {
    desc->immutable_buffers = {
        {"device FLT4* const",
         GetByteBufferConverted(linear_buf->data, options.storage_precision)},
    };
  } else if (hwc_buf) {
    std::vector<uint8_t> size_bits =
        GetByteBuffer(std::vector<int>{hwc_buf->shape.w, hwc_buf->shape.h});
    desc->uniform_buffers = {
        {"constant int2&",
         [size_bits](const std::map<ValueId, BHWC>& buffers) {
           return size_bits;
         }},
    };
    desc->immutable_buffers = {
        {"device FLT4* const",
         GetByteBufferConverted(ConvertToPHWC4(*hwc_buf),
                                options.storage_precision)},
    };
  }
  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
