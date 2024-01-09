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

#include "tensorflow/lite/delegates/gpu/gl/kernels/elementwise.h"

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class ElementwiseOneArgument : public NodeShader {
 public:
  explicit ElementwiseOneArgument(OperationType operation_type)
      : operation_type_(operation_type) {}
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    std::string source;
    switch (operation_type_) {
      case OperationType::ABS:
        source = "value_0 = abs(value_0);";
        break;
      case OperationType::COS:
        source = "value_0 = cos(value_0);";
        break;
      case OperationType::COPY:
        source = "value_0 = value_0;";
        break;
      case OperationType::ELU:
        source = R"(
            value_0.x = value_0.x < 0.0 ? exp(value_0.x) - 1.0 : value_0.x;
            value_0.y = value_0.y < 0.0 ? exp(value_0.y) - 1.0 : value_0.y;
            value_0.z = value_0.z < 0.0 ? exp(value_0.z) - 1.0 : value_0.z;
            value_0.w = value_0.w < 0.0 ? exp(value_0.w) - 1.0 : value_0.w;
        )";
        break;
      case OperationType::EXP:
        source = "value_0 = exp(value_0);";
        break;
      case tflite::gpu::OperationType::FLOOR:
        source = "value_0 = floor(value_0);";
        break;
      case tflite::gpu::OperationType::GELU:
        // Matches the approximate implementation of the cpu reference op.
        // There's no gpu implementation of erfc so we can't match the accurate
        // version.
        // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        source =
            "value_0 = 0.5 * value_0 * (1.0 + tanh(0.7978845608 * (value_0 + "
            "0.044715 * value_0 * value_0 * value_0)));";
        break;
      case OperationType::HARD_SWISH:
        source =
            "value_0 *= clamp(value_0 / 6.0 + vec4(0.5), vec4(0.0), "
            "vec4(1.0));";
        break;
      case OperationType::LOG:
        source = R"(
            const float nan = normalize(vec4(0, 0, 0, 0)).x;
            value_0.x = value_0.x > 0.0 ? log(value_0.x) : nan;
            value_0.y = value_0.y > 0.0 ? log(value_0.y) : nan;
            value_0.z = value_0.z > 0.0 ? log(value_0.z) : nan;
            value_0.w = value_0.w > 0.0 ? log(value_0.w) : nan;
        )";
        break;
      case OperationType::NEG:
        source = "value_0 = -(value_0);";
        break;
      case OperationType::RSQRT:
        source = R"(
            const float nan = normalize(vec4(0, 0, 0, 0)).x;
            value_0.x = value_0.x > 0.0 ? 1.0 / sqrt(value_0.x) : nan;
            value_0.y = value_0.y > 0.0 ? 1.0 / sqrt(value_0.y) : nan;
            value_0.z = value_0.z > 0.0 ? 1.0 / sqrt(value_0.z) : nan;
            value_0.w = value_0.w > 0.0 ? 1.0 / sqrt(value_0.w) : nan;
        )";
        break;
      case OperationType::SIGMOID:
        source = "value_0 = 1.0 / (1.0 + exp(-1.0 * value_0));";
        break;
      case OperationType::SIN:
        source = "value_0 = sin(value_0);";
        break;
      case OperationType::SQRT:
        source = R"(
            const float nan = normalize(vec4(0, 0, 0, 0)).x;
            value_0.x = value_0.x >= 0.0 ? sqrt(value_0.x) : nan;
            value_0.y = value_0.y >= 0.0 ? sqrt(value_0.y) : nan;
            value_0.z = value_0.z >= 0.0 ? sqrt(value_0.z) : nan;
            value_0.w = value_0.w >= 0.0 ? sqrt(value_0.w) : nan;
        )";
        break;
      case OperationType::SQUARE:
        source = "value_0 = value_0 * value_0;";
        break;
      case OperationType::TANH:
        source = "value_0 = tanh(value_0);";
        break;
      default:
        return absl::InvalidArgumentError(
            "Incorrect elementwise operation type.");
    }
    *generated_code = {
        /*parameters=*/{},
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        source,
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }

 private:
  OperationType operation_type_;
};

class ElementwiseTwoArguments : public NodeShader {
 public:
  explicit ElementwiseTwoArguments(OperationType operation_type)
      : operation_type_(operation_type) {}

  inline bool IsElementwiseSupported(const GenerationContext& ctx) const {
    return ctx.input_shapes.size() == 2 &&
           ctx.input_shapes[0] == ctx.input_shapes[1];
  }

  inline bool IsBroadcastSupported(const GenerationContext& ctx) const {
    return ctx.input_shapes.size() == 2 && ctx.input_shapes[1][1] == 1 &&
           ctx.input_shapes[1][2] == 1 &&
           ctx.input_shapes[0][3] == ctx.input_shapes[1][3];
  }

  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    std::vector<Variable> parameters;
    std::vector<std::pair<std::string, Object>> objects;
    std::string argument0, argument1;
    if (IsElementwiseSupported(ctx)) {
      argument0 = "value_0";
      argument1 = "value_1";
    } else if (IsBroadcastSupported(ctx)) {
      argument0 = "$input_data_0[gid.x, gid.y, gid.z]$";
      argument1 = "$input_data_1[0, 0, gid.z]$";
    } else {  // Scalar of const vector case
      const auto& attr =
          std::any_cast<const ElementwiseAttributes&>(ctx.op_attr);
      const auto* tensor =
          std::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.param);
      const auto* scalar = std::get_if<float>(&attr.param);
      if (!tensor && !scalar) {
        return absl::InvalidArgumentError(
            "Couldn't read scalar of const vector data from the attributes.");
      }

      argument0 = "value_0";
      if (tensor) {
        argument1 = "$const_data[gid.z]$";
        objects.push_back({"const_data", MakeReadonlyObject(tensor->data)});
      } else {
        argument1 = "vec4($const_data$)";
        parameters.push_back({"const_data", *scalar});
      }
      if (attr.runtime_tensor_is_second) {
        argument0 = argument1;
        argument1 = "value_0";
      }
    }

    std::string source;
    switch (operation_type_) {
      case OperationType::DIV: {
        source = "value_0 = $0/$1;";
        break;
      }
      case tflite::gpu::OperationType::FLOOR_DIV:
        source = "value_0 = floor($0 / $1);";
        break;
      case tflite::gpu::OperationType::FLOOR_MOD:
        source = "value_0 = $0 - floor($0 / $1) * $1;";
        break;
      case OperationType::MAXIMUM: {
        source = "value_0 = max($0, $1);";
        break;
      }
      case OperationType::MINIMUM: {
        source = "value_0 = min($0, $1);";
        break;
      }
      case OperationType::SQUARED_DIFF: {
        source = "value_0 = ($0 - $1) * ($0 - $1);";
        break;
      }
      case OperationType::SUB: {
        source = "value_0 = $0 - $1;";
        break;
      }
      case OperationType::POW: {
        source = "value_0 = pow($0, $1);";
        break;
      }
      default:
        return absl::InvalidArgumentError(
            "Incorrect elementwise with scalar operation type.");
    }
    source = absl::Substitute(source, argument0, argument1);
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/std::move(objects),
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/source,
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }

 private:
  OperationType operation_type_;
};

}  // namespace

std::unique_ptr<NodeShader> NewElementwiseNodeShader(
    OperationType operation_type) {
  switch (operation_type) {
    case OperationType::ABS:
    case OperationType::COS:
    case OperationType::COPY:
    case OperationType::ELU:
    case OperationType::EXP:
    case OperationType::FLOOR:
    case OperationType::GELU:
    case OperationType::HARD_SWISH:
    case OperationType::LOG:
    case OperationType::NEG:
    case OperationType::RSQRT:
    case OperationType::SIGMOID:
    case OperationType::SIN:
    case OperationType::SQRT:
    case OperationType::SQUARE:
    case OperationType::TANH:
      return std::make_unique<ElementwiseOneArgument>(operation_type);
    case OperationType::DIV:
    case OperationType::FLOOR_DIV:
    case OperationType::FLOOR_MOD:
    case OperationType::MAXIMUM:
    case OperationType::MINIMUM:
    case OperationType::POW:
    case OperationType::SQUARED_DIFF:
    case OperationType::SUB:
      return std::make_unique<ElementwiseTwoArguments>(operation_type);
    default:
      return nullptr;
  }
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
