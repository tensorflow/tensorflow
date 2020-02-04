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

#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class ElementwiseOneArgument : public NodeShader {
 public:
  explicit ElementwiseOneArgument(OperationType operation_type)
      : operation_type_(operation_type) {}
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    std::string source;
    switch (operation_type_) {
      case OperationType::ABS:
        source = "value_0 = abs(value_0);";
        break;
      case OperationType::COS:
        source = "value_0 = cos(value_0);";
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
      case OperationType::RSQRT:
        source = R"(
            const float nan = normalize(vec4(0, 0, 0, 0)).x;
            value_0.x = value_0.x >= 0.0 ? 1.0 / sqrt(value_0.x) : nan;
            value_0.y = value_0.y >= 0.0 ? 1.0 / sqrt(value_0.y) : nan;
            value_0.z = value_0.z >= 0.0 ? 1.0 / sqrt(value_0.z) : nan;
            value_0.w = value_0.w >= 0.0 ? 1.0 / sqrt(value_0.w) : nan;
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
        return InvalidArgumentError("Incorrect elementwise operation type.");
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
    return OkStatus();
  }

 private:
  OperationType operation_type_;
};

class ElementwiseTwoArguments : public NodeShader {
 public:
  explicit ElementwiseTwoArguments(OperationType operation_type)
      : operation_type_(operation_type) {}

  bool IsSupportedElemwise(const GenerationContext& ctx) const {
    auto inputs = ctx.graph->FindInputs(ctx.node->id);

    // Implementation supports concatenation of 2 tensors only.
    if (inputs.size() != 2) {
      return false;
    }

    auto shape0 = inputs[0]->tensor.shape;
    auto shape1 = inputs[1]->tensor.shape;

    // Shapes must be the same
    if (shape0 != shape1) {
      return false;
    }
    return true;
  }

  Status ImplementElementwise(const GenerationContext& ctx,
                              GeneratedCode* generated_code) const {
    std::string source;
    switch (operation_type_) {
      case OperationType::SUB: {
        source = "value_0 -= value_1;";
        break;
      }
      case OperationType::DIV: {
        source = "value_0 /= value_1;";
        break;
      }
      case OperationType::POW: {
        // From documentation :
        // The result is undefined if x<0 or if x=0 and y≤0.
        source = "value_0 = pow(value_0, value_1);";
        break;
      }
      case OperationType::SQUARED_DIFF: {
        source = "value_0 = (value_0 - value_1) * (value_0 - value_1);";
        break;
      }

      default:
        return InvalidArgumentError(
            "Incorrect elementwise with two arguments operation type.");
    }
    *generated_code = {
        /*parameters=*/{},
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/source,
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return OkStatus();
  }

  bool IsSupportedBroadcast(const GenerationContext& ctx) const {
    auto inputs = ctx.graph->FindInputs(ctx.node->id);
    auto outputs = ctx.graph->FindOutputs(ctx.node->id);

    if (inputs.size() != 2) {
      return false;
    }
    if (inputs[1]->tensor.shape.h != 1 || inputs[1]->tensor.shape.w != 1 ||
        inputs[0]->tensor.shape.c != inputs[1]->tensor.shape.c) {
      return false;
    }
    return true;
  }

  Status ImplementElementwiseBroadcast(const GenerationContext& ctx,
                                       GeneratedCode* generated_code) const {
    std::string source;
    switch (operation_type_) {
      case OperationType::SQUARED_DIFF: {
        source = R"(
        vec4 diff = $input_data_0[gid.x, gid.y, gid.z]$ -
                    $input_data_1[0, 0, gid.z]$;
        value_0 = diff * diff;
        )";
        break;
      }

      default:
        return InvalidArgumentError(
            "Incorrect elementwise with two arguments operation type.");
    }
    *generated_code = {
        /*parameters=*/{},
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/source,
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return OkStatus();
  }

  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    if (IsSupportedElemwise(ctx)) {
      return ImplementElementwise(ctx, generated_code);
    }
    if (IsSupportedBroadcast(ctx)) {
      return ImplementElementwiseBroadcast(ctx, generated_code);
    }
    return InvalidArgumentError(
        "This case is not supported by subtract operation");
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
    case OperationType::LOG:
    case OperationType::HARD_SWISH:
    case OperationType::RSQRT:
    case OperationType::SIGMOID:
    case OperationType::SIN:
    case OperationType::SQRT:
    case OperationType::SQUARE:
    case OperationType::TANH:
      return absl::make_unique<ElementwiseOneArgument>(operation_type);
    case OperationType::DIV:
    case OperationType::POW:
    case OperationType::SQUARED_DIFF:
    case OperationType::SUB:
      return absl::make_unique<ElementwiseTwoArguments>(operation_type);
    default:
      return nullptr;
  }
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
