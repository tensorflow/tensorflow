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
      case OperationType::ABS: {
        source = "value_0 = abs(value_0);";
        break;
      }
      case OperationType::SIN: {
        source = "value_0 = sin(value_0);";
        break;
      }
      case OperationType::COS: {
        source = "value_0 = cos(value_0);";
        break;
      }
      case OperationType::LOG: {
        source = R"(
        const float nan = normalize(vec4(0,0,0,0)).x;
        value_0.x = value_0.x > 0.0 ? log(value_0.x) : nan;
        value_0.y = value_0.y > 0.0 ? log(value_0.y) : nan;
        value_0.z = value_0.z > 0.0 ? log(value_0.z) : nan;
        value_0.w = value_0.w > 0.0 ? log(value_0.w) : nan;
    )";
        break;
      }
      case OperationType::SQRT: {
        source = R"(
        const float nan = normalize(vec4(0,0,0,0)).x;
        value_0.x = value_0.x >= 0.0 ? sqrt(value_0.x) : nan;
        value_0.y = value_0.y >= 0.0 ? sqrt(value_0.y) : nan;
        value_0.z = value_0.z >= 0.0 ? sqrt(value_0.z) : nan;
        value_0.w = value_0.w >= 0.0 ? sqrt(value_0.w) : nan;
    )";
        break;
      }
      case OperationType::RSQRT: {
        source = R"(
        const float nan = normalize(vec4(0,0,0,0)).x;
        value_0.x = value_0.x >= 0.0 ? 1.0 / sqrt(value_0.x) : nan;
        value_0.y = value_0.y >= 0.0 ? 1.0 / sqrt(value_0.y) : nan;
        value_0.z = value_0.z >= 0.0 ? 1.0 / sqrt(value_0.z) : nan;
        value_0.w = value_0.w >= 0.0 ? 1.0 / sqrt(value_0.w) : nan;
    )";
        break;
      }
      case OperationType::SQUARE: {
        source = "value_0 = value_0 * value_0;";
        break;
      }
      case OperationType::SIGMOID: {
        source = "value_0 = 1.0 / (1.0 + exp(-1.0 * value_0));";
        break;
      }
      case OperationType::TANH: {
        source = "value_0 = tanh(value_0);";
        break;
      }
      default:
        return InvalidArgumentError("Incorrect elementwise operation type.");
    }
    *generated_code = {
        /*parameters=*/{},
        /*objects=*/{},
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
  static bool IsSupported(const GenerationContext& ctx) {
    auto inputs = ctx.graph->FindInputs(ctx.node->id);

    // Implementation supports concatenation of 2 tensors only.
    if (inputs.size() != 2) {
      std::cerr << "ElementwiseTwoArguments3\n";
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

  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    if (!IsSupported(ctx)) {
      return InvalidArgumentError(
          "This case is not supported by subtract operation");
    }
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
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/source,
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return OkStatus();
  }

 private:
  OperationType operation_type_;
};

}  // namespace

std::unique_ptr<NodeShader> NewElementwiseNodeShader(
    OperationType operation_type) {
  switch (operation_type) {
    case OperationType::ABS:
    case OperationType::SIN:
    case OperationType::COS:
    case OperationType::LOG:
    case OperationType::SQRT:
    case OperationType::RSQRT:
    case OperationType::SQUARE:
    case OperationType::SIGMOID:
    case OperationType::TANH:
      return absl::make_unique<ElementwiseOneArgument>(operation_type);
    case OperationType::SUB:
    case OperationType::DIV:
    case OperationType::POW:
    case OperationType::SQUARED_DIFF:
      return absl::make_unique<ElementwiseTwoArguments>(operation_type);
    default:
      return nullptr;
  }
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
