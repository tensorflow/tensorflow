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

#include "tensorflow/lite/delegates/gpu/cl/kernels/elementwise.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"

namespace tflite {
namespace gpu {
namespace cl {

ElementwiseOneInput::ElementwiseOneInput(ElementwiseOneInput&& operation)
    : ElementwiseOperation(std::move(operation)),
      op_type_(operation.op_type_) {}

ElementwiseOneInput& ElementwiseOneInput::operator=(
    ElementwiseOneInput&& operation) {
  if (this != &operation) {
    std::swap(op_type_, operation.op_type_);
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string ElementwiseOneInput::GetCoreCode(
    const LinkingContext& context) const {
  std::string result;
  switch (op_type_) {
    case OperationType::ABS:
      result = "$0 = fabs($0);\n";
      break;
    case OperationType::COS:
      result = "$0 = cos($0);\n";
      break;
    case OperationType::HARD_SWISH:
      result =
          "$0 *= clamp($0 * (FLT)(0.16666667f) + (FLT)(0.5f), (FLT4)(0.0f), "
          "(FLT4)(1.0f));\n";
      break;
    case OperationType::LOG:
      result = "$0 = log($0);\n";
      break;
    case OperationType::RSQRT:
      result = "$0 = (FLT4)(1.0f) / sqrt($0);\n";
      break;
    case OperationType::SIGMOID:
      if (definition_.precision != CalculationsPrecision::F32) {
        result =
            "$0.x = convert_half(native_recip(1.0f + "
            "native_exp(convert_float(-$0.x))));\n";
        result +=
            "$0.y = convert_half(native_recip(1.0f + "
            "native_exp(convert_float(-$0.y))));\n";
        result +=
            "$0.z = convert_half(native_recip(1.0f + "
            "native_exp(convert_float(-$0.z))));\n";
        result +=
            "$0.w = convert_half(native_recip(1.0f + "
            "native_exp(convert_float(-$0.w))));\n";
      } else {
        result = "$0 = (FLT4)(1.0f) / ((FLT4)(1.0f) + exp(-($0)));\n";
      }
      break;
    case OperationType::SIN:
      result = "$0 = sin($0);\n";
      break;
    case OperationType::SQRT:
      result = "$0 = sqrt($0);\n";
      break;
    case OperationType::SQUARE:
      result = "$0 *= $0;\n";
      break;
    case OperationType::TANH:
      result = "$0 = tanh($0);\n";
      break;
    default:
      return "Unknown operation type;\n";
  }
  return absl::Substitute(result, context.var_name);
}

ElementwiseOneInput CreateElementwiseOneInput(const OperationDef& definition,
                                              const OperationType& op_type) {
  ElementwiseOneInput operation(definition, op_type);
  operation.SetLinkIndex(0);
  return operation;
}

ElementwiseTwoInput::ElementwiseTwoInput(ElementwiseTwoInput&& operation)
    : ElementwiseOperation(std::move(operation)),
      link_index_(operation.link_index_),
      op_type_(operation.op_type_) {}

ElementwiseTwoInput& ElementwiseTwoInput::operator=(
    ElementwiseTwoInput&& operation) {
  if (this != &operation) {
    link_index_ = operation.link_index_;
    op_type_ = operation.op_type_;
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

void ElementwiseTwoInput::SetLinkIndex(int index) { link_index_ = index; }

std::string ElementwiseTwoInput::GetCoreCode(
    const LinkingContext& context) const {
  TensorCodeGenerator src_tensor(absl::StrCat("src_data_", link_index_),
                                 {"src_size.x", "src_size.y", "src_size.z"},
                                 definition_.src_tensors[1]);
  std::string result;
  switch (op_type_) {
    case OperationType::DIV:
      result = "$0 /= $1;\n";
      break;
    case OperationType::POW:
      result = "$0 = pow($0, $1);\n";
      break;
    case OperationType::SQUARED_DIFF:
      result = "$0 -= $1;\n";
      result += "$0 *= $0;\n";
      break;
    case OperationType::SUB:
      result = "$0 -= $1;\n";
      break;
    default:
      return "Unknown operation type;\n";
  }
  return absl::Substitute(
      result, context.var_name,
      src_tensor.Read3D(context.x_coord, context.y_coord, context.z_coord));
}

std::string ElementwiseTwoInput::GetArgsDeclaration() const {
  std::string args;
  TensorCodeGenerator src_tensor(absl::StrCat("src_data_", link_index_),
                                 {"src_size.x", "src_size.y", "src_size.z"},
                                 definition_.src_tensors[1]);
  absl::StrAppend(&args, ",\n", src_tensor.GetDeclaration(AccessType::READ));
  absl::StrAppend(&args, ",\n   int4 src_size_", link_index_);
  return args;
}

Status ElementwiseTwoInput::BindArguments(CLKernel* kernel) {
  RETURN_IF_ERROR(kernel->SetMemoryAuto(src_[1]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel->SetBytesAuto(src_[1]->GetWBatchedHSB()));
  return OkStatus();
}

ElementwiseTwoInput CreateElementwiseTwoInput(const OperationDef& definition,
                                              const OperationType& op_type) {
  ElementwiseTwoInput operation(definition, op_type);
  operation.SetLinkIndex(0);
  return operation;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
