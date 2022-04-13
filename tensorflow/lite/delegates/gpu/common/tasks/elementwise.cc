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

#include "tensorflow/lite/delegates/gpu/common/tasks/elementwise.h"

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"

namespace tflite {
namespace gpu {

namespace {
std::string GetOneInputCode(const GpuInfo& gpu_info,
                            const OperationType& op_type,
                            CalculationsPrecision precision,
                            const std::string& input0) {
  std::string result;
  switch (op_type) {
    case OperationType::ABS:
      result = "$0 = fabs($0);\n";
      break;
    case OperationType::COS:
      result = "$0 = cos($0);\n";
      break;
    case OperationType::COPY:
      // No op as inout_value will be copied to dest automatically.
      result = "\n";
      break;
    case OperationType::ELU:
      if (gpu_info.IsApiOpenCl()) {
        result = R"(
$0.x = $0.x < INIT_FLT(0.0f) ? expm1($0.x) : $0.x;
$0.y = $0.y < INIT_FLT(0.0f) ? expm1($0.y) : $0.y;
$0.z = $0.z < INIT_FLT(0.0f) ? expm1($0.z) : $0.z;
$0.w = $0.w < INIT_FLT(0.0f) ? expm1($0.w) : $0.w;)";
      } else {
        result = R"(
$0.x = $0.x < INIT_FLT(0.0f) ? exp($0.x) - INIT_FLT(1.0f) : $0.x;
$0.y = $0.y < INIT_FLT(0.0f) ? exp($0.y) - INIT_FLT(1.0f) : $0.y;
$0.z = $0.z < INIT_FLT(0.0f) ? exp($0.z) - INIT_FLT(1.0f) : $0.z;
$0.w = $0.w < INIT_FLT(0.0f) ? exp($0.w) - INIT_FLT(1.0f) : $0.w;)";
      }
      break;
    case OperationType::EXP:
      result = "$0 = exp($0);\n";
      break;
    case OperationType::FLOOR:
      result = "$0 = floor($0);\n";
      break;
    case OperationType::HARD_SWISH:
      result =
          "$0 *= clamp($0 * INIT_FLT(0.16666667f) + INIT_FLT(0.5f), "
          "INIT_FLT4(0.0f), "
          "INIT_FLT4(1.0f));\n";
      break;
    case OperationType::LOG:
      result = "$0 = log($0);\n";
      break;
    case OperationType::NEG:
      result = "$0 = -($0);\n";
      break;
    case OperationType::RSQRT:
      result = "$0 = rsqrt($0);\n";
      break;
    case OperationType::SIGMOID:
      if (gpu_info.IsApiOpenCl() && precision != CalculationsPrecision::F32) {
        result =
            "$0 = convert_half4(native_recip(1.0f + "
            "native_exp(convert_float4(-$0))));\n";
      } else {
        result = "$0 = INIT_FLT4(1.0f) / (INIT_FLT4(1.0f) + exp(-($0)));\n";
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
  return absl::Substitute(result, input0);
}

std::string GetTwoInputCode(const OperationType& op_type,
                            const std::string& result_var,
                            const std::string& input0,
                            const std::string& input1,
                            bool swap_inputs = false) {
  std::string result;
  switch (op_type) {
    case OperationType::ADD:
      result += "$0 = $1 + $2;\n";
      break;
    case OperationType::DIV:
      result += "$0 = $1 / $2;\n";
      break;
    case OperationType::FLOOR_DIV:
      result = "$0 = floor($1 / $2);\n";
      break;
    case OperationType::FLOOR_MOD:
      result = "$0 = $1 - floor($1 / $2) * $2;\n";
      break;
    case OperationType::MAXIMUM:
      result += "$0 = max($1, $2);\n";
      break;
    case OperationType::MINIMUM:
      result += "$0 = min($1, $2);\n";
      break;
    case OperationType::MUL:
      result += "$0 = $1 * $2;\n";
      break;
    case OperationType::POW:
      result += "$0 = pow($1, $2);\n";
      break;
    case OperationType::SQUARED_DIFF:
      result += "$0 = ($1 - $2) * ($1 - $2);\n";
      break;
    case OperationType::SUB:
      result += "$0 = $1 - $2;\n";
      break;
    // Comparison operators
    case OperationType::LESS:
      result = "$0.x = $1.x < $2.x ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.y = $1.y < $2.y ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.z = $1.z < $2.z ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.w = $1.w < $2.w ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      break;
    case OperationType::LESS_EQUAL:
      result = "$0.x = $1.x <= $2.x ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.y = $1.y <= $2.y ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.z = $1.z <= $2.z ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.w = $1.w <= $2.w ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      break;
    case OperationType::GREATER:
      result = "$0.x = $1.x > $2.x ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.y = $1.y > $2.y ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.z = $1.z > $2.z ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.w = $1.w > $2.w ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      break;
    case OperationType::GREATER_EQUAL:
      result = "$0.x = $1.x >= $2.x ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.y = $1.y >= $2.y ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.z = $1.z >= $2.z ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.w = $1.w >= $2.w ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      break;
    case OperationType::EQUAL:
      result = "$0.x = $1.x == $2.x ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.y = $1.y == $2.y ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.z = $1.z == $2.z ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.w = $1.w == $2.w ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      break;
    case OperationType::NOT_EQUAL:
      result = "$0.x = $1.x != $2.x ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.y = $1.y != $2.y ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.z = $1.z != $2.z ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      result += "$0.w = $1.w != $2.w ? INIT_FLT(1.0f) : INIT_FLT(0.0f);\n";
      break;
    default:
      return "Unknown operation type;\n";
  }
  if (swap_inputs) {
    return absl::Substitute(result, result_var, input1, input0);
  } else {
    return absl::Substitute(result, result_var, input0, input1);
  }
}

// Creates simple two input (first input is runtime tensor and second input is
// scalar argument) operation, for example sub, div, pow, etc.
GPUOperation CreateElementwiseOneRuntimeOneScalar(
    const OperationDef& definition, const OperationType& op_type,
    float scalar_parameter, bool swap_inputs) {
  GPUOperation op(definition);
  op.elementwise_ = true;
  if (definition.precision == CalculationsPrecision::F32) {
    op.args_.AddFloat("scalar", scalar_parameter);
  } else {
    op.args_.AddHalf("scalar", half(scalar_parameter));
  }
  op.code_ = "FLT4 second_val = INIT_FLT4(args.scalar);\n";
  op.code_ += GetTwoInputCode(op_type, "in_out_value", "in_out_value",
                              "second_val", swap_inputs);
  return op;
}

// Creates simple two input(first input is runtime tensor and second input is
// constant linear tensor) operation, for example sub, div and etc.
GPUOperation CreateElementwiseTwoInput(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& constant_tensor,
    bool swap_inputs) {
  const BHWC shape = BHWC(1, 1, 1, constant_tensor.shape.v);
  TensorStorageType storage_type;
  auto status = SelectBestStorageType(
      gpu_info, shape, definition.GetPrimaryStorageType(),
      definition.GetDataType(), Layout::HWC, &storage_type);
  if (!status.ok()) {
    storage_type = TensorStorageType::BUFFER;
  }
  TensorDescriptor desc{definition.GetDataType(), storage_type, Layout::HWC};
  desc.UploadData(constant_tensor);

  GPUOperation result(definition);
  result.elementwise_ = true;
  result.args_.AddObject("second_tensor",
                         absl::make_unique<TensorDescriptor>(std::move(desc)));
  const std::string s_coord = shape.c == 1 ? "0" : "S_COORD";
  result.code_ = absl::StrCat(
      "args.second_tensor::type second_val = args.second_tensor.Read(0, 0, ",
      s_coord, ");\n");
  if (shape.c == 1) {
    result.code_ += "  second_val.y = second_val.x;\n";
    result.code_ += "  second_val.z = second_val.x;\n";
    result.code_ += "  second_val.w = second_val.x;\n";
  }
  result.code_ += GetTwoInputCode(op_type, "in_out_value", "in_out_value",
                                  "second_val", swap_inputs);
  return result;
}

// Creates simple two input(first input is runtime tensor and second input is
// constant HWC tensor) operation, for example sub, div and etc.
GPUOperation CreateElementwiseTwoInput(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& constant_tensor,
    bool swap_inputs) {
  const BHWC shape = BHWC(1, constant_tensor.shape.h, constant_tensor.shape.w,
                          constant_tensor.shape.c);
  TensorStorageType storage_type;
  auto status = SelectBestStorageType(
      gpu_info, shape, definition.GetPrimaryStorageType(),
      definition.GetDataType(), Layout::HWC, &storage_type);
  if (!status.ok()) {
    storage_type = TensorStorageType::BUFFER;
  }
  TensorDescriptor desc{definition.GetDataType(), storage_type, Layout::HWC};
  desc.UploadData(constant_tensor);

  GPUOperation result(definition);
  result.elementwise_ = true;
  result.args_.AddObject("second_tensor",
                         absl::make_unique<TensorDescriptor>(std::move(desc)));
  const std::string x_coord = shape.w == 1 ? "0" : "X_COORD";
  const std::string y_coord = shape.h == 1 ? "0" : "Y_COORD";
  const std::string s_coord = shape.c == 1 ? "0" : "S_COORD";
  result.code_ = absl::StrCat(
      "args.second_tensor::type second_val = args.second_tensor.Read(", x_coord,
      ", ", y_coord, ", ", s_coord, ");\n");
  if (shape.c == 1) {
    result.code_ += "  second_val.y = second_val.x;\n";
    result.code_ += "  second_val.z = second_val.x;\n";
    result.code_ += "  second_val.w = second_val.x;\n";
  }
  result.code_ += GetTwoInputCode(op_type, "in_out_value", "in_out_value",
                                  "second_val", swap_inputs);

  return result;
}

}  // namespace

GPUOperation CreateElementwiseOneInput(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       const OperationType& op_type) {
  GPUOperation op(definition);
  op.elementwise_ = true;
  op.code_ =
      GetOneInputCode(gpu_info, op_type, definition.precision, "in_out_value");
  return op;
}

GPUOperation CreateElementwise(const GpuInfo& gpu_info,
                               const OperationDef& definition,
                               const OperationType& op_type,
                               const ElementwiseAttributes& attr) {
  const float* scalar = absl::get_if<float>(&attr.param);
  const auto* linear_tensor =
      absl::get_if<tflite::gpu::Tensor<Linear, DataType::FLOAT32>>(&attr.param);
  const auto* hwc_tensor =
      absl::get_if<tflite::gpu::Tensor<HWC, DataType::FLOAT32>>(&attr.param);

  if (scalar) {
    return CreateElementwiseOneRuntimeOneScalar(definition, op_type, *scalar,
                                                attr.runtime_tensor_is_second);
  } else if (linear_tensor) {
    return CreateElementwiseTwoInput(gpu_info, definition, op_type,
                                     *linear_tensor,
                                     attr.runtime_tensor_is_second);
  } else if (hwc_tensor) {
    return CreateElementwiseTwoInput(gpu_info, definition, op_type, *hwc_tensor,
                                     attr.runtime_tensor_is_second);
  } else {
    return GPUOperation(definition);
  }
}

GPUOperation CreateElementwiseTwoInput(const OperationDef& definition,
                                       const OperationType& op_type,
                                       const BHWC& shape) {
  GPUOperation op(definition);
  op.elementwise_ = true;
  auto src_desc = definition.src_tensors[1];
  if (definition.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  op.AddSrcTensor("second_tensor", src_desc);
  const std::string x_coord = shape.w == 1 ? "0" : "X_COORD";
  const std::string y_coord = shape.h == 1 ? "0" : "Y_COORD";
  const std::string s_coord = shape.c == 1 ? "0" : "S_COORD";
  op.code_ = absl::StrCat(
      "args.second_tensor::type second_val = args.second_tensor.Read(", x_coord,
      ", ", y_coord, ", ", s_coord, ");\n");
  if (shape.c == 1) {
    op.code_ += "  second_val.y = second_val.x;\n";
    op.code_ += "  second_val.z = second_val.x;\n";
    op.code_ += "  second_val.w = second_val.x;\n";
  }
  op.code_ += GetTwoInputCode(op_type, "in_out_value", "in_out_value",
                              "second_val", false);
  return op;
}

}  // namespace gpu
}  // namespace tflite
