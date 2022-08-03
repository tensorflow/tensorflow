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

#include <memory>
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
                            const std::string& input_value,
                            const std::string& output_value) {
  const bool use_native_opencl_functions =
      gpu_info.IsApiOpenCl() && precision != CalculationsPrecision::F32 &&
      gpu_info.IsAdreno();
  std::string result;
  switch (op_type) {
    case OperationType::ABS:
      result = "$0 = fabs($1);\n";
      break;
    case OperationType::COS:
      if (use_native_opencl_functions) {
        result = "$0 = convert_half4(native_cos(convert_float4($1)));\n";
      } else {
        result = "$0 = cos($1);\n";
      }
      break;
    case OperationType::COPY:
      result = "$0 = $1;\n";
      break;
    case OperationType::ELU:
      if (gpu_info.IsApiOpenCl()) {
        result = R"(
$0.x = $1.x < INIT_FLT(0.0f) ? expm1($1.x) : $1.x;
$0.y = $1.y < INIT_FLT(0.0f) ? expm1($1.y) : $1.y;
$0.z = $1.z < INIT_FLT(0.0f) ? expm1($1.z) : $1.z;
$0.w = $1.w < INIT_FLT(0.0f) ? expm1($1.w) : $1.w;)";
      } else {
        result = R"(
$0.x = $1.x < INIT_FLT(0.0f) ? exp($1.x) - INIT_FLT(1.0f) : $1.x;
$0.y = $1.y < INIT_FLT(0.0f) ? exp($1.y) - INIT_FLT(1.0f) : $1.y;
$0.z = $1.z < INIT_FLT(0.0f) ? exp($1.z) - INIT_FLT(1.0f) : $1.z;
$0.w = $1.w < INIT_FLT(0.0f) ? exp($1.w) - INIT_FLT(1.0f) : $1.w;)";
      }
      break;
    case OperationType::EXP:
      if (use_native_opencl_functions) {
        result = "$0 = convert_half4(native_exp(convert_float4($1)));\n";
      } else {
        result = "$0 = exp($1);\n";
      }
      break;
    case OperationType::FLOOR:
      result = "$0 = floor($1);\n";
      break;
    case OperationType::HARD_SWISH:
      result =
          "$0 = $1 * clamp($1 * INIT_FLT(0.16666667f) + INIT_FLT(0.5f), "
          "INIT_FLT4(0.0f), "
          "INIT_FLT4(1.0f));\n";
      break;
    case OperationType::LOG:
      if (use_native_opencl_functions) {
        result = "$0 = convert_half4(native_log(convert_float4($1)));\n";
      } else {
        result = "$0 = log($1);\n";
      }
      break;
    case OperationType::NEG:
      result = "$0 = -($1);\n";
      break;
    case OperationType::RSQRT:
      if (use_native_opencl_functions) {
        result = "$0 = convert_half4(native_rsqrt(convert_float4($1)));\n";
      } else {
        result = "$0 = rsqrt($1);\n";
      }
      break;
    case OperationType::SIGMOID:
      if (use_native_opencl_functions) {
        result =
            "$0 = convert_half4(native_recip(1.0f + "
            "native_exp(convert_float4(-$1))));\n";
      } else {
        result = "$0 = INIT_FLT4(1.0f) / (INIT_FLT4(1.0f) + exp(-($1)));\n";
      }
      break;
    case OperationType::SIN:
      if (use_native_opencl_functions) {
        result = "$0 = convert_half4(native_sin(convert_float4($1)));\n";
      } else {
        result = "$0 = sin($1);\n";
      }
      break;
    case OperationType::SQRT:
      if (use_native_opencl_functions) {
        result = "$0 = convert_half4(native_sqrt(convert_float4($1)));\n";
      } else {
        result = "$0 = sqrt($1);\n";
      }
      break;
    case OperationType::SQUARE:
      result = "$0 = $1 * $1;\n";
      break;
    case OperationType::TANH:
      if (use_native_opencl_functions) {
        result =
            "  FLT4 exp_val = convert_half4(native_exp(2.0f * "
            "convert_float4($1)));\n";
        result +=
            "$0 = ((exp_val - INIT_FLT4(1.0f)) / (exp_val + "
            "INIT_FLT4(1.0f)));\n";
      } else {
        result = "$0 = tanh($1);\n";
      }
      break;
    default:
      return "Unknown operation type;\n";
  }
  return absl::Substitute(result, output_value, input_value);
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
      result = "$0.x = $1.x < $2.x;\n";
      result += "$0.y = $1.y < $2.y;\n";
      result += "$0.z = $1.z < $2.z;\n";
      result += "$0.w = $1.w < $2.w;\n";
      break;
    case OperationType::LESS_EQUAL:
      result = "$0.x = $1.x <= $2.x;\n";
      result += "$0.y = $1.y <= $2.y;\n";
      result += "$0.z = $1.z <= $2.z;\n";
      result += "$0.w = $1.w <= $2.w;\n";
      break;
    case OperationType::GREATER:
      result = "$0.x = $1.x > $2.x;\n";
      result += "$0.y = $1.y > $2.y;\n";
      result += "$0.z = $1.z > $2.z;\n";
      result += "$0.w = $1.w > $2.w;\n";
      break;
    case OperationType::GREATER_EQUAL:
      result = "$0.x = $1.x >= $2.x;\n";
      result += "$0.y = $1.y >= $2.y;\n";
      result += "$0.z = $1.z >= $2.z;\n";
      result += "$0.w = $1.w >= $2.w;\n";
      break;
    case OperationType::EQUAL:
      result = "$0.x = $1.x == $2.x;\n";
      result += "$0.y = $1.y == $2.y;\n";
      result += "$0.z = $1.z == $2.z;\n";
      result += "$0.w = $1.w == $2.w;\n";
      break;
    case OperationType::NOT_EQUAL:
      result = "$0.x = $1.x != $2.x;\n";
      result += "$0.y = $1.y != $2.y;\n";
      result += "$0.z = $1.z != $2.z;\n";
      result += "$0.w = $1.w != $2.w;\n";
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
  ElementwiseDescriptor op_desc;
  if (definition.precision == CalculationsPrecision::F32) {
    op_desc.args.AddFloat("scalar", scalar_parameter);
  } else {
    op_desc.args.AddHalf("scalar", half(scalar_parameter));
  }
  op_desc.code = "FLT4 second_val = INIT_FLT4(args.scalar);\n";
  op_desc.code += GetTwoInputCode(op_type, "out_value", "in_value",
                                  "second_val", swap_inputs);
  return CreateGpuOperation(definition, std::move(op_desc));
}

// Creates simple two input(first input is runtime tensor and second input is
// constant linear tensor) operation, for example sub, div and etc.
GPUOperation CreateElementwiseTwoInput(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& constant_tensor,
    bool swap_inputs) {
  const BHWC shape = BHWC(1, 1, 1, constant_tensor.shape.v);
  TensorDescriptor const_tensor_desc = definition.src_tensors[0];
  auto status = const_tensor_desc.UpdateToSupportedStorageType(gpu_info, shape);
  const_tensor_desc.UploadData(constant_tensor);

  ElementwiseDescriptor op_desc;
  op_desc.args.AddObject("second_tensor", std::make_unique<TensorDescriptor>(
                                              std::move(const_tensor_desc)));
  const std::string s_coord = shape.c == 1 ? "0" : "S_COORD";
  op_desc.code = absl::StrCat(
      "args.second_tensor::type second_val = args.second_tensor.Read(0, 0, ",
      s_coord, ");\n");
  if (shape.c == 1) {
    op_desc.code += "  second_val.y = second_val.x;\n";
    op_desc.code += "  second_val.z = second_val.x;\n";
    op_desc.code += "  second_val.w = second_val.x;\n";
  }
  op_desc.code += GetTwoInputCode(op_type, "out_value", "in_value",
                                  "second_val", swap_inputs);
  return CreateGpuOperation(definition, std::move(op_desc));
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
  TensorDescriptor const_tensor_desc = definition.src_tensors[0];
  auto status = const_tensor_desc.UpdateToSupportedStorageType(gpu_info, shape);
  const_tensor_desc.UploadData(constant_tensor);

  ElementwiseDescriptor op_desc;
  op_desc.args.AddObject("second_tensor", std::make_unique<TensorDescriptor>(
                                              std::move(const_tensor_desc)));
  const std::string x_coord = shape.w == 1 ? "0" : "X_COORD";
  const std::string y_coord = shape.h == 1 ? "0" : "Y_COORD";
  const std::string s_coord = shape.c == 1 ? "0" : "S_COORD";
  op_desc.code = absl::StrCat(
      "args.second_tensor::type second_val = args.second_tensor.Read(", x_coord,
      ", ", y_coord, ", ", s_coord, ");\n");
  if (shape.c == 1) {
    op_desc.code += "  second_val.y = second_val.x;\n";
    op_desc.code += "  second_val.z = second_val.x;\n";
    op_desc.code += "  second_val.w = second_val.x;\n";
  }
  op_desc.code += GetTwoInputCode(op_type, "out_value", "in_value",
                                  "second_val", swap_inputs);

  return CreateGpuOperation(definition, std::move(op_desc));
}

}  // namespace

GPUOperation CreateElementwiseOneInput(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       const OperationType& op_type) {
  ElementwiseDescriptor op_desc;
  op_desc.code = GetOneInputCode(gpu_info, op_type, definition.precision,
                                 "in_value", "out_value");
  return CreateGpuOperation(definition, std::move(op_desc));
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
  ElementwiseDescriptor op_desc;
  const std::string x_coord = shape.w == 1 ? "0" : "X_COORD";
  const std::string y_coord = shape.h == 1 ? "0" : "Y_COORD";
  const std::string s_coord = shape.c == 1 ? "0" : "S_COORD";
  std::string coords = absl::StrCat(x_coord, ", ", y_coord, ", ", s_coord);
  if (definition.src_tensors[1].HasAxis(Axis::BATCH)) {
    const std::string b_coord = shape.b == 1 ? "0" : "B_COORD";
    coords += ", " + b_coord;
  }
  op_desc.code = absl::StrCat(
      "args.src_tensor_1::type second_val = args.src_tensor_1.Read(", coords,
      ");\n");
  if (shape.c == 1) {
    op_desc.code += "  second_val.y = second_val.x;\n";
    op_desc.code += "  second_val.z = second_val.x;\n";
    op_desc.code += "  second_val.w = second_val.x;\n";
  }
  op_desc.code +=
      GetTwoInputCode(op_type, "out_value", "in_value", "second_val", false);
  return CreateGpuOperation(definition, std::move(op_desc));
}

}  // namespace gpu
}  // namespace tflite
