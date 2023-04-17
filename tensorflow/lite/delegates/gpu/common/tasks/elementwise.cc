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
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"

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
      result = "$0 = fabs($1);";
      break;
    case OperationType::CEIL:
      result = "$0 = ceil($1);";
      break;
    case OperationType::COS:
      if (use_native_opencl_functions) {
        result = "$0 = convert_half4(native_cos(convert_float4($1)));";
      } else {
        result = "$0 = cos($1);";
      }
      break;
    case OperationType::COPY:
      result = "$0 = $1;";
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
        result = "$0 = convert_half4(native_exp(convert_float4($1)));";
      } else {
        result = "$0 = exp($1);";
      }
      break;
    case OperationType::FLOOR:
      result = "$0 = floor($1);";
      break;
    case OperationType::HARD_SWISH:
      result =
          "$0 = $1 * clamp($1 * INIT_FLT(0.16666667f) + INIT_FLT(0.5f), "
          "INIT_FLT4(0.0f), "
          "INIT_FLT4(1.0f));";
      break;
    case OperationType::LOG:
      if (use_native_opencl_functions) {
        result = "$0 = convert_half4(native_log(convert_float4($1)));";
      } else {
        result = "$0 = log($1);";
      }
      break;
    case OperationType::NEG:
      result = "$0 = -($1);";
      break;
    case OperationType::RSQRT:
      if (use_native_opencl_functions) {
        result = "$0 = convert_half4(native_rsqrt(convert_float4($1)));";
      } else {
        result = "$0 = rsqrt($1);";
      }
      break;
    case OperationType::SIGMOID:
      if (use_native_opencl_functions) {
        result =
            "$0 = convert_half4(native_recip(1.0f + "
            "native_exp(convert_float4(-$1))));";
      } else {
        result = "$0 = INIT_FLT4(1.0f) / (INIT_FLT4(1.0f) + exp(-($1)));";
      }
      break;
    case OperationType::SIGN:
      result = "$0 = sign($1);";
      break;
    case OperationType::SIN:
      if (use_native_opencl_functions) {
        result = "$0 = convert_half4(native_sin(convert_float4($1)));";
      } else {
        result = "$0 = sin($1);";
      }
      break;
    case OperationType::SQRT:
      if (use_native_opencl_functions) {
        result = "$0 = convert_half4(native_sqrt(convert_float4($1)));";
      } else {
        result = "$0 = sqrt($1);";
      }
      break;
    case OperationType::SQUARE:
      result = "$0 = $1 * $1;";
      break;
    case OperationType::TANH:
      if (use_native_opencl_functions) {
        result =
            "FLT4 exp_val = convert_half4(native_exp(2.0f * "
            "convert_float4($1)));\n";
        result +=
            "$0 = ((exp_val - INIT_FLT4(1.0f)) / (exp_val + "
            "INIT_FLT4(1.0f)));";
      } else {
        result = "$0 = tanh($1);";
      }
      break;
    default:
      return "Unknown operation type;";
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
      result += "$0 = $1 + $2;";
      break;
    case OperationType::DIV:
      result += "$0 = $1 / $2;";
      break;
    case OperationType::FLOOR_DIV:
      result = "$0 = floor($1 / $2);";
      break;
    case OperationType::FLOOR_MOD:
      result = "$0 = $1 - floor($1 / $2) * $2;";
      break;
    case OperationType::MAXIMUM:
      result += "$0 = max($1, $2);";
      break;
    case OperationType::MINIMUM:
      result += "$0 = min($1, $2);";
      break;
    case OperationType::MUL:
      result += "$0 = $1 * $2;";
      break;
    case OperationType::POW:
      result += "$0 = pow($1, $2);";
      break;
    case OperationType::SQUARED_DIFF:
      result += "$0 = ($1 - $2) * ($1 - $2);";
      break;
    case OperationType::SUB:
      result += "$0 = $1 - $2;";
      break;
    // Comparison operators
    case OperationType::LESS:
      result = "$0.x = $1.x < $2.x;\n";
      result += "$0.y = $1.y < $2.y;\n";
      result += "$0.z = $1.z < $2.z;\n";
      result += "$0.w = $1.w < $2.w;";
      break;
    case OperationType::LESS_EQUAL:
      result = "$0.x = $1.x <= $2.x;\n";
      result += "$0.y = $1.y <= $2.y;\n";
      result += "$0.z = $1.z <= $2.z;\n";
      result += "$0.w = $1.w <= $2.w;";
      break;
    case OperationType::GREATER:
      result = "$0.x = $1.x > $2.x;\n";
      result += "$0.y = $1.y > $2.y;\n";
      result += "$0.z = $1.z > $2.z;\n";
      result += "$0.w = $1.w > $2.w;";
      break;
    case OperationType::GREATER_EQUAL:
      result = "$0.x = $1.x >= $2.x;\n";
      result += "$0.y = $1.y >= $2.y;\n";
      result += "$0.z = $1.z >= $2.z;\n";
      result += "$0.w = $1.w >= $2.w;";
      break;
    case OperationType::EQUAL:
      result = "$0.x = $1.x == $2.x;\n";
      result += "$0.y = $1.y == $2.y;\n";
      result += "$0.z = $1.z == $2.z;\n";
      result += "$0.w = $1.w == $2.w;";
      break;
    case OperationType::NOT_EQUAL:
      result = "$0.x = $1.x != $2.x;\n";
      result += "$0.y = $1.y != $2.y;\n";
      result += "$0.z = $1.z != $2.z;\n";
      result += "$0.w = $1.w != $2.w;";
      break;
    case OperationType::LOGICAL_AND:
      result = "$0.x = ($1.x != 0) && ($2.x != 0);\n";
      result += "$0.y = ($1.y != 0) && ($2.y != 0);\n";
      result += "$0.z = ($1.z != 0) && ($2.z != 0);\n";
      result += "$0.w = ($1.w != 0) && ($2.w != 0);";
      break;
    default:
      return "Unknown operation type;";
  }
  if (swap_inputs) {
    return absl::Substitute(result, result_var, input1, input0);
  } else {
    return absl::Substitute(result, result_var, input0, input1);
  }
}

// Creates simple two input (first input is runtime tensor and second input is
// scalar argument) operation, for example sub, div, pow, etc.
template <typename T>
ElementwiseDescriptor CreateElementwiseOneRuntimeOneScalar(
    const OperationDef& definition, const OperationType& op_type,
    T scalar_parameter, bool swap_inputs) {
  ElementwiseDescriptor op_desc;
  if (std::is_same<T, int32_t>::value) {
    op_desc.args.AddInt("scalar", scalar_parameter);
    op_desc.code =
        "int4 second_val = CONVERT_TO_INT4(INIT_FLT4(args.scalar));\n";
    op_desc.code += GetTwoInputCode(op_type, "out_value", "in_value",
                                    "second_val", swap_inputs);
    return op_desc;
  }
  if (definition.precision == CalculationsPrecision::F32) {
    op_desc.args.AddFloat("scalar", scalar_parameter);
  } else {
    op_desc.args.AddHalf("scalar", half(scalar_parameter));
  }
  op_desc.code = "FLT4 second_val = INIT_FLT4(args.scalar);\n";
  op_desc.code += GetTwoInputCode(op_type, "out_value", "in_value",
                                  "second_val", swap_inputs);
  return op_desc;
}

// Creates simple two input(first input is runtime tensor and second input is
// constant linear tensor) operation, for example sub, div and etc.
template <DataType DataTypeT>
ElementwiseDescriptor CreateElementwiseTwoInput(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const tflite::gpu::Tensor<Linear, DataTypeT>& constant_tensor,
    bool swap_inputs) {
  TensorDescriptor const_tensor_desc = CreateConstantLinearTensorDescriptor(
      gpu_info, definition.src_tensors[0].GetDataType(), constant_tensor);
  ElementwiseDescriptor op_desc;
  op_desc.args.AddObject("second_tensor", std::make_unique<TensorDescriptor>(
                                              std::move(const_tensor_desc)));
  const std::string s_coord = constant_tensor.shape.v == 1 ? "0" : "S_COORD";
  op_desc.code = absl::StrCat(
      "args.second_tensor::type second_val = args.second_tensor.Read(", s_coord,
      ");\n");
  if (constant_tensor.shape.v == 1) {
    op_desc.code += "  second_val.y = second_val.x;\n";
    op_desc.code += "  second_val.z = second_val.x;\n";
    op_desc.code += "  second_val.w = second_val.x;\n";
  }
  op_desc.code += GetTwoInputCode(op_type, "out_value", "in_value",
                                  "second_val", swap_inputs);
  return op_desc;
}

// Creates simple two input(first input is runtime tensor and second input is
// constant HWC tensor) operation, for example sub, div and etc.
template <DataType DataTypeT>
ElementwiseDescriptor CreateElementwiseTwoInput(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const tflite::gpu::Tensor<HWC, DataTypeT>& constant_tensor,
    bool swap_inputs) {
  const BHWC shape = BHWC(1, constant_tensor.shape.h, constant_tensor.shape.w,
                          constant_tensor.shape.c);
  TensorDescriptor const_tensor_desc =
      TensorDescriptor(definition.src_tensors[0].GetDataType(),
                       definition.src_tensors[0].GetStorageType(), Layout::HWC);
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

  return op_desc;
}

template <DataType DataTypeT, typename T>
ElementwiseDescriptor CreateElementwiseDesc(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const ElementwiseAttributesBase<DataTypeT, T>& attr) {
  const T* scalar = std::get_if<T>(&attr.param);
  const auto* linear_tensor =
      std::get_if<tflite::gpu::Tensor<Linear, DataTypeT>>(&attr.param);
  const auto* hwc_tensor =
      std::get_if<tflite::gpu::Tensor<HWC, DataTypeT>>(&attr.param);

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
    return ElementwiseDescriptor();
  }
}

}  // namespace

ElementwiseDescriptor CreateElementwiseOneInput(const GpuInfo& gpu_info,
                                                CalculationsPrecision precision,
                                                const OperationType& op_type) {
  ElementwiseDescriptor op_desc;
  op_desc.code =
      GetOneInputCode(gpu_info, op_type, precision, "in_value", "out_value");
  return op_desc;
}

GPUOperation CreateElementwiseOneInput(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       const OperationType& op_type) {
  return CreateGpuOperation(
      definition,
      CreateElementwiseOneInput(gpu_info, definition.precision, op_type));
}

template <DataType DataTypeT, typename T>
GPUOperation CreateElementwise(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const ElementwiseAttributesBase<DataTypeT, T>& attr) {
  return CreateGpuOperation(
      definition, CreateElementwiseDesc(gpu_info, definition, op_type, attr));
}

GPUOperation CreateElementwiseTwoInput(const OperationDef& definition,
                                       const OperationType& op_type,
                                       const BHWC& shape) {
  ElementwiseDescriptor op_desc;
  op_desc.code =
      GetTwoInputCode(op_type, "out_value", "in_value", "in2_value", false);
  return CreateGpuOperation(definition, std::move(op_desc), shape);
}

namespace {
std::string GetKernelBodyCode(const TensorDescriptor& dst_desc) {
  std::string c;
  c += "MAIN_FUNCTION($$0) {\n";
  if (dst_desc.HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) return; \n";
  c += "  args.dst_tensor::type result;\n";
  c += "  $0\n";
  c += "  args.dst_tensor.Write(result, X, Y, S);\n";
  c += "} \n";
  return c;
}
std::string GetReadBroadcastedValueCode(const BHWC& src_shape,
                                        const TensorDescriptor& src_desc,
                                        const BHWC& dst_shape) {
  const std::string x_coord = src_shape.w != dst_shape.w ? "0" : "X";
  const std::string y_coord = src_shape.h != dst_shape.h ? "0" : "Y";
  const std::string s_coord = src_shape.c != dst_shape.c ? "0" : "S";
  std::string coords = absl::StrCat(x_coord, ", ", y_coord, ", ", s_coord);
  if (src_desc.HasAxis(Axis::BATCH)) {
    const std::string b_coord = src_shape.b != dst_shape.b ? "0" : "B";
    coords += ", " + b_coord;
  }
  std::string read_value_code =
      absl::StrCat("args.$0::type $1 = args.$0.Read(", coords, ");\n");
  if (src_shape.c != dst_shape.c) {
    read_value_code += "  $1.y = $1.x;\n";
    read_value_code += "  $1.z = $1.x;\n";
    read_value_code += "  $1.w = $1.x;\n";
  }
  return read_value_code;
}
}  // namespace

GPUOperation CreateElementwiseOneInputWithBroadcast(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type, const BHWC& input_shape,
    const BHWC& output_shape) {
  GPUOperation op(definition);
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  std::string c;
  c += "  " + absl::Substitute(
                  GetReadBroadcastedValueCode(
                      input_shape, definition.src_tensors[0], output_shape),
                  "src_tensor", "first_value");
  c += "  " + GetOneInputCode(gpu_info, op_type, definition.precision,
                              "first_value", "result");
  op.code_ = absl::Substitute(GetKernelBodyCode(definition.dst_tensors[0]), c);
  return op;
}

template <DataType DataTypeT, typename T>
GPUOperation CreateElementwiseWithBroadcast(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const ElementwiseAttributesBase<DataTypeT, T>& attr,
    const BHWC& input_shape, const BHWC& output_shape) {
  ElementwiseDescriptor op_desc =
      CreateElementwiseDesc(gpu_info, definition, op_type, attr);

  GPUOperation op(definition);
  op.args_ = std::move(op_desc.args);
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  std::string c;
  c += "  " + absl::Substitute(
                  GetReadBroadcastedValueCode(
                      input_shape, definition.src_tensors[0], output_shape),
                  "src_tensor", "first_value");
  c += "  " + absl::StrReplaceAll(op_desc.code, {{"in_value", "first_value"},
                                                 {"out_value", "result"},
                                                 {"X_COORD", "X"},
                                                 {"Y_COORD", "Y"},
                                                 {"S_COORD", "S"},
                                                 {"B_COORD", "B"}});
  op.code_ = absl::Substitute(GetKernelBodyCode(definition.dst_tensors[0]), c);
  return op;
}

GPUOperation CreateElementwiseTwoInputWithBroadcast(
    const OperationDef& definition, const OperationType& op_type,
    const BHWC& first_input_shape, const BHWC& second_input_shape,
    const BHWC& output_shape) {
  GPUOperation op(definition);
  op.AddSrcTensor("src0_tensor", definition.src_tensors[0]);
  op.AddSrcTensor("src1_tensor", definition.src_tensors[1]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  std::string c;
  c += "  " + absl::Substitute(GetReadBroadcastedValueCode(
                                   first_input_shape, definition.src_tensors[0],
                                   output_shape),
                               "src0_tensor", "first_value");
  c += "  " + absl::Substitute(GetReadBroadcastedValueCode(
                                   second_input_shape,
                                   definition.src_tensors[1], output_shape),
                               "src1_tensor", "second_value");
  c += "  " +
       GetTwoInputCode(op_type, "result", "first_value", "second_value", false);
  op.code_ = absl::Substitute(GetKernelBodyCode(definition.dst_tensors[0]), c);
  return op;
}

template GPUOperation CreateElementwise(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const ElementwiseAttributesBase<DataType::BOOL, bool>& attr);

template GPUOperation CreateElementwise(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const ElementwiseAttributesBase<DataType::FLOAT32, float>& attr);

template GPUOperation CreateElementwise(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const ElementwiseAttributesBase<DataType::INT32, int32_t>& attr);

template GPUOperation CreateElementwiseWithBroadcast(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const ElementwiseAttributesBase<DataType::BOOL, bool>& attr,
    const BHWC& input_shape, const BHWC& output_shape);

template GPUOperation CreateElementwiseWithBroadcast(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const ElementwiseAttributesBase<DataType::FLOAT32, float>& attr,
    const BHWC& input_shape, const BHWC& output_shape);

template GPUOperation CreateElementwiseWithBroadcast(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const ElementwiseAttributesBase<DataType::INT32, int>& attr,
    const BHWC& input_shape, const BHWC& output_shape);

}  // namespace gpu
}  // namespace tflite
