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

#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_buffer_1x1.h"

#include <array>
#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

// element_size must be 1, 2 or 4
// 1 - is FLT4
// 2 - is FLT8
// 4 - is FLT16
// This function generates code for arithmetic part of convolution
std::string GetComputationPart(const int3& block_size, int element_size,
                               CalculationsPrecision precision) {
  const std::string hexes[16] = {"0", "1", "2", "3", "4", "5", "6", "7",
                                 "8", "9", "a", "b", "c", "d", "e", "f"};
  std::string c;
  for (int z = 0; z < block_size.z; ++z) {
    const std::string z_s = std::to_string(z);
    c += "    FLT16 W" + z_s + " = weights_cache[" + z_s + "];\n";
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x; ++x) {
        std::string s_index = std::to_string(y * block_size.x + x);
        for (int e = 0; e < element_size; ++e) {
          std::string r_index =
              z_s + std::to_string(y) + std::to_string(x * element_size + e);
          const std::string f0 = "W" + z_s + ".s0123";
          const std::string f1 = "W" + z_s + ".s4567";
          const std::string f2 = "W" + z_s + ".s89ab";
          const std::string f3 = "W" + z_s + ".scdef";
          switch (precision) {
            case CalculationsPrecision::F32:
            case CalculationsPrecision::F16:
              c += "    r" + r_index + " += " + f0 + " * s" + s_index + ".s" +
                   hexes[e * 4 + 0] + ";\n";
              c += "    r" + r_index + " += " + f1 + " * s" + s_index + ".s" +
                   hexes[e * 4 + 1] + ";\n";
              c += "    r" + r_index + " += " + f2 + " * s" + s_index + ".s" +
                   hexes[e * 4 + 2] + ";\n";
              c += "    r" + r_index + " += " + f3 + " * s" + s_index + ".s" +
                   hexes[e * 4 + 3] + ";\n";
              break;
            case CalculationsPrecision::F32_F16:
              c += "    r" + r_index + " += convert_float4(" + f0 + " * s" +
                   s_index + ".s" + hexes[e * 4 + 0] + " + " + f1 + " * s" +
                   s_index + ".s" + hexes[e * 4 + 1] + " + " + f2 + " * s" +
                   s_index + ".s" + hexes[e * 4 + 2] + " + " + f3 + " * s" +
                   s_index + ".s" + hexes[e * 4 + 3] + ");\n";
              break;
          }
        }
      }
    }
  }
  return c;
}

std::string GenerateConvBuffer1x1(
    const OperationDef& op_def, const ConvBuffer1x1::ConvParams& conv_params,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  std::string c = GetCommonDefines(op_def.precision);
  TensorCodeGenerator dst_tensor(
      "dst_data", WHSPoint{"dst_size.x", "dst_size.y", "dst_size.z"},
      op_def.dst_tensors[0]);

  switch (op_def.precision) {
    case CalculationsPrecision::F32:
      c += "#define FLT8 float8\n";
      c += "#define FLT16 float16\n";
      break;
    case CalculationsPrecision::F32_F16:
    case CalculationsPrecision::F16:
      c += "#define FLT8 half8\n";
      c += "#define FLT16 half16\n";
      break;
  }

  const int3 block_size = conv_params.block_size;
  const int element_size = conv_params.element_size / 4;

  c += "__kernel void main_function(\n";
  c += "    __global FLT" + std::to_string(element_size * 4) + "* src_data,\n";
  c += "    __global FLT16* filters_buffer,   \n";
  c += "    __global FLT4* biases             \n";
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,                   \n";
  c += "    int4 dst_size                    \n";
  c += ") {\n";
  c += "  int X = get_global_id(0) * " +
       std::to_string(block_size.x * element_size) + ";\n";
  c += "  int X_SRC = get_global_id(0) * " + std::to_string(block_size.x) +
       ";\n";
  c += "  int Y = get_global_id(1) * " + std::to_string(block_size.y) + ";\n";
  c += "  int Z = get_global_id(2) * " + std::to_string(block_size.z) + ";\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;\n";
  if (conv_params.different_weights_for_height) {
    c += "  __global FLT16* weights_cache = filters_buffer + (Z * src_size.y + "
         "Y * " +
         std::to_string(block_size.z) +
         ") * "
         "src_size.z;\n";
  } else {
    c += "  __global FLT16* weights_cache = filters_buffer + Z * src_size.z;\n";
  }
  for (int z = 0; z < block_size.z; ++z) {
    const std::string z_s = std::to_string(z);
    c += "  ACCUM_FLT4 bias_val_" + z_s + " = TO_ACCUM_TYPE(biases[Z + " + z_s +
         "]);\n";
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x * element_size; ++x) {
        c += "  ACCUM_FLT4 r" + z_s + std::to_string(y) + std::to_string(x) +
             " = bias_val_" + z_s + ";\n";
      }
    }
  }
  for (int x = 0; x < block_size.x; ++x) {
    std::string x_s = std::to_string(x);
    c += "  int xc" + x_s + " = min(X_SRC + " + std::to_string(x) +
         ", src_size.x - 1);\n";
  }
  for (int y = 0; y < block_size.y; ++y) {
    std::string y_s = std::to_string(y);
    c += "  int yc" + y_s + " = min(Y + " + y_s + ", src_size.y - 1);\n";
  }
  for (int y = 0; y < block_size.y; ++y) {
    std::string y_s = std::to_string(y);
    for (int x = 0; x < block_size.x; ++x) {
      std::string x_s = std::to_string(x);
      std::string i_s = std::to_string(y * block_size.x + x);
      c += "  int src_addr_" + i_s + " = (yc" + y_s + ") * src_size.x + (xc" +
           x_s + ");\n";
    }
  }
  c += "  for (int s = 0; s < src_size.z; ++s) {\n";
  for (int y = 0; y < block_size.y; ++y) {
    std::string y_s = std::to_string(y);
    for (int x = 0; x < block_size.x; ++x) {
      std::string x_s = std::to_string(x);
      std::string i_s = std::to_string(y * block_size.x + x);
      c += "    FLT" + std::to_string(element_size * 4) + " s" + i_s +
           " = src_data[src_addr_" + i_s + "];\n";
    }
  }
  c += GetComputationPart(block_size, element_size, op_def.precision);
  for (int i = 0; i < block_size.x * block_size.y; ++i) {
    std::string i_s = std::to_string(i);
    c += "    src_addr_" + i_s + " += src_size.w;\n";
  }
  c += "    weights_cache += " + std::to_string(block_size.z) + ";\n";
  c += "  }\n";  // src_size.z = SRC_DEPTH

  for (int z = 0; z < block_size.z; ++z) {
    const std::string z_s = std::to_string(z);
    if (z != 0) {
      c += "  if (Z + " + z_s + " >= dst_size.z) return;\n";
    }
    for (int y = 0; y < block_size.y; ++y) {
      const std::string y_s = std::to_string(y);
      for (int x = 0; x < block_size.x * element_size; ++x) {
        const std::string x_s = std::to_string(x);
        c += "  if (X + " + x_s + " < dst_size.x && Y + " + y_s +
             " < dst_size.y) {\n";
        c += "    FLT4 res = TO_FLT4(r" + z_s + y_s + x_s + ");\n";
        const LinkingContext context{"res", "X + " + x_s, "Y + " + y_s,
                                     "Z + " + z_s};
        c += PostProcess(linked_operations, context);
        c += "    " + dst_tensor.WriteWHS("res", "X + " + x_s, "Y + " + y_s,
                                          "Z + " + z_s);
        c += "  }\n";
      }
    }
  }
  c += "}\n";
  return c;
}

ConvBuffer1x1::ConvParams GetBestParams(const CLDevice& device,
                                        const OperationDef& definition,
                                        const BHWC& shape, int src_depth,
                                        int dst_depth) {
  ConvBuffer1x1::ConvParams conv_params;
  conv_params.element_size = 4;
  conv_params.block_size = int3(1, 1, 1);
  if (!device.IsMali()) {
    return conv_params;
  }
  bool can_use_flt8 = (shape.w * shape.b) % 2 == 0 &&
                      definition.precision != CalculationsPrecision::F32;
  bool is_midgard = device.IsMali() && device.GetInfo().mali_info.IsMidgard();
  if (is_midgard) {
    if (can_use_flt8) {
      conv_params.element_size = 8;
    }
    if (definition.precision == CalculationsPrecision::F16 || !can_use_flt8) {
      conv_params.block_size.x = 2;
    }
    return conv_params;
  }

  int task_size = shape.w * shape.b * shape.h * dst_depth;
  int block_size =
      GetRecommendedBlockSizeForConv(device, definition.precision, task_size);

  if (!can_use_flt8 && block_size > 4) {
    block_size = 4;
  }

  if (can_use_flt8 && block_size >= 2) {
    conv_params.element_size = 8;
    block_size /= 2;
  }
  if (block_size == 4) {
    conv_params.block_size.x = 2;
    if (definition.precision == CalculationsPrecision::F32 && dst_depth < 32) {
      conv_params.block_size.y = 2;
    } else {
      conv_params.block_size.z = 2;
    }
  } else if (block_size == 2) {
    if (dst_depth >= 32) {
      conv_params.block_size.z = 2;
    } else {
      conv_params.block_size.x = 2;
    }
  }

  return conv_params;
}

ConvBuffer1x1::ConvParams GetBestParams(const CLDevice& device,
                                        const OperationDef& definition,
                                        int src_depth, int dst_depth) {
  ConvBuffer1x1::ConvParams conv_params;
  conv_params.element_size = 4;
  conv_params.block_size = int3(1, 1, 1);
  if (device.IsMali() && definition.precision == CalculationsPrecision::F16 &&
      device.GetInfo().compute_units_count <= 4) {
    conv_params.block_size.x *= 2;
  }
  return conv_params;
}

}  // namespace

ConvBuffer1x1::ConvBuffer1x1(const OperationDef& definition,
                             const ConvParams& conv_params)
    : GPUOperation(definition), conv_params_(conv_params) {}

ConvBuffer1x1::ConvBuffer1x1(ConvBuffer1x1&& operation)
    : GPUOperation(std::move(operation)),
      weights_(std::move(operation.weights_)),
      biases_(std::move(operation.biases_)),
      conv_params_(std::move(operation.conv_params_)),
      kernel_(std::move(operation.kernel_)) {}

ConvBuffer1x1& ConvBuffer1x1::operator=(ConvBuffer1x1&& operation) {
  if (this != &operation) {
    weights_ = std::move(operation.weights_);
    biases_ = std::move(operation.biases_);
    std::swap(conv_params_, operation.conv_params_);
    kernel_ = std::move(operation.kernel_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status ConvBuffer1x1::Compile(const CreationContext& creation_context) {
  std::string code =
      GenerateConvBuffer1x1(definition_, conv_params_, linked_operations_);
  RETURN_IF_ERROR(creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_));
  return OkStatus();
}

Status ConvBuffer1x1::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_.GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  const int src_width_elements = IntegralDivideRoundUp(
      src_[0]->Width() * src_[0]->Batch(), (conv_params_.element_size / 4));
  int4 src_size = int4(src_width_elements, src_[0]->Height(), src_[0]->Slices(),
                       src_width_elements * src_[0]->Height());
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_size));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWBatchedHSB()));
  return OkStatus();
}

int3 ConvBuffer1x1::GetGridSize() const {
  const int dst_width_elements = IntegralDivideRoundUp(
      dst_[0]->Width() * dst_[0]->Batch(), (conv_params_.element_size / 4));
  const int grid_x =
      IntegralDivideRoundUp(dst_width_elements, conv_params_.block_size.x);
  const int grid_y =
      IntegralDivideRoundUp(dst_[0]->Height(), conv_params_.block_size.y);
  const int grid_z =
      IntegralDivideRoundUp(dst_[0]->Slices(), conv_params_.block_size.z);
  return int3(grid_x, grid_y, grid_z);
}

Status ConvBuffer1x1::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroupConv(params, kernel_, GetGridSize(),
                              &conv_params_.work_group_size);
}

Status ConvBuffer1x1::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(),
                                 conv_params_.work_group_size);
}

bool IsConvBuffer1x1Supported(const OperationDef& definition,
                              const Convolution2DAttributes& attr) {
  auto src_storage_type = definition.src_tensors[0].storage_type;
  return src_storage_type == TensorStorageType::BUFFER &&
         attr.weights.shape.w == 1 && attr.weights.shape.h == 1 &&
         attr.dilations.w == 1 && attr.dilations.h == 1 &&
         attr.strides.w == 1 && attr.strides.h == 1 &&
         attr.padding.prepended.w == 0 && attr.padding.prepended.h == 0 &&
         attr.padding.appended.w == 0 && attr.padding.appended.h == 0;
}

Status CreateConvBuffer1x1(const CreationContext& creation_context,
                           const OperationDef& definition,
                           const Convolution2DAttributes& attr,
                           ConvBuffer1x1* result, const BHWC* shape) {
  if (!IsConvBuffer1x1Supported(definition, attr)) {
    return InvalidArgumentError("ConvBuffer1x1 doesn't supported");
  }
  const int dst_depth = IntegralDivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
  ConvBuffer1x1::ConvParams conv_params;
  if (shape) {
    conv_params = GetBestParams(*creation_context.device, definition, *shape,
                                src_depth, dst_depth);
  } else {
    conv_params = GetBestParams(*creation_context.device, definition, src_depth,
                                dst_depth);
  }
  *result = ConvBuffer1x1(definition, conv_params);
  return result->UploadData(attr.weights, attr.bias, creation_context.context);
}

Status CreateConvBuffer1x1(const CreationContext& creation_context,
                           const OperationDef& definition,
                           const FullyConnectedAttributes& attr,
                           ConvBuffer1x1* result, const BHWC* shape) {
  const int dst_depth = IntegralDivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
  ConvBuffer1x1::ConvParams conv_params;
  if (shape) {
    conv_params = GetBestParams(*creation_context.device, definition, *shape,
                                src_depth, dst_depth);
  } else {
    conv_params = GetBestParams(*creation_context.device, definition, src_depth,
                                dst_depth);
  }
  conv_params.block_size.x *= conv_params.block_size.y;
  conv_params.block_size.y = 1;
  *result = ConvBuffer1x1(definition, conv_params);
  return result->UploadData(attr.weights, attr.bias, creation_context.context);
}

Status CreateConvBuffer1x1Wino4x4To6x6(const CreationContext& creation_context,
                                       const OperationDef& definition,
                                       const Convolution2DAttributes& attr,
                                       ConvBuffer1x1* result,
                                       const BHWC* shape) {
  const int dst_depth = IntegralDivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
  ConvBuffer1x1::ConvParams conv_params;
  if (shape) {
    conv_params = GetBestParams(*creation_context.device, definition, *shape,
                                src_depth, dst_depth);
  } else {
    conv_params = GetBestParams(*creation_context.device, definition, src_depth,
                                dst_depth);
  }
  conv_params.block_size.x *= conv_params.block_size.y;
  conv_params.block_size.y = 1;
  conv_params.different_weights_for_height = true;
  *result = ConvBuffer1x1(definition, conv_params);
  return result->UploadDataForWinograd4x4To6x6(
      attr.weights, *creation_context.device, creation_context.context);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
