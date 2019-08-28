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

#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_powervr.h"

#include <algorithm>
#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

ConvPowerVR::ConvPowerVR(const OperationDef& definition,
                         const Convolution2DAttributes& attr,
                         const ConvParams& conv_params)
    : GPUOperation(definition),
      kernel_size_(attr.weights.shape.w, attr.weights.shape.h),
      stride_(attr.strides.w, attr.strides.h),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h),
      dilation_(attr.dilations.w, attr.dilations.h),
      conv_params_(conv_params) {}

ConvPowerVR::ConvPowerVR(ConvPowerVR&& operation)
    : GPUOperation(std::move(operation)),
      weights_(std::move(operation.weights_)),
      biases_(std::move(operation.biases_)),
      kernel_size_(operation.kernel_size_),
      stride_(operation.stride_),
      padding_(operation.padding_),
      dilation_(operation.dilation_),
      conv_params_(operation.conv_params_),
      kernel_(std::move(operation.kernel_)) {}

ConvPowerVR& ConvPowerVR::operator=(ConvPowerVR&& operation) {
  if (this != &operation) {
    weights_ = std::move(operation.weights_);
    biases_ = std::move(operation.biases_);
    std::swap(kernel_size_, operation.kernel_size_);
    std::swap(stride_, operation.stride_);
    std::swap(padding_, operation.padding_);
    std::swap(dilation_, operation.dilation_);
    std::swap(conv_params_, operation.conv_params_);
    kernel_ = std::move(operation.kernel_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status ConvPowerVR::Compile(const CreationContext& creation_context) {
  const std::string code = GenerateConvPowerVR1x1(
      definition_.src_tensors[0], definition_.dst_tensors[0],
      definition_.precision, conv_params_, linked_operations_);
  std::vector<CompilerOptions> options;
  if (definition_.precision == CalculationsPrecision::F16 &&
      creation_context.device->IsPowerVR()) {
    options.push_back(CompilerOptions::POWERVR_FP16);
  }
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", options, *creation_context.context,
      *creation_context.device, &kernel_);
}

Status ConvPowerVR::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_.GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));
  return OkStatus();
}

int3 ConvPowerVR::GetGridSize() const {
  const int grid_x =
      IntegralDivideRoundUp(dst_[0]->Width(), conv_params_.block_size.x);
  const int grid_y =
      IntegralDivideRoundUp(dst_[0]->Height(), conv_params_.block_size.y);
  const int grid_z =
      IntegralDivideRoundUp(dst_[0]->Depth(), conv_params_.block_size.z);
  const int wg_x =
      IntegralDivideRoundUp(grid_x, conv_params_.work_group_size.x);
  const int wg_y =
      IntegralDivideRoundUp(grid_y, conv_params_.work_group_size.y);
  const int wg_z =
      IntegralDivideRoundUp(grid_z, conv_params_.work_group_size.z);
  return int3(wg_z * conv_params_.work_group_size.x,
              wg_x * conv_params_.work_group_size.y,
              wg_y * conv_params_.work_group_size.z);
}

Status ConvPowerVR::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(),
                                 conv_params_.work_group_size);
}

std::string GenerateConvPowerVR1x1(
    const TensorDescriptor& src_descriptor,
    const TensorDescriptor& dst_descriptor, CalculationsPrecision precision,
    const ConvPowerVR::ConvParams& conv_params,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  std::string c = GetCommonDefines(precision);
  TensorCodeGenerator src_tensor("src_data", "src_size", src_descriptor);
  TensorCodeGenerator dst_tensor("dst_data", "dst_size", dst_descriptor);

  c += "#define SIMD_BARRIER " +
       (!conv_params.explicit_sync
            ? std::string("")
            : std::string("barrier(CLK_LOCAL_MEM_FENCE)")) +
       "\n";
  c += "#define SIMD_WAIT_EVENT(E) " +
       (!conv_params.explicit_sync ? std::string("")
                                   : std::string("wait_group_events(1, &E);")) +
       "\n";
  const int3 work_group_size = conv_params.work_group_size;
  const int3 block_size = conv_params.block_size;
  c += "__attribute__((reqd_work_group_size(" +
       std::to_string(work_group_size.x) + ", " +
       std::to_string(work_group_size.y) + ", " +
       std::to_string(work_group_size.z) + ")))\n";
  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += "    __global ACCUM_FLT4* filters_buffer,    \n";
  c += "    __global ACCUM_FLT4* biases             \n";
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,                   \n";
  c += "    int4 dst_size                    \n";
  c += ") {\n";
  c += "  int X = (get_group_id(1) * 8 + get_local_id(0)) * " +
       std::to_string(block_size.x) + ";\n";
  c += "  int Y = (get_group_id(2) * 4 + get_local_id(1)) * " +
       std::to_string(block_size.y) + ";\n";
  c += "  int Z = (get_group_id(0) * 1 + get_local_id(2)) * " +
       std::to_string(block_size.z) + ";\n";
  for (int z = 0; z < block_size.z; ++z) {
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x; ++x) {
        c += "  ACCUM_FLT4 r" + std::to_string(z) + std::to_string(y) +
             std::to_string(x) + " = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
      }
    }
  }
  c += "  __local ACCUM_FLT4 data[" +
       std::to_string(block_size.z * 4 * conv_params.src_depth_loop_size) +
       "];\n";
  c += "  __global ACCUM_FLT4* filters_loc = filters_buffer + Z * 4 * "
       "src_size.w;\n";
  if (src_descriptor.storage_type == TensorStorageType::BUFFER) {
    c += "  const int src_layer_offset = src_size.x * src_size.y;\n";
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x; ++x) {
        std::string xc = "min(X + " + std::to_string(x) + ", src_size.x - 1)";
        std::string yc = "min(Y + " + std::to_string(y) + ", src_size.y - 1)";
        std::string id = std::to_string(y) + std::to_string(x);
        c += "  int src_a_" + id + " = " + yc + " * src_size.x + " + xc + ";\n";
      }
    }
  }

  auto declare_src = [&]() {
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x; ++x) {
        const std::string id = std::to_string(y) + std::to_string(x);
        if (precision == CalculationsPrecision::F32_F16) {
          c += "    ACCUM_FLT4 src" + id + ";\n";
        } else {
          c += "    FLT4 src" + id + ";\n";
        }
      }
    }
  };
  auto read_src = [&]() {
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x; ++x) {
        if (src_descriptor.storage_type == TensorStorageType::BUFFER) {
          std::string id = std::to_string(y) + std::to_string(x);
          if (precision == CalculationsPrecision::F32_F16) {
            c += "    src" + id + " = convert_float4(src_data[src_a_" + id +
                 "]);\n";
          } else {
            c += "    src" + id + " = src_data[src_a_" + id + "];\n";
          }
          c += "    src_a_" + id + " += src_layer_offset;\n";
        } else {
          std::string id = std::to_string(y) + std::to_string(x);
          if (precision == CalculationsPrecision::F32_F16) {
            c += "    src" + id + " = " +
                 src_tensor.ReadAsFloat3D("X + " + std::to_string(x),
                                          "Y + " + std::to_string(y), "s",
                                          TextureAddressMode::DONT_CARE) +
                 ";\n";
          } else {
            c += "    src" + id + " = " +
                 src_tensor.Read3D("X + " + std::to_string(x),
                                   "Y + " + std::to_string(y), "s",
                                   TextureAddressMode::DONT_CARE) +
                 ";\n";
          }
        }
      }
    }
  };
  auto conv_core = [&]() {
    const std::string channels[] = {"x", "y", "z", "w"};
    for (int z = 0; z < block_size.z; ++z) {
      for (int ch = 0; ch < 4; ++ch) {
        for (int y = 0; y < block_size.y; ++y) {
          for (int x = 0; x < block_size.x; ++x) {
            std::string id = std::to_string(y) + std::to_string(x);
            c += "    r" + std::to_string(z) + id + " += data[" +
                 std::to_string(z * 4 + ch) + "] * src" + id + "." +
                 channels[ch] + ";\n";
          }
        }
      }
    }
  };

  c += "  int s = 0;\n";
  c += "  do {\n";
  declare_src();
  c += "    SIMD_BARRIER;\n";
  c += "    event_t e = async_work_group_copy(data, filters_loc, " +
       std::to_string(block_size.z * 4 * conv_params.src_depth_loop_size) +
       ", 0);\n";
  read_src();
  c += "    SIMD_WAIT_EVENT(e);\n";
  c += "    s += 1;\n";
  conv_core();
  for (int i = 1; i < conv_params.src_depth_loop_size; ++i) {
    read_src();
    conv_core();
    c += "    s += 1;\n";
  }
  c += "    filters_loc += " +
       std::to_string(block_size.z * 4 * conv_params.src_depth_loop_size) +
       ";\n";
  c += "  } while (s < src_size.w);\n";
  c += "  SIMD_BARRIER;\n";
  c += "  event_t e = async_work_group_copy(data, biases + Z, " +
       std::to_string(block_size.z) + ", 0);\n";
  c += "  SIMD_WAIT_EVENT(e);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.w) {\n";
  c += "    return;\n";
  c += "  }\n";
  for (int z = 0; z < block_size.z; ++z) {
    c += "  if (Z + " + std::to_string(z) + " >= dst_size.w) return;\n";
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x; ++x) {
        const std::string xs = "X + " + std::to_string(x);
        const std::string ys = "Y + " + std::to_string(y);
        const std::string zs = "Z + " + std::to_string(z);
        const std::string r_id =
            std::to_string(z) + std::to_string(y) + std::to_string(x);
        bool need_x_check = x != 0;
        bool need_y_check = y != 0;
        if (need_x_check && need_y_check) {
          c += "  if (" + xs + " < dst_size.x && " + ys + " < dst_size.y) {\n";
        } else if (need_x_check && !need_y_check) {
          c += "  if (" + xs + " < dst_size.x) {\n";
        } else if (!need_x_check && need_y_check) {
          c += "  if (" + ys + " < dst_size.y) {\n";
        } else {
          c += "  {\n";
        }
        c += "    FLT4 res = TO_FLT4(r" + r_id + " + data[" +
             std::to_string(z) + "]);\n";
        c += "    " + dst_tensor.GetAddress("address", xs, ys, zs) + "\n";
        c += PostProcess(linked_operations, "res", zs, "address");
        c += "    " + dst_tensor.Write3D("res", "address") + "\n";
        c += "  }\n";
      }
    }
  }
  c += "}\n";
  return c;
}

bool IsConvPowerVRSupported(const OperationDef& definition,
                            const Convolution2DAttributes& attr) {
  return attr.weights.shape.w == 1 && attr.weights.shape.h == 1 &&
         attr.strides == HW(1, 1) && attr.dilations == HW(1, 1) &&
         attr.padding.prepended == HW(0, 0) &&
         attr.padding.appended == HW(0, 0);
}

ConvPowerVR::ConvParams GuessBestParams(const CLDevice& device,
                                        const OperationDef& definition,
                                        const Convolution2DAttributes& attr) {
  ConvPowerVR::ConvParams conv_params;
  conv_params.block_size = int3(1, 1, 4);
  conv_params.work_group_size = int3(8, 4, 1);
  conv_params.src_depth_loop_size = 1;
  conv_params.explicit_sync = !device.IsPowerVR();
  const int dst_depth = IntegralDivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
  if (dst_depth % 8 == 0 || dst_depth >= 32) {
    conv_params.block_size.z = 8;
  } else if (dst_depth % 4 == 0 || dst_depth >= 8) {
    conv_params.block_size.z = 4;
  } else if (dst_depth % 2 == 0 || dst_depth >= 4) {
    conv_params.block_size.z = 2;
  } else {
    conv_params.block_size.z = dst_depth;
  }
  if (definition.precision == CalculationsPrecision::F16) {
    conv_params.block_size.z = std::min(4, conv_params.block_size.z);
    if (src_depth % 2 == 0) {
      conv_params.src_depth_loop_size = 2;
    }
    if (src_depth % 4 == 0 && conv_params.block_size.z <= 2) {
      conv_params.src_depth_loop_size = 4;
    }
    if (conv_params.block_size.z == 1) {
      if (src_depth % 8 == 0) {
        conv_params.src_depth_loop_size = 8;
      }
      if (src_depth % 4 == 0) {
        conv_params.src_depth_loop_size = 4;
      }
      if (src_depth % 2 == 0) {
        conv_params.src_depth_loop_size = 2;
      }
      if (src_depth <= 8) {
        conv_params.src_depth_loop_size = src_depth;
      }
    }
    conv_params.block_size.x = 2;
    conv_params.work_group_size = int3(4, 8, 1);
  }

  return conv_params;
}

Status CreateConvPowerVR(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const Convolution2DAttributes& attr,
                         ConvPowerVR* result) {
  *result =
      ConvPowerVR(definition, attr,
                  GuessBestParams(*creation_context.device, definition, attr));
  RETURN_IF_ERROR(
      result->UploadWeights(attr.weights, creation_context.context));
  LinearStorageCreateInfo create_info;
  create_info.storage_type = LinearStorageType::BUFFER;
  create_info.data_type = definition.precision == CalculationsPrecision::F16
                              ? DataType::FLOAT16
                              : DataType::FLOAT32;
  create_info.aligned_size = attr.weights.shape.o;
  RETURN_IF_ERROR(CreateLinearStorage(
      create_info, attr.bias, creation_context.context, &result->biases_));

  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
