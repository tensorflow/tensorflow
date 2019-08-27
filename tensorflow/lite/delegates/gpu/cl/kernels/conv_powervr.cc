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
namespace {

std::string GenerateConvPowerVR1x1(
    const TensorDescriptor& src_descriptor,
    const TensorDescriptor& dst_descriptor, CalculationsPrecision precision,
    const int3& block_size,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  std::string c = GetCommonDefines(precision);
  TensorCodeGenerator src_tensor("src_data", "src_size", src_descriptor);
  TensorCodeGenerator dst_tensor("dst_data", "dst_size", dst_descriptor);

  bool power_vr = true;
  c += "#define SIMD_BARRIER " +
       (power_vr ? std::string("")
                 : std::string("barrier(CLK_LOCAL_MEM_FENCE)")) +
       "\n";
  c += "#define SIMD_WAIT_EVENT(E) " +
       (power_vr ? std::string("") : std::string("wait_group_events(1, &E);")) +
       "\n";
  c += "__attribute__((reqd_work_group_size(8, 4, 1)))\n";
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
  c += "  __local ACCUM_FLT4 data[" + std::to_string(block_size.z * 4) + "];\n";
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
  c += "  int s = 0;\n";
  c += "  do {\n";
  for (int y = 0; y < block_size.y; ++y) {
    for (int x = 0; x < block_size.x; ++x) {
      if (src_descriptor.storage_type == TensorStorageType::BUFFER) {
        std::string id = std::to_string(y) + std::to_string(x);
        if (precision == CalculationsPrecision::F32_F16) {
          c += "    ACCUM_FLT4 src" + id + " = convert_float4(src_data[src_a_" +
               id + "]);\n";
        } else {
          c += "    FLT4 src" + id + " = src_data[src_a_" + id + "];\n";
        }
        c += "    src_a_" + id + " += src_layer_offset;\n";
      } else {
        std::string id = std::to_string(y) + std::to_string(x);
        if (precision == CalculationsPrecision::F32_F16) {
          c += "    ACCUM_FLT4 src" + id + " = " +
               src_tensor.ReadAsFloat3D("X + " + std::to_string(x),
                                        "Y + " + std::to_string(y), "s",
                                        TextureAddressMode::DONT_CARE) +
               ";\n";
        } else {
          c += "    FLT4 src" + id + " = " +
               src_tensor.Read3D("X + " + std::to_string(x),
                                 "Y + " + std::to_string(y), "s",
                                 TextureAddressMode::DONT_CARE) +
               ";\n";
        }
      }
    }
  }
  c += "    SIMD_BARRIER;\n";
  c += "    event_t e = async_work_group_copy(data, filters_loc, " +
       std::to_string(block_size.z * 4) + ", 0);\n";
  c += "    SIMD_WAIT_EVENT(e);\n";
  c += "    s += 1;\n";
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
  c += "    filters_loc += " + std::to_string(block_size.z * 4) + ";\n";
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
        c += "  if (" + xs + " < dst_size.x && " + ys + " < dst_size.y) {\n";
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
}  // namespace

ConvPowerVR::ConvPowerVR(const OperationDef& definition,
                         const Convolution2DAttributes& attr,
                         const int3& block_size)
    : GPUOperation(definition),
      kernel_size_(attr.weights.shape.w, attr.weights.shape.h),
      stride_(attr.strides.w, attr.strides.h),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h),
      dilation_(attr.dilations.w, attr.dilations.h),
      block_size_(block_size),
      work_group_size_(8, 4, 1) {}

ConvPowerVR::ConvPowerVR(ConvPowerVR&& operation)
    : GPUOperation(std::move(operation)),
      weights_(std::move(operation.weights_)),
      biases_(std::move(operation.biases_)),
      kernel_size_(operation.kernel_size_),
      stride_(operation.stride_),
      padding_(operation.padding_),
      dilation_(operation.dilation_),
      block_size_(operation.block_size_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

ConvPowerVR& ConvPowerVR::operator=(ConvPowerVR&& operation) {
  if (this != &operation) {
    weights_ = std::move(operation.weights_);
    biases_ = std::move(operation.biases_);
    std::swap(kernel_size_, operation.kernel_size_);
    std::swap(stride_, operation.stride_);
    std::swap(padding_, operation.padding_);
    std::swap(dilation_, operation.dilation_);
    std::swap(block_size_, operation.block_size_);
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status ConvPowerVR::Compile(const CreationContext& creation_context) {
  const std::string code = GenerateConvPowerVR1x1(
      definition_.src_tensors[0], definition_.dst_tensors[0],
      definition_.precision, block_size_, linked_operations_);
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
  const int grid_x = IntegralDivideRoundUp(dst_[0]->Width(), block_size_.x);
  const int grid_y = IntegralDivideRoundUp(dst_[0]->Height(), block_size_.y);
  const int grid_z = IntegralDivideRoundUp(dst_[0]->Depth(), block_size_.z);
  const int wg_x = IntegralDivideRoundUp(grid_x, work_group_size_.x);
  const int wg_y = IntegralDivideRoundUp(grid_y, work_group_size_.y);
  const int wg_z = IntegralDivideRoundUp(grid_z, work_group_size_.z);
  return int3(wg_z * work_group_size_.x, wg_x * work_group_size_.y,
              wg_y * work_group_size_.z);
}

Status ConvPowerVR::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

bool IsConvPowerVRSupported(const OperationDef& definition,
                            const Convolution2DAttributes& attr) {
  return attr.weights.shape.w == 1 && attr.weights.shape.h == 1 &&
         attr.strides == HW(1, 1) && attr.dilations == HW(1, 1) &&
         attr.padding.prepended == HW(0, 0) &&
         attr.padding.appended == HW(0, 0);
}

Status CreateConvPowerVR(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const Convolution2DAttributes& attr,
                         ConvPowerVR* result) {
  int3 block_size = int3(1, 1, 4);
  const int dst_depth = IntegralDivideRoundUp(attr.weights.shape.o, 4);
  if (dst_depth % 8 == 0 || dst_depth >= 32) {
    block_size.z = 8;
  } else if (dst_depth % 4 == 0 || dst_depth >= 8) {
    block_size.z = 4;
  } else if (dst_depth % 2 == 0 || dst_depth >= 4) {
    block_size.z = 2;
  } else {
    block_size.z = dst_depth;
  }
  if (definition.precision == CalculationsPrecision::F16) {
    block_size.y = 2;
  }
  *result = ConvPowerVR(definition, attr, block_size);
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
