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
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
std::string GenerateUploadByThreads(const std::string& local_ptr_name,
                                    const std::string& global_ptr_name,
                                    const std::string& global_offset_name,
                                    const std::string& lid_name,
                                    int total_work_items,
                                    int elements_to_upload) {
  std::string c;
  std::string offset =
      global_offset_name.empty() ? "" : global_offset_name + " + ";
  const int groups = elements_to_upload / total_work_items;
  const int reminder = elements_to_upload % total_work_items;
  for (int i = 0; i < groups; ++i) {
    c += "    " + local_ptr_name + "[" + lid_name + " + " +
         std::to_string(total_work_items * i) + "] = " + global_ptr_name + "[" +
         offset + lid_name + " + " + std::to_string(total_work_items * i) +
         "];\n";
  }
  if (reminder != 0) {
    c += "    if (" + lid_name + " < " + std::to_string(reminder) + ") {\n";
    c += "      " + local_ptr_name + "[" + lid_name + " + " +
         std::to_string(total_work_items * groups) + "] = " + global_ptr_name +
         "[" + offset + lid_name + " + " +
         std::to_string(total_work_items * groups) + "];\n";
    c += "    }\n";
  }
  return c;
}

std::string GenerateAsyncUpload(const std::string& local_ptr_name,
                                const std::string& global_ptr_name,
                                const std::string& global_offset_name,
                                int elements_to_upload) {
  std::string c;
  std::string offset =
      global_offset_name.empty() ? "" : " + " + global_offset_name;
  c += "    async_work_group_copy(" + local_ptr_name + ", " + global_ptr_name +
       offset + ", " + std::to_string(elements_to_upload) + ", 0);\n";
  return c;
}

std::string GenerateBlockCoords(const int3& block_size,
                                const int3& work_group_launch_order,
                                bool linear_hw) {
  std::string c;
  int3 launch_remap;
  launch_remap[work_group_launch_order.x] = 0;
  launch_remap[work_group_launch_order.y] = 1;
  launch_remap[work_group_launch_order.z] = 2;
  if (linear_hw) {
    if (work_group_launch_order[0] == 0) {
      c += "  int linear_hw = get_global_id(0);\n";
    } else {
      c += "  int linear_hw = get_group_id(" + std::to_string(launch_remap[0]) +
           ") * get_local_size(0) + get_local_id(0);\n";
    }
    c += "  int Y = (linear_hw / task_size_x) * " +
         std::to_string(block_size.y) + ";\n";
    c += "  int X = (linear_hw % task_size_x) * " +
         std::to_string(block_size.x) + ";\n";
    if (work_group_launch_order[1] == 1) {
      c += "  int Z = get_global_id(1) * " + std::to_string(block_size.z) +
           ";\n";
    } else {
      c += "  int Z = (get_group_id(" + std::to_string(launch_remap[1]) +
           ") * get_local_size(1) + get_local_id(1)) * " +
           std::to_string(block_size.z) + ";\n";
    }
  } else {
    if (work_group_launch_order[0] == 0) {
      c += "  int X = get_global_id(0) * " + std::to_string(block_size.x) +
           ";\n";
    } else {
      c += "  int X = (get_group_id(" + std::to_string(launch_remap[0]) +
           ") * get_local_size(0) + get_local_id(0)) * " +
           std::to_string(block_size.x) + ";\n";
    }
    if (work_group_launch_order[1] == 1) {
      c += "  int Y = get_global_id(1) * " + std::to_string(block_size.y) +
           ";\n";
    } else {
      c += "  int Y = (get_group_id(" + std::to_string(launch_remap[1]) +
           ") * get_local_size(1) + get_local_id(1)) * " +
           std::to_string(block_size.y) + ";\n";
    }
    if (work_group_launch_order[2] == 2) {
      c += "  int Z = get_global_id(2) * " + std::to_string(block_size.z) +
           ";\n";
    } else {
      c += "  int Z = (get_group_id(" + std::to_string(launch_remap[2]) +
           ") * get_local_size(2) + get_local_id(2)) * " +
           std::to_string(block_size.z) + ";\n";
    }
  }

  return c;
}
}  // namespace

ConvPowerVR::ConvPowerVR(const OperationDef& definition,
                         const Convolution2DAttributes& attr,
                         const CLDevice& device, const BHWC* dst_shape)
    : GPUOperation(definition),
      stride_padding_(attr.strides.w, attr.strides.h, -attr.padding.prepended.w,
                      -attr.padding.prepended.h),
      kernel_dilation_(attr.weights.shape.w, attr.weights.shape.h,
                       attr.dilations.w, attr.dilations.h),
      conv_params_(GuessBestParams(device, definition, attr, dst_shape)) {}

ConvPowerVR::ConvPowerVR(const OperationDef& definition,
                         const Convolution2DAttributes& attr,
                         const BHWC& weights_shape, const CLDevice& device,
                         const BHWC* dst_shape)
    : GPUOperation(definition),
      stride_padding_(attr.strides.w, attr.strides.h, -attr.padding.prepended.w,
                      -attr.padding.prepended.h),
      kernel_dilation_(weights_shape.w, weights_shape.h, attr.dilations.w,
                       attr.dilations.h),
      conv_params_(GuessBestParams(device, definition, attr, weights_shape,
                                   dst_shape)) {}

ConvPowerVR::ConvPowerVR(const OperationDef& definition,
                         const FullyConnectedAttributes& attr,
                         const CLDevice& device, const BHWC* dst_shape)
    : GPUOperation(definition),
      stride_padding_(1, 1, 0, 0),
      kernel_dilation_(1, 1, 1, 1),
      conv_params_(GuessBestParams(device, definition, attr, dst_shape)) {}

ConvPowerVR::ConvPowerVR(const OperationDef& definition)
    : GPUOperation(definition),
      stride_padding_(1, 1, 0, 0),
      kernel_dilation_(1, 1, 1, 1) {}

ConvPowerVR::ConvPowerVR(ConvPowerVR&& operation)
    : GPUOperation(std::move(operation)),
      weights_(std::move(operation.weights_)),
      biases_(std::move(operation.biases_)),
      stride_padding_(operation.stride_padding_),
      kernel_dilation_(operation.kernel_dilation_),
      conv_params_(operation.conv_params_),
      kernel_(std::move(operation.kernel_)) {}

ConvPowerVR& ConvPowerVR::operator=(ConvPowerVR&& operation) {
  if (this != &operation) {
    weights_ = std::move(operation.weights_);
    biases_ = std::move(operation.biases_);
    std::swap(stride_padding_, operation.stride_padding_);
    std::swap(kernel_dilation_, operation.kernel_dilation_);
    std::swap(conv_params_, operation.conv_params_);
    kernel_ = std::move(operation.kernel_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status ConvPowerVR::Compile(const CreationContext& creation_context) {
  const bool stride_correction =
      definition_.IsBatchSupported() && stride_padding_.x != 1;
  const std::string code =
      GenerateConv(*creation_context.device, definition_, stride_correction,
                   conv_params_, linked_operations_);
  std::vector<CompilerOptions> options;
  if (definition_.precision == CalculationsPrecision::F16 &&
      creation_context.device->IsPowerVR()) {
    options.push_back(CompilerOptions::POWERVR_FP16);
  }
  if (conv_params_.IsPrivateMemBroadcast()) {
    options.push_back(CompilerOptions::CL_2_0);
  }
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", options, *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status ConvPowerVR::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  if (definition_.src_tensors.size() == 1) {
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_.GetMemoryPtr()));
  } else {
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[1]->GetMemoryPtr()));
  }
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  if (!conv_params_.x_kernel_is_1 || !conv_params_.y_kernel_is_1) {
    RETURN_IF_ERROR(kernel_.SetBytesAuto(
        int4(stride_padding_.x, stride_padding_.y,
             stride_padding_.z * src_[0]->Batch(), stride_padding_.w)));
    RETURN_IF_ERROR(kernel_.SetBytesAuto(
        int4(kernel_dilation_.x, kernel_dilation_.y,
             kernel_dilation_.z * src_[0]->Batch(), kernel_dilation_.w)));
  }
  if (conv_params_.linear_hw) {
    const int grid_x = DivideRoundUp(dst_[0]->Width() * dst_[0]->Batch(),
                                     conv_params_.block_size.x);
    RETURN_IF_ERROR(kernel_.SetBytesAuto(grid_x));
  }
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWBatchedHSB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWBatchedHSB()));
  return absl::OkStatus();
}

int3 ConvPowerVR::GetGridSize() const {
  const int grid_x = DivideRoundUp(dst_[0]->Width() * dst_[0]->Batch(),
                                   conv_params_.block_size.x);
  const int grid_y =
      DivideRoundUp(dst_[0]->Height(), conv_params_.block_size.y);
  const int grid_z =
      DivideRoundUp(dst_[0]->Slices(), conv_params_.block_size.z);
  int3 wg;

  if (conv_params_.linear_hw) {
    wg.x = DivideRoundUp(grid_x * grid_y, conv_params_.work_group_size.x);
    wg.y = DivideRoundUp(grid_z, conv_params_.work_group_size.y);
    return int3(wg[conv_params_.work_group_launch_order[0]] *
                    conv_params_.work_group_size.x,
                wg[conv_params_.work_group_launch_order[1]] *
                    conv_params_.work_group_size.y,
                1);
  } else {
    wg.x = DivideRoundUp(grid_x, conv_params_.work_group_size.x);
    wg.y = DivideRoundUp(grid_y, conv_params_.work_group_size.y);
    wg.z = DivideRoundUp(grid_z, conv_params_.work_group_size.z);
    return int3(wg[conv_params_.work_group_launch_order[0]] *
                    conv_params_.work_group_size.x,
                wg[conv_params_.work_group_launch_order[1]] *
                    conv_params_.work_group_size.y,
                wg[conv_params_.work_group_launch_order[2]] *
                    conv_params_.work_group_size.z);
  }
}

absl::Status ConvPowerVR::Tune(const TuningParameters& params) {
  if (conv_params_.weights_upload_type ==
          WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP ||
      conv_params_.weights_upload_type ==
          WeightsUploadType::LOCAL_MEM_BY_THREADS ||
      conv_params_.fixed_work_group_size) {
    return absl::OkStatus();
  }
  if (conv_params_.work_group_launch_order[0] == 0 &&
      conv_params_.work_group_launch_order[1] == 1 &&
      conv_params_.work_group_launch_order[2] == 2) {
    RETURN_IF_ERROR(BindArguments());
    return GetBestWorkGroupConv(params, kernel_, GetGridSize(),
                                &conv_params_.work_group_size);
  }
  return absl::OkStatus();
}

absl::Status ConvPowerVR::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(),
                                 conv_params_.work_group_size);
}

std::string GenerateConv(
    const CLDevice& device, const OperationDef& op_def, bool stride_correction,
    const ConvPowerVR::ConvParams& conv_params,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  std::string c = GetCommonDefines(op_def.precision);
  TensorCodeGenerator src_tensor(
      "src_data", WHSPoint{"src_size.x", "src_size.y", "src_size.z"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data", WHSPoint{"dst_size.x", "dst_size.y", "dst_size.z"},
      op_def.dst_tensors[0]);

  const bool is1x1 = conv_params.x_kernel_is_1 && conv_params.y_kernel_is_1;
  const auto src_tensor_type = op_def.src_tensors[0].storage_type;
  const bool buffer_type = src_tensor_type == TensorStorageType::BUFFER ||
                           src_tensor_type == TensorStorageType::IMAGE_BUFFER;
  const bool manual_clamp = buffer_type && !is1x1;

  const bool need_local_mem =
      conv_params.weights_upload_type ==
          ConvPowerVR::WeightsUploadType::LOCAL_MEM_BY_THREADS ||
      conv_params.weights_upload_type ==
          ConvPowerVR::WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP;

  const int local_mem_size =
      conv_params.block_size.z * 4 * conv_params.src_depth_loop_size;

  const bool use_simd_broadcast = conv_params.IsPrivateMemBroadcast();
  const int simd_size = conv_params.GetSimdSize();

  const bool late_oob_check = need_local_mem || use_simd_broadcast;

  const std::string weights_space =
      conv_params.weights_upload_type ==
              ConvPowerVR::WeightsUploadType::CONSTANT_MEM
          ? "__constant"
          : "__global";

  const std::string weights_data_type =
      conv_params.weights_data_type == DataType::FLOAT32 ? "float4" : "half4";

  const std::string weights_global_ptr =
      weights_space + " " + weights_data_type + "*";

  if (use_simd_broadcast) {
    if (device.cl_version() == OpenCLVersion::CL_2_0) {
      c += "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n";
    }
  }

  const int3 work_group_size = conv_params.work_group_size;
  const int3 block_size = conv_params.block_size;
  if (conv_params.fixed_work_group_size) {
    c += "__attribute__((reqd_work_group_size(" +
         std::to_string(work_group_size.x) + ", " +
         std::to_string(work_group_size.y) + ", " +
         std::to_string(work_group_size.z) + ")))\n";
  }
  if (use_simd_broadcast && device.IsIntel()) {
    c += "__attribute__((intel_reqd_sub_group_size(" +
         std::to_string(simd_size) + ")))\n";
  }
  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += "    " + weights_global_ptr + " filters_buffer,    \n";
  c += "    " + weights_global_ptr + " biases             \n";
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  if (!is1x1) {
    c += "    int4 stride_padding,           \n";
    c += "    int4 kernel_dilation,          \n";
  }
  if (conv_params.linear_hw) {
    c += "    int task_size_x,               \n";
  }
  c += "    int4 src_size,                   \n";
  c += "    int4 dst_size                    \n";
  c += ") {\n";
  c += GenerateBlockCoords(conv_params.block_size,
                           conv_params.work_group_launch_order,
                           conv_params.linear_hw);
  std::vector<std::string> dst_x(conv_params.block_size.x);
  for (int x = 0; x < conv_params.block_size.x; ++x) {
    dst_x[x] = "(X + " + std::to_string(x) + ")";
  }
  std::vector<std::string> dst_y(conv_params.block_size.y);
  for (int y = 0; y < conv_params.block_size.y; ++y) {
    dst_y[y] = "(Y + " + std::to_string(y) + ")";
  }
  if (!late_oob_check) {
    c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) {\n";
    c += "    return;\n";
    c += "  }\n";
  }
  if (conv_params.weights_upload_type ==
      ConvPowerVR::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    if (conv_params.linear_hw) {
      c += "  int lid = get_local_id(0);\n";
    } else {
      c += "  int lid = get_local_id(1) * " +
           std::to_string(work_group_size.x) + " + get_local_id(0);\n";
    }
  }
  if (use_simd_broadcast) {
    c += "  int simd_id = get_sub_group_local_id();\n";
  }
  for (int z = 0; z < block_size.z; ++z) {
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x; ++x) {
        c += "  ACCUM_FLT4 r" + std::to_string(z) + std::to_string(y) +
             std::to_string(x) + " = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
      }
    }
  }
  if (!is1x1) {
    for (int x = 0; x < block_size.x; ++x) {
      if (stride_correction) {
        c += "  int xc" + std::to_string(x) + " = " +
             GetXStrideCorrected(dst_x[x], "src_size.w", "stride_padding.x",
                                 "stride_padding.z") +
             ";\n";
      } else {
        c += "  int xc" + std::to_string(x) + " = " + dst_x[x] +
             " * stride_padding.x + stride_padding.z;\n";
      }
    }
    for (int y = 0; y < block_size.y; ++y) {
      c += "  int yc" + std::to_string(y) + " = " + dst_y[y] +
           " * stride_padding.y + stride_padding.w;\n";
    }
  }
  if (need_local_mem) {
    c += "  __local " + weights_data_type + " weights_cache[" +
         std::to_string(local_mem_size) + "];\n";
  } else {
    c += "    " + weights_global_ptr + " weights_cache;\n";
  }
  if (is1x1) {
    if (conv_params.different_weights_for_height) {
      c += "  " + weights_global_ptr +
           " filters_loc = filters_buffer + (Z * src_size.y + Y * " +
           std::to_string(block_size.z) +
           ") * "
           "4 * src_size.z;\n";
    } else {
      c += "  " + weights_global_ptr +
           " filters_loc = filters_buffer + Z * 4 * "
           "src_size.z;\n";
    }
  } else {
    c += "  " + weights_global_ptr +
         " filters_loc = filters_buffer + Z * 4 * "
         "src_size.z * kernel_dilation.x * kernel_dilation.y;\n";
  }
  if (buffer_type) {
    c += "  const int src_layer_offset = src_size.x * src_size.y;\n";
  }
  if (!is1x1) {
    c += "  for (int ky = 0; ky < kernel_dilation.y; ++ky) {\n";
    for (int y = 0; y < block_size.y; ++y) {
      const std::string yck = "yck" + std::to_string(y);
      c += "  int " + yck + " = ky * kernel_dilation.w + yc" +
           std::to_string(y) + ";\n";
      if (manual_clamp) {
        c += "  bool my" + std::to_string(y) + " = " + yck + " >= 0 && " + yck +
             " < src_size.y;\n";
        c += "  " + yck + " = clamp(" + yck + ", 0, src_size.y - 1);\n";
      }
    }
    c += "  for (int kx = 0; kx < kernel_dilation.x; ++kx) {\n";
    for (int x = 0; x < block_size.x; ++x) {
      const std::string xck = "xck" + std::to_string(x);
      c += "  int xck" + std::to_string(x) + " = kx * kernel_dilation.z + xc" +
           std::to_string(x) + ";\n";
      if (manual_clamp) {
        c += "  bool mx" + std::to_string(x) + " = " + xck + " >= 0 && " + xck +
             " < src_size.x;\n";
        c += "  " + xck + " = clamp(" + xck + ", 0, src_size.x - 1);\n";
      }
    }
  }
  if (buffer_type) {
    for (int y = 0; y < block_size.y; ++y) {
      const std::string yck = "yck" + std::to_string(y);
      for (int x = 0; x < block_size.x; ++x) {
        const std::string xck = "xck" + std::to_string(x);
        std::string xc = is1x1 ? "min(" + dst_x[x] + ", src_size.x - 1)" : xck;
        std::string yc = is1x1 ? "min(" + dst_y[y] + ", src_size.y - 1)" : yck;
        std::string id = std::to_string(y) + std::to_string(x);
        c += "  int src_a_" + id + " = " + yc + " * src_size.x + " + xc + ";\n";
      }
    }
  }

  auto declare_src = [&]() {
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x; ++x) {
        const std::string id = std::to_string(y) + std::to_string(x);
        c += "    " + weights_data_type + " src" + id + ";\n";
      }
    }
  };
  const auto mode = TextureAddressMode::ZERO;
  const bool conditional_read = device.IsMali();
  auto read_src = [&]() {
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x; ++x) {
        if (buffer_type) {
          std::string id = std::to_string(y) + std::to_string(x);
          if (is1x1) {
            c += "    src" + id + " = " +
                 src_tensor.ReadAsType(conv_params.weights_data_type,
                                       "src_a_" + id) +
                 ";\n";
          } else {
            std::string condition =
                "mx" + std::to_string(x) + " && my" + std::to_string(y);
            if (conditional_read) {
              c += "    src" + id + " = " + condition + " ? " +
                   src_tensor.ReadAsType(conv_params.weights_data_type,
                                         "src_a_" + id) +
                   " : (FLT4)(0.0f);\n";
            } else {
              c += "    src" + id + " = " +
                   src_tensor.ReadAsType(conv_params.weights_data_type,
                                         "src_a_" + id) +
                   " * (FLT)(" + condition + ");\n";
            }
          }
          c += "    src_a_" + id + " += src_layer_offset;\n";
        } else {
          std::string id = std::to_string(y) + std::to_string(x);
          const std::string xc = is1x1 ? dst_x[x] : "xck" + std::to_string(x);
          const std::string yc = is1x1 ? dst_y[y] : "yck" + std::to_string(y);
          c += "    src" + id + " = " +
               src_tensor.ReadAsTypeWHS(conv_params.weights_data_type, xc, yc,
                                        "s", mode) +
               ";\n";
        }
      }
    }
  };
  const bool weights_type_as_accum_type =
      !(op_def.precision == CalculationsPrecision::F32_F16 &&
        conv_params.weights_data_type == DataType::FLOAT16);
  auto conv_core = [&](int shared_offset) {
    const std::string channels[] = {"x", "y", "z", "w"};
    for (int z = 0; z < block_size.z; ++z) {
      if (weights_type_as_accum_type) {
        for (int ch = 0; ch < 4; ++ch) {
          for (int y = 0; y < block_size.y; ++y) {
            for (int x = 0; x < block_size.x; ++x) {
              std::string id = std::to_string(y) + std::to_string(x);
              if (use_simd_broadcast) {
                int simd_id = (z * 4 + ch + shared_offset) / simd_size;
                int thread_id = (z * 4 + ch + shared_offset) % simd_size;
                std::string w_val_x = "sub_group_broadcast(simd_w" +
                                      std::to_string(simd_id) + ".x, " +
                                      std::to_string(thread_id) + "u)";
                std::string w_val_y = "sub_group_broadcast(simd_w" +
                                      std::to_string(simd_id) + ".y, " +
                                      std::to_string(thread_id) + "u)";
                std::string w_val_z = "sub_group_broadcast(simd_w" +
                                      std::to_string(simd_id) + ".z, " +
                                      std::to_string(thread_id) + "u)";
                std::string w_val_w = "sub_group_broadcast(simd_w" +
                                      std::to_string(simd_id) + ".w, " +
                                      std::to_string(thread_id) + "u)";
                c += "    r" + std::to_string(z) + id + ".x += " + w_val_x +
                     " * src" + id + "." + channels[ch] + ";\n";
                c += "    r" + std::to_string(z) + id + ".y += " + w_val_y +
                     " * src" + id + "." + channels[ch] + ";\n";
                c += "    r" + std::to_string(z) + id + ".z += " + w_val_z +
                     " * src" + id + "." + channels[ch] + ";\n";
                c += "    r" + std::to_string(z) + id + ".w += " + w_val_w +
                     " * src" + id + "." + channels[ch] + ";\n";
              } else {
                std::string w_val = "weights_cache[" +
                                    std::to_string(z * 4 + ch + shared_offset) +
                                    "]";
                c += "    r" + std::to_string(z) + id + " += " + w_val +
                     " * src" + id + "." + channels[ch] + ";\n";
              }
            }
          }
        }
      } else {  // F32_F16 precision and weights type is float16
        for (int y = 0; y < block_size.y; ++y) {
          for (int x = 0; x < block_size.x; ++x) {
            std::string id = std::to_string(y) + std::to_string(x);
            std::string R = "r" + std::to_string(z) + id;
            std::string S = "src" + id;
            const int dz = z * 4 + shared_offset;
            std::string f0 = "weights_cache[" + std::to_string(dz + 0) + "]";
            std::string f1 = "weights_cache[" + std::to_string(dz + 1) + "]";
            std::string f2 = "weights_cache[" + std::to_string(dz + 2) + "]";
            std::string f3 = "weights_cache[" + std::to_string(dz + 3) + "]";
            c += "    " + R + " += convert_float4(" + S + ".x * " + f0 + " + " +
                 S + ".y * " + f1 + " + " + S + ".z * " + f2 + " + " + S +
                 ".w * " + f3 + ");\n";
          }
        }
      }
    }
  };

  c += "  int s = 0;\n";
  c += "  do {\n";
  declare_src();
  const int total_work_items =
      work_group_size.x * work_group_size.y * work_group_size.z;
  if (conv_params.weights_upload_type ==
      ConvPowerVR::WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP) {
    c += GenerateAsyncUpload("weights_cache", "filters_loc",
                             /*global_offset_name*/ "", local_mem_size);
  } else if (conv_params.weights_upload_type ==
             ConvPowerVR::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    c += "    barrier(CLK_LOCAL_MEM_FENCE);\n";
    c += GenerateUploadByThreads("weights_cache", "filters_loc",
                                 /*global_offset_name*/ "", "lid",
                                 total_work_items, local_mem_size);
  } else if (use_simd_broadcast) {
    int parts = local_mem_size / simd_size;
    int reminder = local_mem_size % simd_size;
    for (int i = 0; i < parts; ++i) {
      c += "    FLT4 simd_w" + std::to_string(i) + " = filters_loc[simd_id + " +
           std::to_string(i * simd_size) + "];\n";
    }
    if (reminder) {
      c += "    FLT4 simd_w" + std::to_string(parts) + ";\n";
      c += "    if (simd_id < " + std::to_string(reminder) + ") {\n";
      c += "      simd_w" + std::to_string(parts) +
           " = filters_loc[simd_id + " + std::to_string(parts * simd_size) +
           "];\n";
      c += "    }\n";
    }
  } else {  // GLOBAL_MEM/CONSTANT_MEM
    c += "    weights_cache = filters_loc;\n";
  }
  read_src();
  c += "    s += 1;\n";
  if (conv_params.weights_upload_type ==
      ConvPowerVR::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    c += "    barrier(CLK_LOCAL_MEM_FENCE);\n";
  }
  conv_core(0);
  for (int i = 1; i < conv_params.src_depth_loop_size; ++i) {
    read_src();
    conv_core(i * block_size.z * 4);
    c += "    s += 1;\n";
  }
  c += "    filters_loc += " + std::to_string(local_mem_size) + ";\n";
  c += "  } while (s < src_size.z);\n";
  if (!is1x1) {
    c += "  };\n";
    c += "  };\n";
  }
  if (conv_params.weights_upload_type ==
      ConvPowerVR::WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP) {
    c += GenerateAsyncUpload("weights_cache", "biases", "Z", block_size.z);
  } else if (conv_params.weights_upload_type ==
             ConvPowerVR::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    c += "    barrier(CLK_LOCAL_MEM_FENCE);\n";
    c += GenerateUploadByThreads("weights_cache", "biases", "Z", "lid",
                                 total_work_items, block_size.z);
    c += "    barrier(CLK_LOCAL_MEM_FENCE);\n";
  } else {
    c += "    weights_cache = biases + Z;\n";
  }
  if (late_oob_check) {
    c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) {\n";
    c += "    return;\n";
    c += "  }\n";
  }
  for (int z = 0; z < block_size.z; ++z) {
    const std::string sz = std::to_string(z);
    c += "  if (Z + " + sz + " >= dst_size.z) return;\n";
    c += "  {\n";
    c += "    FLT4 bias_val = TO_FLT4(weights_cache[" + sz + "]);\n";
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x; ++x) {
        const std::string xs = dst_x[x];
        const std::string ys = dst_y[y];
        const std::string zs = "Z + " + sz;
        const std::string r_id = sz + std::to_string(y) + std::to_string(x);
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
        c += "    FLT4 res = TO_FLT4(r" + r_id + ") + bias_val;\n";
        const LinkingContext context{"res", xs, ys, zs};
        c += PostProcess(linked_operations, context);
        c += "    " + dst_tensor.WriteWHS("res", xs, ys, zs) + "\n";
        c += "  }\n";
      }
    }
    c += "  }\n";
  }
  c += "}\n";
  return c;
}

ConvPowerVR::ConvParams ConvPowerVR::GuessBestParams(
    const CLDevice& device, const OperationDef& definition, int src_depth,
    int dst_depth, bool x_kernel_is_1, bool y_kernel_is_1,
    bool different_weights_for_height, const BHWC* dst_shape) const {
  ConvParams conv_params;
  conv_params.linear_hw = false;
  conv_params.weights_data_type =
      DeduceDataTypeFromPrecision(definition.precision);
  conv_params.x_kernel_is_1 = x_kernel_is_1;
  conv_params.y_kernel_is_1 = y_kernel_is_1;
  conv_params.different_weights_for_height = different_weights_for_height;
  if (device.IsNvidia()) {
    if (different_weights_for_height) {
      conv_params.work_group_size = int3(32, 1, 1);
      conv_params.work_group_launch_order = int3(2, 0, 1);
      conv_params.fixed_work_group_size = true;
    } else {
      conv_params.linear_hw = true;
      conv_params.work_group_size = int3(32, 1, 1);
      conv_params.work_group_launch_order = int3(1, 0, 2);
      conv_params.fixed_work_group_size = true;
    }
    conv_params.block_size = int3(1, 1, 4);
    conv_params.src_depth_loop_size = 1;
    conv_params.weights_upload_type = WeightsUploadType::LOCAL_MEM_BY_THREADS;
    if (dst_depth % 4 == 0 || dst_depth >= 8) {
      conv_params.block_size.z = 4;
    } else if (dst_depth % 2 == 0 || dst_depth >= 4) {
      conv_params.block_size.z = 2;
    } else {
      conv_params.block_size.z = dst_depth;
    }
    if (src_depth % 2 == 0) {
      conv_params.src_depth_loop_size = 2;
    }
    if (src_depth % 4 == 0 && conv_params.block_size.z <= 2) {
      conv_params.src_depth_loop_size = 4;
    }
  } else if (device.IsPowerVR()) {
    if (different_weights_for_height) {
      conv_params.work_group_size = int3(32, 1, 1);
      conv_params.work_group_launch_order = int3(2, 0, 1);
      conv_params.fixed_work_group_size = true;
    } else {
      conv_params.linear_hw = true;
      conv_params.work_group_size = int3(32, 1, 1);
      conv_params.work_group_launch_order = int3(1, 0, 2);
      conv_params.fixed_work_group_size = true;
    }
    conv_params.weights_data_type =
        definition.precision == CalculationsPrecision::F16 ? DataType::FLOAT16
                                                           : DataType::FLOAT32;
    conv_params.block_size = int3(1, 1, 4);
    conv_params.src_depth_loop_size = 1;
    conv_params.weights_upload_type =
        WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP;
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
        if (src_depth % 2 == 0) {
          conv_params.src_depth_loop_size = 2;
        }
        if (src_depth % 4 == 0) {
          conv_params.src_depth_loop_size = 4;
        }
        if (src_depth <= 8) {
          conv_params.src_depth_loop_size = src_depth;
        }
      }
      conv_params.block_size.x = 2;
    }
  } else if (device.IsAMD()) {
    if (different_weights_for_height) {
      conv_params.work_group_size = int3(32, 1, 1);
      conv_params.work_group_launch_order = int3(2, 0, 1);
      conv_params.fixed_work_group_size = true;
    } else {
      conv_params.work_group_size = int3(8, 4, 1);
      conv_params.work_group_launch_order = int3(2, 0, 1);
      conv_params.fixed_work_group_size = true;
    }

    conv_params.block_size = int3(2, 1, 1);
    if (x_kernel_is_1 && y_kernel_is_1) {
      conv_params.block_size.y = 2;
    }
    conv_params.src_depth_loop_size = 1;
    conv_params.weights_upload_type = WeightsUploadType::CONSTANT_MEM;
    if (dst_depth % 8 == 0 || dst_depth >= 32) {
      conv_params.block_size.z = 8;
    } else if (dst_depth % 4 == 0 || dst_depth >= 8) {
      conv_params.block_size.z = 4;
    } else if (dst_depth % 2 == 0 || dst_depth >= 4) {
      conv_params.block_size.z = 2;
    } else {
      conv_params.block_size.z = 1;
    }
    if (src_depth % 2 == 0 && src_depth >= 16) {
      conv_params.src_depth_loop_size = 2;
    }
  } else if (device.IsMali()) {
    int block_size = 2;
    if (dst_shape) {
      int task_size = dst_shape->w * dst_shape->b * dst_shape->h * dst_depth;
      block_size = GetRecommendedBlockSizeForConv(device, definition.precision,
                                                  task_size);
    }
    if (!x_kernel_is_1 || !y_kernel_is_1) {
      block_size = std::min(block_size, 4);
    }
    if (block_size == 8) {
      if (dst_depth == 1 || dst_depth == 3) {
        conv_params.block_size = int3(2, 2, 1);
      } else {
        conv_params.block_size = int3(2, 2, 2);
      }
    } else if (block_size == 4) {
      if (dst_depth == 1 || dst_depth == 3) {
        conv_params.block_size = int3(2, 2, 1);
      } else {
        conv_params.block_size = int3(2, 1, 2);
      }
    } else if (block_size == 2) {
      conv_params.block_size = int3(2, 1, 1);
    } else {
      conv_params.block_size = int3(1, 1, 1);
    }
    conv_params.src_depth_loop_size = 1;
    MaliInfo mali_info = device.GetInfo().mali_info;
    if (src_depth % 2 == 0 && block_size <= 2 && !mali_info.IsMidgard()) {
      conv_params.src_depth_loop_size = 2;
    }
    if (src_depth % 4 == 0 && block_size == 1 && !mali_info.IsMidgard() &&
        definition.precision == CalculationsPrecision::F16) {
      conv_params.src_depth_loop_size = 4;
    }
    conv_params.work_group_size = int3(4, 4, 1);
    conv_params.work_group_launch_order = int3(0, 1, 2);
    conv_params.fixed_work_group_size = false;
    conv_params.weights_upload_type = WeightsUploadType::GLOBAL_MEM;
  } else if (device.IsAdreno()) {
    conv_params.block_size = int3(2, 2, 1);
    conv_params.work_group_size = int3(8, 2, 1);
    conv_params.work_group_launch_order = int3(0, 1, 2);
    conv_params.fixed_work_group_size = false;
    conv_params.src_depth_loop_size = 1;
    conv_params.weights_upload_type = WeightsUploadType::GLOBAL_MEM;
  } else if (device.IsIntel()) {
    if (different_weights_for_height) {
      conv_params.work_group_size = int3(16, 1, 1);
      conv_params.work_group_launch_order = int3(0, 1, 2);
      conv_params.fixed_work_group_size = true;
    } else {
      conv_params.linear_hw = true;
      conv_params.work_group_size = int3(16, 1, 1);
      conv_params.work_group_launch_order = int3(0, 1, 2);
      conv_params.fixed_work_group_size = true;
    }
    conv_params.block_size = int3(1, 1, 4);
    conv_params.src_depth_loop_size = 1;
    if (definition.precision != CalculationsPrecision::F32_F16 &&
        device.SupportsExtension("cl_khr_subgroups") &&
        device.SupportsExtension("cl_intel_required_subgroup_size") &&
        device.IsCL20OrHigher() && device.SupportsSubGroupWithSize(16)) {
      conv_params.weights_upload_type =
          WeightsUploadType::PRIVATE_MEM_SIMD16_BROADCAST;
    } else {
      conv_params.weights_upload_type = WeightsUploadType::LOCAL_MEM_BY_THREADS;
    }
    if (dst_depth % 4 == 0 || dst_depth >= 8) {
      conv_params.block_size.z = 4;
    } else if (dst_depth % 2 == 0 || dst_depth >= 4) {
      conv_params.block_size.z = 2;
    } else {
      conv_params.block_size.z = dst_depth;
    }
    if (src_depth % 2 == 0) {
      conv_params.src_depth_loop_size = 2;
    }
    if (src_depth % 4 == 0 && conv_params.block_size.z <= 2) {
      conv_params.src_depth_loop_size = 4;
    }
  } else {
    conv_params.block_size = int3(1, 1, 4);
    conv_params.work_group_size = int3(8, 2, 1);
    conv_params.work_group_launch_order = int3(0, 1, 2);
    conv_params.fixed_work_group_size = false;
    conv_params.src_depth_loop_size = 1;
    conv_params.weights_upload_type = WeightsUploadType::GLOBAL_MEM;
    if (dst_depth % 4 == 0 || dst_depth >= 8) {
      conv_params.block_size.z = 4;
    } else if (dst_depth % 2 == 0 || dst_depth >= 4) {
      conv_params.block_size.z = 2;
    } else {
      conv_params.block_size.z = dst_depth;
    }
    if (src_depth % 2 == 0) {
      conv_params.src_depth_loop_size = 2;
    }
    if (src_depth % 4 == 0 && conv_params.block_size.z <= 2) {
      conv_params.src_depth_loop_size = 4;
    }
  }

  return conv_params;
}

ConvPowerVR::ConvParams ConvPowerVR::GuessBestParams(
    const CLDevice& device, const OperationDef& definition,
    const Convolution2DAttributes& attr, const BHWC* dst_shape) const {
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  const bool x_kernel_is_1 = attr.weights.shape.w == 1 && attr.strides.w == 1 &&
                             attr.dilations.w == 1 &&
                             attr.padding.prepended.w == 0 &&
                             attr.padding.appended.w == 0;
  const bool y_kernel_is_1 = attr.weights.shape.h == 1 && attr.strides.h == 1 &&
                             attr.dilations.h == 1 &&
                             attr.padding.prepended.h == 0 &&
                             attr.padding.appended.h == 0;
  return GuessBestParams(device, definition, src_depth, dst_depth,
                         x_kernel_is_1, y_kernel_is_1, false, dst_shape);
}

ConvPowerVR::ConvParams ConvPowerVR::GuessBestParams(
    const CLDevice& device, const OperationDef& definition,
    const Convolution2DAttributes& attr, const BHWC& weights_shape,
    const BHWC* dst_shape) const {
  const int dst_depth = DivideRoundUp(weights_shape.b, 4);
  const int src_depth = DivideRoundUp(weights_shape.c, 4);
  const bool x_kernel_is_1 =
      weights_shape.w == 1 && attr.strides.w == 1 && attr.dilations.w == 1 &&
      attr.padding.prepended.w == 0 && attr.padding.appended.w == 0;
  const bool y_kernel_is_1 =
      weights_shape.h == 1 && attr.strides.h == 1 && attr.dilations.h == 1 &&
      attr.padding.prepended.h == 0 && attr.padding.appended.h == 0;
  return GuessBestParams(device, definition, src_depth, dst_depth,
                         x_kernel_is_1, y_kernel_is_1, false, dst_shape);
}

ConvPowerVR::ConvParams ConvPowerVR::GuessBestParams(
    const CLDevice& device, const OperationDef& definition,
    const FullyConnectedAttributes& attr, const BHWC* dst_shape) const {
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  ConvPowerVR::ConvParams params = GuessBestParams(
      device, definition, src_depth, dst_depth, true, true, false, dst_shape);
  params.work_group_size.x *= params.work_group_size.y;
  params.work_group_size.y = 1;
  params.block_size.x *= params.block_size.y;
  params.block_size.y = 1;
  return params;
}

ConvPowerVR::ConvParams ConvPowerVR::GuessBestParamsWinograd(
    const CLDevice& device, const OperationDef& definition,
    const Convolution2DAttributes& attr, const BHWC* dst_shape) const {
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  ConvPowerVR::ConvParams params = GuessBestParams(
      device, definition, src_depth, dst_depth, true, true, true, dst_shape);
  params.block_size.x *= params.block_size.y;
  params.block_size.y = 1;
  return params;
}

absl::Status CreateConvPowerVR(const CreationContext& creation_context,
                               const OperationDef& definition,
                               const Convolution2DAttributes& attr,
                               ConvPowerVR* result, const BHWC* dst_shape) {
  *result = ConvPowerVR(definition, attr, *creation_context.device, dst_shape);
  return result->UploadData(attr.weights, attr.bias, creation_context.context);
}

absl::Status CreateConvPowerVR(const CreationContext& creation_context,
                               const OperationDef& definition,
                               const FullyConnectedAttributes& attr,
                               ConvPowerVR* result, const BHWC* dst_shape) {
  *result = ConvPowerVR(definition, attr, *creation_context.device, dst_shape);
  return result->UploadData(attr.weights, attr.bias, creation_context.context);
}

absl::Status CreateConvPowerVRDynamicWeights(
    const CreationContext& creation_context, const OperationDef& definition,
    const Convolution2DAttributes& attr, const BHWC& weights_shape,
    ConvPowerVR* result, const BHWC* dst_shape) {
  *result = ConvPowerVR(definition, attr, weights_shape,
                        *creation_context.device, dst_shape);
  LinearStorageCreateInfo create_info;
  create_info.storage_type = LinearStorageType::BUFFER;
  create_info.data_type = result->conv_params_.weights_data_type;
  create_info.aligned_size = weights_shape.b;
  return CreateLinearStorage(create_info, attr.bias, creation_context.context,
                             &result->biases_);
}

absl::Status CreateConvPowerVRWino4x4To6x6(
    const CreationContext& creation_context, const OperationDef& definition,
    const Convolution2DAttributes& attr, ConvPowerVR* result,
    const BHWC* dst_shape) {
  *result = ConvPowerVR(definition);
  result->conv_params_ = result->GuessBestParamsWinograd(
      *creation_context.device, definition, attr, dst_shape);
  return result->UploadDataForWinograd4x4To6x6(
      attr.weights, *creation_context.device, creation_context.context);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
