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
}  // namespace

ConvPowerVR::ConvPowerVR(const OperationDef& definition,
                         const Convolution2DAttributes& attr,
                         const CLDevice& device)
    : GPUOperation(definition),
      stride_padding_(attr.strides.w, attr.strides.h, -attr.padding.prepended.w,
                      -attr.padding.prepended.h),
      kernel_dilation_(attr.weights.shape.w, attr.weights.shape.h,
                       attr.dilations.w, attr.dilations.h),
      conv_params_(GuessBestParams(device, definition, attr)) {}

ConvPowerVR::ConvPowerVR(const OperationDef& definition,
                         const FullyConnectedAttributes& attr,
                         const CLDevice& device)
    : GPUOperation(definition),
      stride_padding_(1, 1, 0, 0),
      kernel_dilation_(1, 1, 1, 1),
      conv_params_(GuessBestParams(device, definition, attr)) {}

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

Status ConvPowerVR::Compile(const CreationContext& creation_context) {
  const bool stride_correction =
      definition_.batch_support && stride_padding_.x != 1;
  const std::string code = GenerateConvPowerVR1x1(
      definition_, stride_correction, conv_params_, linked_operations_);
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
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  if (!conv_params_.x_kernel_is_1 || !conv_params_.y_kernel_is_1) {
    RETURN_IF_ERROR(kernel_.SetBytesAuto(
        int4(stride_padding_.x, stride_padding_.y,
             stride_padding_.z * src_[0]->Batch(), stride_padding_.w)));
    RETURN_IF_ERROR(kernel_.SetBytesAuto(
        int4(kernel_dilation_.x, kernel_dilation_.y,
             kernel_dilation_.z * src_[0]->Batch(), kernel_dilation_.w)));
  }
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWBatchedHDB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWBatchedHDB()));
  return OkStatus();
}

int3 ConvPowerVR::GetGridSize() const {
  const int grid_x = IntegralDivideRoundUp(dst_[0]->Width() * dst_[0]->Batch(),
                                           conv_params_.block_size.x);
  const int grid_y =
      IntegralDivideRoundUp(dst_[0]->Height(), conv_params_.block_size.y);
  const int grid_z =
      IntegralDivideRoundUp(dst_[0]->Depth(), conv_params_.block_size.z);
  int3 wg;
  wg.x = IntegralDivideRoundUp(grid_x, conv_params_.work_group_size.x);
  wg.y = IntegralDivideRoundUp(grid_y, conv_params_.work_group_size.y);
  wg.z = IntegralDivideRoundUp(grid_z, conv_params_.work_group_size.z);
  return int3(wg[conv_params_.work_group_launch_order[0]] *
                  conv_params_.work_group_size.x,
              wg[conv_params_.work_group_launch_order[1]] *
                  conv_params_.work_group_size.y,
              wg[conv_params_.work_group_launch_order[2]] *
                  conv_params_.work_group_size.z);
}

Status ConvPowerVR::Tune(const TuningParameters& params) {
  if (conv_params_.weights_upload_type ==
          WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP ||
      conv_params_.weights_upload_type ==
          WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    return OkStatus();
  }
  if (conv_params_.work_group_launch_order[0] == 0 &&
      conv_params_.work_group_launch_order[1] == 1 &&
      conv_params_.work_group_launch_order[2] == 2) {
    RETURN_IF_ERROR(BindArguments());
    return GetBestWorkGroupConv(params, kernel_, GetGridSize(),
                                &conv_params_.work_group_size);
  }
  return OkStatus();
}

Status ConvPowerVR::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(),
                                 conv_params_.work_group_size);
}

std::string GenerateConvPowerVR1x1(
    const OperationDef& op_def, bool stride_correction,
    const ConvPowerVR::ConvParams& conv_params,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  std::string c = GetCommonDefines(op_def.precision);
  TensorCodeGenerator src_tensor("src_data",
                                 {"src_size.x", "src_size.y", "src_size.z"},
                                 op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor("dst_data",
                                 {"dst_size.x", "dst_size.y", "dst_size.z"},
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

  const int3 work_group_size = conv_params.work_group_size;
  const int3 block_size = conv_params.block_size;
  if (need_local_mem) {  // we use fixed workgroup size when use local mem
    c += "__attribute__((reqd_work_group_size(" +
         std::to_string(work_group_size.x) + ", " +
         std::to_string(work_group_size.y) + ", " +
         std::to_string(work_group_size.z) + ")))\n";
  }
  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += "    __global ACCUM_FLT4* filters_buffer,    \n";
  c += "    __global ACCUM_FLT4* biases             \n";
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  if (!is1x1) {
    c += "    int4 stride_padding,           \n";
    c += "    int4 kernel_dilation,          \n";
  }
  c += "    int4 src_size,                   \n";
  c += "    int4 dst_size                    \n";
  c += ") {\n";
  int3 launch_remap;
  launch_remap[conv_params.work_group_launch_order.x] = 0;
  launch_remap[conv_params.work_group_launch_order.y] = 1;
  launch_remap[conv_params.work_group_launch_order.z] = 2;
  if (conv_params.work_group_launch_order[0] == 0) {
    c += "  int X = get_global_id(0) * " + std::to_string(block_size.x) + ";\n";
  } else {
    c += "  int X = (get_group_id(" + std::to_string(launch_remap[0]) +
         ") * get_local_size(0) + get_local_id(0)) * " +
         std::to_string(block_size.x) + ";\n";
  }
  if (conv_params.work_group_launch_order[1] == 1) {
    c += "  int Y = get_global_id(1) * " + std::to_string(block_size.y) + ";\n";
  } else {
    c += "  int Y = (get_group_id(" + std::to_string(launch_remap[1]) +
         ") * get_local_size(1) + get_local_id(1)) * " +
         std::to_string(block_size.y) + ";\n";
  }
  if (conv_params.work_group_launch_order[2] == 2) {
    c += "  int Z = get_global_id(2) * " + std::to_string(block_size.z) + ";\n";
  } else {
    c += "  int Z = (get_group_id(" + std::to_string(launch_remap[2]) +
         ") * get_local_size(2) + get_local_id(2)) * " +
         std::to_string(block_size.z) + ";\n";
  }
  if (!need_local_mem) {
    c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) {\n";
    c += "    return;\n";
    c += "  }\n";
  }
  if (conv_params.weights_upload_type ==
      ConvPowerVR::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    c += "  int lid = get_local_id(1) * " + std::to_string(work_group_size.x) +
         " + get_local_id(0);\n";
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
      const std::string xc = "(X + " + std::to_string(x) + ")";
      if (stride_correction) {
        c += "  int xc" + std::to_string(x) + " = " +
             GetXStrideCorrected(xc, "src_size.w", "stride_padding.x",
                                 "stride_padding.z") +
             ";\n";
      } else {
        c += "  int xc" + std::to_string(x) + " = " + xc +
             " * stride_padding.x + stride_padding.z;\n";
      }
    }
    for (int y = 0; y < block_size.y; ++y) {
      const std::string yc = "(Y + " + std::to_string(y) + ")";
      c += "  int yc" + std::to_string(y) + " = " + yc +
           " * stride_padding.y + stride_padding.w;\n";
    }
  }
  if (need_local_mem) {
    c += "  __local ACCUM_FLT4 weights_cache[" +
         std::to_string(block_size.z * 4 * conv_params.src_depth_loop_size) +
         "];\n";
  }
  if (conv_params.weights_upload_type ==
      ConvPowerVR::WeightsUploadType::GLOBAL_MEM) {
    c += "    __global ACCUM_FLT4* weights_cache;\n";
  }
  if (is1x1) {
    c += "  __global ACCUM_FLT4* filters_loc = filters_buffer + Z * 4 * "
         "src_size.z;\n";
  } else {
    c += "  __global ACCUM_FLT4* filters_loc = filters_buffer + Z * 4 * "
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
        std::string xc =
            is1x1 ? "min(X + " + std::to_string(x) + ", src_size.x - 1)" : xck;
        std::string yc =
            is1x1 ? "min(Y + " + std::to_string(y) + ", src_size.y - 1)" : yck;
        std::string id = std::to_string(y) + std::to_string(x);
        c += "  int src_a_" + id + " = " + yc + " * src_size.x + " + xc + ";\n";
      }
    }
  }

  auto declare_src = [&]() {
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x; ++x) {
        const std::string id = std::to_string(y) + std::to_string(x);
        if (op_def.precision == CalculationsPrecision::F32_F16) {
          c += "    ACCUM_FLT4 src" + id + ";\n";
        } else {
          c += "    FLT4 src" + id + ";\n";
        }
      }
    }
  };
  const auto mode = TextureAddressMode::ZERO;
  auto read_src = [&]() {
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x; ++x) {
        if (buffer_type) {
          std::string id = std::to_string(y) + std::to_string(x);
          std::string multiplier = is1x1
                                       ? ""
                                       : " * (FLT)(mx" + std::to_string(x) +
                                             " && my" + std::to_string(y) + ")";
          if (src_tensor_type == TensorStorageType::BUFFER) {
            if (op_def.precision == CalculationsPrecision::F32_F16) {
              c += "    src" + id + " = convert_float4(src_data[src_a_" + id +
                   "]" + multiplier + ");\n";
            } else {
              c += "    src" + id + " = src_data[src_a_" + id + "]" +
                   multiplier + ";\n";
            }
          }
          if (src_tensor_type == TensorStorageType::IMAGE_BUFFER) {
            if (op_def.precision == CalculationsPrecision::F32_F16) {
              c += "    src" + id + " = " +
                   src_tensor.ReadAsFloat("src_a_" + id) + multiplier + ";\n";
            } else {
              c += "    src" + id + " = " + src_tensor.Read("src_a_" + id) +
                   multiplier + ";\n";
            }
          }
          c += "    src_a_" + id + " += src_layer_offset;\n";
        } else {
          std::string id = std::to_string(y) + std::to_string(x);
          const std::string xc =
              is1x1 ? "X + " + std::to_string(x) : "xck" + std::to_string(x);
          const std::string yc =
              is1x1 ? "Y + " + std::to_string(y) : "yck" + std::to_string(y);
          if (op_def.precision == CalculationsPrecision::F32_F16) {
            c += "    src" + id + " = " +
                 src_tensor.ReadAsFloat3D(xc, yc, "s", mode) + ";\n";
          } else {
            c += "    src" + id + " = " + src_tensor.Read3D(xc, yc, "s", mode) +
                 ";\n";
          }
        }
      }
    }
  };
  auto conv_core = [&](int shared_offset) {
    const std::string channels[] = {"x", "y", "z", "w"};
    for (int z = 0; z < block_size.z; ++z) {
      for (int ch = 0; ch < 4; ++ch) {
        for (int y = 0; y < block_size.y; ++y) {
          for (int x = 0; x < block_size.x; ++x) {
            std::string id = std::to_string(y) + std::to_string(x);
            c += "    r" + std::to_string(z) + id + " += weights_cache[" +
                 std::to_string(z * 4 + ch + shared_offset) + "] * src" + id +
                 "." + channels[ch] + ";\n";
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
    c +=
        GenerateAsyncUpload("weights_cache", "filters_loc",
                            /*global_offset_name*/ "",
                            block_size.z * 4 * conv_params.src_depth_loop_size);
  } else if (conv_params.weights_upload_type ==
             ConvPowerVR::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    c += "    barrier(CLK_LOCAL_MEM_FENCE);\n";
    c += GenerateUploadByThreads(
        "weights_cache", "filters_loc",
        /*global_offset_name*/ "", "lid", total_work_items,
        block_size.z * 4 * conv_params.src_depth_loop_size);
  } else {  // GLOBAL_MEM
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
  c += "    filters_loc += " +
       std::to_string(block_size.z * 4 * conv_params.src_depth_loop_size) +
       ";\n";
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
  } else {  // GLOBAL_MEM
    c += "    weights_cache = biases + Z;\n";
  }
  if (need_local_mem) {
    c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) {\n";
    c += "    return;\n";
    c += "  }\n";
  }
  for (int z = 0; z < block_size.z; ++z) {
    c += "  if (Z + " + std::to_string(z) + " >= dst_size.z) return;\n";
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
        c += "    FLT4 res = TO_FLT4(r" + r_id + " + weights_cache[" +
             std::to_string(z) + "]);\n";
        const LinkingContext context{"res", xs, ys, zs};
        c += PostProcess(linked_operations, context);
        c += "    " + dst_tensor.Write3D("res", xs, ys, zs) + "\n";
        c += "  }\n";
      }
    }
  }
  c += "}\n";
  return c;
}

ConvPowerVR::ConvParams ConvPowerVR::GuessBestParams(
    const CLDevice& device, const OperationDef& definition, int src_depth,
    int dst_depth, bool x_kernel_is_1, bool y_kernel_is_1) const {
  ConvParams conv_params;
  conv_params.x_kernel_is_1 = x_kernel_is_1;
  conv_params.y_kernel_is_1 = y_kernel_is_1;
  if (device.IsNvidia()) {
    conv_params.block_size = int3(1, 1, 4);
    conv_params.work_group_size = int3(8, 4, 1);
    conv_params.work_group_launch_order = int3(2, 0, 1);
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
    conv_params.block_size = int3(1, 1, 4);
    conv_params.work_group_size = int3(8, 4, 1);
    conv_params.work_group_launch_order = int3(2, 0, 1);
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
      conv_params.work_group_size = int3(4, 8, 1);
    }
  } else {
    conv_params.block_size = int3(1, 1, 4);
    conv_params.work_group_size = int3(8, 4, 1);
    conv_params.work_group_launch_order = int3(0, 1, 2);
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
    const Convolution2DAttributes& attr) const {
  const int dst_depth = IntegralDivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
  const bool x_kernel_is_1 = attr.weights.shape.w == 1 && attr.strides.w == 1 &&
                             attr.dilations.w == 1 &&
                             attr.padding.prepended.w == 0 &&
                             attr.padding.appended.w == 0;
  const bool y_kernel_is_1 = attr.weights.shape.h == 1 && attr.strides.h == 1 &&
                             attr.dilations.h == 1 &&
                             attr.padding.prepended.h == 0 &&
                             attr.padding.appended.h == 0;
  return GuessBestParams(device, definition, src_depth, dst_depth,
                         x_kernel_is_1, y_kernel_is_1);
}

ConvPowerVR::ConvParams ConvPowerVR::GuessBestParams(
    const CLDevice& device, const OperationDef& definition,
    const FullyConnectedAttributes& attr) const {
  const int dst_depth = IntegralDivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
  ConvPowerVR::ConvParams params =
      GuessBestParams(device, definition, src_depth, dst_depth, true, true);
  params.work_group_size = int3(32, 1, 1);
  return params;
}

Status CreateConvPowerVR(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const Convolution2DAttributes& attr,
                         ConvPowerVR* result) {
  *result = ConvPowerVR(definition, attr, *creation_context.device);
  return result->UploadData(attr.weights, attr.bias, creation_context.context);
}

Status CreateConvPowerVR(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const FullyConnectedAttributes& attr,
                         ConvPowerVR* result) {
  *result = ConvPowerVR(definition, attr, *creation_context.device);
  return result->UploadData(attr.weights, attr.bias, creation_context.context);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
