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

#include "tensorflow/lite/delegates/gpu/common/tasks/conv_powervr.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

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

std::string GenerateBlockCoords(const int4& block_size,
                                const int3& work_group_launch_order,
                                bool linear_spatial, bool linear_all,
                                bool need_depth) {
  std::string c;
  int3 launch_remap;
  launch_remap[work_group_launch_order.x] = 0;
  launch_remap[work_group_launch_order.y] = 1;
  launch_remap[work_group_launch_order.z] = 2;
  if (linear_all) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int DST_S = (linear_id / args.task_size_spatial) * " +
         std::to_string(block_size.w) + ";\n";
    c += "  int linear_spatial = linear_id % args.task_size_spatial;\n";
    if (need_depth) {
      c += "  int DST_X = (linear_spatial % args.task_size_x) * " +
           std::to_string(block_size.x) + ";\n";
      c += "  linear_spatial = linear_spatial / args.task_size_x;\n";
      c += "  int DST_Y = (linear_spatial % args.task_size_y) * " +
           std::to_string(block_size.y) + ";\n";
      c += "  int DST_Z = (linear_spatial / args.task_size_y) * " +
           std::to_string(block_size.z) + ";\n";
    } else {
      c += "  int DST_Y = (linear_spatial / args.task_size_x) * " +
           std::to_string(block_size.y) + ";\n";
      c += "  int DST_X = (linear_spatial % args.task_size_x) * " +
           std::to_string(block_size.x) + ";\n";
    }
  } else if (linear_spatial) {
    if (work_group_launch_order[0] == 0) {
      c += "  int linear_spatial = GLOBAL_ID_0;\n";
    } else {
      c += "  int linear_spatial = GROUP_ID_" +
           std::to_string(launch_remap[0]) + " * GROUP_SIZE_0 + LOCAL_ID_0;\n";
    }
    if (need_depth) {
      c += "  int DST_X = (linear_spatial % args.task_size_x) * " +
           std::to_string(block_size.x) + ";\n";
      c += "  linear_spatial = linear_spatial / args.task_size_x;\n";
      c += "  int DST_Y = (linear_spatial % args.task_size_y) * " +
           std::to_string(block_size.y) + ";\n";
      c += "  int DST_Z = (linear_spatial / args.task_size_y) * " +
           std::to_string(block_size.z) + ";\n";
    } else {
      c += "  int DST_Y = (linear_spatial / args.task_size_x) * " +
           std::to_string(block_size.y) + ";\n";
      c += "  int DST_X = (linear_spatial % args.task_size_x) * " +
           std::to_string(block_size.x) + ";\n";
    }
    if (work_group_launch_order[1] == 1) {
      c +=
          "  int DST_S = GLOBAL_ID_1 * " + std::to_string(block_size.w) + ";\n";
    } else {
      c += "  int DST_S = (GROUP_ID_" + std::to_string(launch_remap[1]) +
           " * GROUP_SIZE_1 + LOCAL_ID_1) * " + std::to_string(block_size.w) +
           ";\n";
    }
  } else {
    if (work_group_launch_order[0] == 0) {
      c +=
          "  int DST_X = GLOBAL_ID_0 * " + std::to_string(block_size.x) + ";\n";
    } else {
      c += "  int DST_X = (GROUP_ID_" + std::to_string(launch_remap[0]) +
           " * GROUP_SIZE_0 + LOCAL_ID_0) * " + std::to_string(block_size.x) +
           ";\n";
    }
    std::string global_id_1;
    if (work_group_launch_order[1] == 1) {
      global_id_1 = "GLOBAL_ID_1";
    } else {
      global_id_1 = "(GROUP_ID_" + std::to_string(launch_remap[1]) +
                    " * GROUP_SIZE_1 + LOCAL_ID_1)";
    }
    if (need_depth) {
      c += "  int linear_id_1 = " + global_id_1 + ";\n";
      c += "  int DST_Z = (linear_id_1 / args.task_size_y) * " +
           std::to_string(block_size.z) + ";\n";
      c += "  int DST_Y = (linear_id_1 % args.task_size_y) * " +
           std::to_string(block_size.y) + ";\n";
    } else {
      c += "  int DST_Y = " + global_id_1 + " * " +
           std::to_string(block_size.y) + ";\n";
    }
    if (work_group_launch_order[2] == 2) {
      c +=
          "  int DST_S = GLOBAL_ID_2 * " + std::to_string(block_size.w) + ";\n";
    } else {
      c += "  int DST_S = (GROUP_ID_" + std::to_string(launch_remap[2]) +
           " * GROUP_SIZE_2 + LOCAL_ID_2) * " + std::to_string(block_size.w) +
           ";\n";
    }
  }

  return c;
}
}  // namespace

ConvPowerVR::ConvPowerVR(const OperationDef& definition,
                         const Convolution2DAttributes& attr,
                         const GpuInfo& gpu_info, const BHWC* dst_shape)
    : GPUOperation(definition),
      stride_(attr.strides.w, attr.strides.h, 1, 1),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h, 0, 0),
      kernel_size_(attr.weights.shape.w, attr.weights.shape.h, 1, 1),
      dilation_(attr.dilations.w, attr.dilations.h, 1, 1),
      conv_params_(GuessBestParams(gpu_info, definition, attr, dst_shape)) {}

ConvPowerVR::ConvPowerVR(const OperationDef& definition,
                         const Convolution2DAttributes& attr,
                         const BHWC& weights_shape, const GpuInfo& gpu_info,
                         const BHWC* dst_shape)
    : GPUOperation(definition),
      stride_(attr.strides.w, attr.strides.h, 1, 1),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h, 0, 0),
      kernel_size_(weights_shape.w, weights_shape.h, 1, 1),
      dilation_(attr.dilations.w, attr.dilations.h, 1, 1),
      conv_params_(GuessBestParams(gpu_info, definition, attr, weights_shape,
                                   dst_shape)) {}

ConvPowerVR::ConvPowerVR(const OperationDef& definition,
                         const FullyConnectedAttributes& attr,
                         const GpuInfo& gpu_info, const BHWC* dst_shape)
    : GPUOperation(definition),
      stride_(1, 1, 1, 1),
      padding_(0, 0, 0, 0),
      kernel_size_(1, 1, 1, 1),
      dilation_(1, 1, 1, 1),
      conv_params_(GuessBestParams(gpu_info, definition, attr, dst_shape)) {}

ConvPowerVR::ConvPowerVR(const OperationDef& definition)
    : GPUOperation(definition),
      stride_(1, 1, 1, 1),
      padding_(0, 0, 0, 0),
      kernel_size_(1, 1, 1, 1),
      dilation_(1, 1, 1, 1) {}

ConvPowerVR::ConvPowerVR(ConvPowerVR&& operation)
    : GPUOperation(std::move(operation)),
      stride_(operation.stride_),
      padding_(operation.padding_),
      kernel_size_(operation.kernel_size_),
      dilation_(operation.dilation_),
      conv_params_(operation.conv_params_) {}

ConvPowerVR::ConvPowerVR(const OperationDef& definition,
                         const Convolution3DAttributes& attr,
                         const GpuInfo& gpu_info, const BHWDC* dst_shape)
    : GPUOperation(definition),
      stride_(attr.strides.w, attr.strides.h, attr.strides.d, 1),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h,
               -attr.padding.prepended.d, 0),
      kernel_size_(attr.weights.shape.w, attr.weights.shape.h,
                   attr.weights.shape.d, 1),
      dilation_(attr.dilations.w, attr.dilations.h, attr.dilations.d, 1),
      conv_params_(GuessBestParams(gpu_info, definition, attr, dst_shape)) {}

ConvPowerVR& ConvPowerVR::operator=(ConvPowerVR&& operation) {
  if (this != &operation) {
    std::swap(stride_, operation.stride_);
    std::swap(padding_, operation.padding_);
    std::swap(kernel_size_, operation.kernel_size_);
    std::swap(dilation_, operation.dilation_);
    std::swap(conv_params_, operation.conv_params_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

void ConvPowerVR::GenerateCode(const GpuInfo& gpu_info) {
  if (conv_params_.linear_all) {
    grid_dimension_ = 1;
  } else if (conv_params_.linear_spatial) {
    grid_dimension_ = 2;
  }
  const bool stride_correction =
      definition_.IsBatchSupported() && stride_.x != 1;
  code_ = GenerateConv(gpu_info, definition_, stride_correction, conv_params_);
  if (definition_.precision == CalculationsPrecision::F16 &&
      gpu_info.IsPowerVR()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  if (gpu_info.IsMali()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  if (conv_params_.IsPrivateMemBroadcast() && gpu_info.IsCL20OrHigher()) {
    compiler_options_.push_back(CompilerOptions::kCl20);
  }
  bool kernel_is_trivial =
      conv_params_.x_kernel_is_1 && conv_params_.y_kernel_is_1;
  if (definition_.src_tensors[0].HasAxis(Axis::DEPTH)) {
    kernel_is_trivial = kernel_is_trivial & conv_params_.z_kernel_is_1;
  }
  if (gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno3xx() &&
      definition_.precision == CalculationsPrecision::F16 &&
      kernel_is_trivial) {
    compiler_options_.push_back(CompilerOptions::kAdrenoFullSimd);
  }
}

absl::Status ConvPowerVR::BindArguments(ArgumentsBinder* args) {
  if (!conv_params_.x_kernel_is_1) {
    RETURN_IF_ERROR(args->SetInt("stride_x", stride_.x));
    RETURN_IF_ERROR(args->SetInt("padding_x", padding_.x * src_[0]->Batch()));
    RETURN_IF_ERROR(args->SetInt("kernel_size_x", kernel_size_.x));
    RETURN_IF_ERROR(args->SetInt("dilation_x", dilation_.x * src_[0]->Batch()));
  }
  if (!conv_params_.y_kernel_is_1) {
    RETURN_IF_ERROR(args->SetInt("stride_y", stride_.y));
    RETURN_IF_ERROR(args->SetInt("padding_y", padding_.y));
    RETURN_IF_ERROR(args->SetInt("kernel_size_y", kernel_size_.y));
    RETURN_IF_ERROR(args->SetInt("dilation_y", dilation_.y));
  }
  if (definition_.src_tensors[0].HasAxis(Axis::DEPTH) &&
      !conv_params_.z_kernel_is_1) {
    RETURN_IF_ERROR(args->SetInt("stride_z", stride_.z));
    RETURN_IF_ERROR(args->SetInt("padding_z", padding_.z));
    RETURN_IF_ERROR(args->SetInt("kernel_size_z", kernel_size_.z));
    RETURN_IF_ERROR(args->SetInt("dilation_z", dilation_.z));
  }
  const int task_size_x = DivideRoundUp(dst_[0]->Width() * dst_[0]->Batch(),
                                        conv_params_.block_size.x);
  const int task_size_y =
      DivideRoundUp(dst_[0]->Height(), conv_params_.block_size.y);
  const int task_size_z =
      DivideRoundUp(dst_[0]->Depth(), conv_params_.block_size.z);
  RETURN_IF_ERROR(args->SetInt("task_size_x", task_size_x));
  RETURN_IF_ERROR(args->SetInt("task_size_y", task_size_y));
  const int task_size_spatial = task_size_x * task_size_y * task_size_z;
  RETURN_IF_ERROR(args->SetInt("task_size_spatial", task_size_spatial));
  return absl::OkStatus();
}

int3 ConvPowerVR::GetGridSize() const {
  const int task_size_x = DivideRoundUp(dst_[0]->Width() * dst_[0]->Batch(),
                                        conv_params_.block_size.x);
  const int task_size_y =
      DivideRoundUp(dst_[0]->Height(), conv_params_.block_size.y);
  const int task_size_z =
      DivideRoundUp(dst_[0]->Depth(), conv_params_.block_size.z);
  const int task_size_s =
      DivideRoundUp(dst_[0]->Slices(), conv_params_.block_size.w);
  int3 wg;

  if (conv_params_.linear_all) {
    return int3(task_size_x * task_size_y * task_size_z * task_size_s, 1, 1);
  } else if (conv_params_.linear_spatial) {
    return int3(task_size_x * task_size_y * task_size_z, task_size_s, 1);
  } else {
    return int3(task_size_x, task_size_y * task_size_z, task_size_s);
  }
}

void ConvPowerVR::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info, std::vector<int3>* work_groups) const {
  if (conv_params_.weights_upload_type ==
          WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP ||
      conv_params_.weights_upload_type ==
          WeightsUploadType::LOCAL_MEM_BY_THREADS ||
      conv_params_.fixed_work_group_size) {
    work_groups->push_back(work_group_size_);
    return;
  }
  GetPossibleWorkGroupsConv(tuning_type, gpu_info, kernel_info, grid_size_,
                            work_groups);
}

std::string ConvPowerVR::GenerateConv(const GpuInfo& gpu_info,
                                      const OperationDef& op_def,
                                      bool stride_correction,
                                      const ConvParams& conv_params) {
  auto src_desc = op_def.src_tensors[0];
  src_desc.SetAddressMode(AddressMode::kZero);
  if (op_def.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  AddSrcTensor("src_tensor", src_desc);
  if (op_def.src_tensors.size() == 2) {
    // dynamic weights
    BufferDescriptor desc;
    desc.element_type = op_def.src_tensors[1].data_type;
    desc.element_size = 4;
    desc.memory_type = conv_params.weights_upload_type ==
                               ConvPowerVR::WeightsUploadType::CONSTANT_MEM
                           ? MemoryType::CONSTANT
                           : MemoryType::GLOBAL;

    AddSrcBuffer("weights", desc);
  }

  const auto& src_def = op_def.src_tensors[0];

  auto generate_id = [&](const std::string& x, const std::string& y,
                         const std::string& z) {
    std::string id;
    if (src_def.HasAxis(Axis::WIDTH)) {
      id += "_w" + x;
    }
    if (src_def.HasAxis(Axis::HEIGHT)) {
      id += "_h" + y;
    }
    if (src_def.HasAxis(Axis::DEPTH)) {
      id += "_d" + z;
    }
    return id;
  };

  auto generate_id_full = [&](const std::string& x, const std::string& y,
                              const std::string& z, const std::string& s) {
    return generate_id(x, y, z) + "_s" + s;
  };

  auto generate_check = [&](const std::string& x, const std::string& y,
                            const std::string& z) {
    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH};
    const std::vector<std::string> names{"in_x", "in_y", "in_z"};
    const std::vector<bool> is_1{conv_params_.x_kernel_is_1,
                                 conv_params_.y_kernel_is_1,
                                 conv_params_.z_kernel_is_1};
    const std::vector<std::string> coords{x, y, z};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_def.HasAxis(axis) && !src_def.SupportsZeroClamp(axis) &&
          !is_1[i]) {
        if (!check.empty()) {
          check += " && ";
        }
        check += names[i] + coords[i];
      }
    }
    return check;
  };

  auto dst_desc = op_def.dst_tensors[0];
  if (op_def.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  AddDstTensor("dst_tensor", dst_desc);

  if (!conv_params_.x_kernel_is_1) {
    args_.AddInt("stride_x");
    args_.AddInt("padding_x");
    args_.AddInt("kernel_size_x");
    args_.AddInt("dilation_x");
  }
  if (!conv_params_.y_kernel_is_1) {
    args_.AddInt("stride_y");
    args_.AddInt("padding_y");
    args_.AddInt("kernel_size_y");
    args_.AddInt("dilation_y");
  }
  if (src_def.HasAxis(Axis::DEPTH) && !conv_params_.z_kernel_is_1) {
    args_.AddInt("stride_z");
    args_.AddInt("padding_z");
    args_.AddInt("kernel_size_z");
    args_.AddInt("dilation_z");
  }
  args_.AddInt("task_size_x");
  args_.AddInt("task_size_y");
  args_.AddInt("task_size_spatial");

  const int wg_total_size =
      work_group_size_.x * work_group_size_.y * work_group_size_.z;
  const std::string barrier =
      wg_total_size == 32 && gpu_info.IsWaveSizeEqualTo32()
          ? "SIMD_LOCAL_MEM_BARRIER"
          : "LOCAL_MEM_BARRIER";

  const bool need_local_mem =
      conv_params.weights_upload_type ==
          ConvPowerVR::WeightsUploadType::LOCAL_MEM_BY_THREADS ||
      conv_params.weights_upload_type ==
          ConvPowerVR::WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP;

  const int local_mem_size =
      conv_params.block_size.w * 4 * conv_params.src_depth_loop_size;

  const bool use_simd_broadcast = conv_params.IsPrivateMemBroadcast();
  const int simd_size = conv_params.simd_size;

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

  std::string c;
  if (use_simd_broadcast && gpu_info.IsApiOpenCl()) {
    if (gpu_info.opencl_info.cl_version == OpenClVersion::kCl2_0) {
      c += "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n";
    } else if (gpu_info.SupportsExtension("cl_intel_subgroups")) {
      c += "#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n";
    }
  }
  const int4 block_size = conv_params.block_size;
  if (conv_params.fixed_work_group_size && gpu_info.IsApiOpenCl()) {
    c += "__attribute__((reqd_work_group_size(" +
         std::to_string(work_group_size_.x) + ", " +
         std::to_string(work_group_size_.y) + ", " +
         std::to_string(work_group_size_.z) + ")))\n";
  }
  if (use_simd_broadcast && gpu_info.IsIntel() && gpu_info.IsApiOpenCl()) {
    c += "__attribute__((intel_reqd_sub_group_size(" +
         std::to_string(simd_size) + ")))\n";
  }
  std::string dst_oob_check;
  if (src_def.HasAxis(Axis::DEPTH)) {
    if (conv_params.linear_all) {
      dst_oob_check = "DST_S >= args.dst_tensor.Slices()";
    } else if (conv_params.linear_spatial) {
      dst_oob_check =
          "DST_Z >= args.dst_tensor.Depth() || DST_S >= "
          "args.dst_tensor.Slices()";
    } else {
      dst_oob_check =
          "DST_X >= args.dst_tensor.Width() || DST_Z >= "
          "args.dst_tensor.Depth() || DST_S >= args.dst_tensor.Slices()";
    }
  } else {
    if (conv_params.linear_all) {
      dst_oob_check = "DST_S >= args.dst_tensor.Slices()";
    } else if (conv_params.linear_spatial) {
      dst_oob_check =
          "DST_Y >= args.dst_tensor.Height() || DST_S >= "
          "args.dst_tensor.Slices()";
    } else {
      dst_oob_check =
          "DST_X >= args.dst_tensor.Width() || DST_Y >= "
          "args.dst_tensor.Height() || DST_S >= args.dst_tensor.Slices()";
    }
  }
  c += "MAIN_FUNCTION($0) {\n";
  c += GenerateBlockCoords(conv_params.block_size, work_group_launch_order_,
                           conv_params.linear_spatial, conv_params.linear_all,
                           src_def.HasAxis(Axis::DEPTH));
  if (!late_oob_check) {
    c += "  if (" + dst_oob_check + ") {\n";
    c += "    return;\n";
    c += "  }\n";
  }
  if (conv_params.weights_upload_type ==
      ConvPowerVR::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    if (conv_params.linear_spatial) {
      c += "  int lid = LOCAL_ID_0;\n";
    } else {
      c += "  int lid = LOCAL_ID_1 * " + std::to_string(work_group_size_.x) +
           " + LOCAL_ID_0;\n";
    }
  }
  if (use_simd_broadcast) {
    c += "  int simd_id = SUB_GROUP_LOCAL_ID;\n";
  }
  for (int s = 0; s < block_size.w; ++s) {
    const std::string sind = std::to_string(s);
    for (int z = 0; z < block_size.z; ++z) {
      const std::string zind = std::to_string(z);
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        for (int x = 0; x < block_size.x; ++x) {
          const std::string xind = std::to_string(x);
          c += "  ACCUM_FLT4 r" + generate_id_full(xind, yind, zind, sind) +
               " = INIT_ACCUM_FLT4(0.0f);\n";
        }
      }
    }
  }
  if (!conv_params_.x_kernel_is_1) {
    for (int x = 0; x < block_size.x; ++x) {
      const std::string xind = std::to_string(x);
      const std::string xc = "(DST_X + " + xind + ")";
      if (stride_correction) {
        c += "  int xc" + xind + " = " +
             GetXStrideCorrected(xc, "args.src_tensor.Batch()", "args.stride_x",
                                 "args.padding_x") +
             ";\n";
      } else {
        c += "  int xc" + xind + " = " + xc +
             " * args.stride_x + args.padding_x;\n";
      }
    }
  } else {
    for (int x = 0; x < block_size.x; ++x) {
      const std::string xind = std::to_string(x);
      c += "  int xc" + xind + " = DST_X + " + xind + ";\n";
      if (!src_def.CanReadOutOfBorder(Axis::WIDTH)) {
        c += "  xc" + xind + " = clamp(xc" + xind +
             ", 0, args.src_tensor.Width() - 1);\n";
      }
    }
  }
  if (!conv_params_.y_kernel_is_1) {
    for (int y = 0; y < block_size.y; ++y) {
      const std::string yind = std::to_string(y);
      const std::string yc = "(DST_Y + " + yind + ")";
      c += "  int yc" + yind + " = " + yc +
           " * args.stride_y + args.padding_y;\n";
    }
  } else {
    for (int y = 0; y < block_size.y; ++y) {
      const std::string yind = std::to_string(y);
      c += "  int yc" + yind + " = DST_Y + " + yind + ";\n";
      if (!src_def.CanReadOutOfBorder(Axis::HEIGHT)) {
        c += "  yc" + yind + " = clamp(yc" + yind +
             ", 0, args.src_tensor.Height() - 1);\n";
      }
    }
  }
  if (src_def.HasAxis(Axis::DEPTH)) {
    if (!conv_params_.z_kernel_is_1) {
      for (int z = 0; z < block_size.z; ++z) {
        const std::string zind = std::to_string(z);
        const std::string zc = "(DST_Z + " + zind + ")";
        c += "  int zc" + zind + " = " + zc +
             " * args.stride_z + args.padding_z;\n";
      }
    } else {
      for (int z = 0; z < block_size.z; ++z) {
        const std::string zind = std::to_string(z);
        c += "  int zc" + zind + " = DST_Z + " + zind + ";\n";
        if (!src_def.CanReadOutOfBorder(Axis::DEPTH)) {
          c += "  zc" + zind + " = clamp(zc" + zind +
               ", 0, args.src_tensor.Depth() - 1);\n";
        }
      }
    }
  }
  bool trivial_kernel_size =
      conv_params_.x_kernel_is_1 && conv_params_.y_kernel_is_1;
  if (src_def.HasAxis(Axis::DEPTH)) {
    trivial_kernel_size = trivial_kernel_size && conv_params_.z_kernel_is_1;
  }
  if (need_local_mem) {
    c += "  __local " + weights_data_type + " weights_cache[" +
         std::to_string(local_mem_size) + "];\n";
  } else if (conv_params.AreWeightsBuffer() &&
             gpu_info.SupportsPointersInKernels()) {
    c += "    " + weights_global_ptr + " weights_cache;\n";
  } else if (!trivial_kernel_size) {
    c += "  int filter_offset = 0;\n";
  }
  if (conv_params.AreWeightsBuffer()) {
    std::string offset;
    if (conv_params.different_weights_for_height) {
      offset = "(DST_S * args.src_tensor.Height() + DST_Y * " +
               std::to_string(block_size.w) +
               ") * 4 * args.src_tensor.Slices()";
    } else {
      std::string kernel_spatial_offset = "";
      if (!conv_params_.x_kernel_is_1) {
        kernel_spatial_offset += " * args.kernel_size_x";
      }
      if (!conv_params_.y_kernel_is_1) {
        kernel_spatial_offset += " * args.kernel_size_y";
      }
      if (src_def.HasAxis(Axis::DEPTH) && !conv_params_.z_kernel_is_1) {
        kernel_spatial_offset += " * args.kernel_size_z";
      }
      offset = "DST_S * 4 * args.src_tensor.Slices()" + kernel_spatial_offset;
    }
    if (gpu_info.SupportsPointersInKernels()) {
      c += "  " + weights_global_ptr +
           " filters_loc = args.weights.GetPtr() + " + offset + ";\n";
    } else {
      c += "  int filters_offset = " + offset + ";\n";
    }
  }
  if (src_def.HasAxis(Axis::DEPTH) && !conv_params_.z_kernel_is_1) {
    c += "  for (int kz = 0; kz < args.kernel_size_z; ++kz) {\n";
    for (int z = 0; z < block_size.z; ++z) {
      const std::string zck = "zck" + std::to_string(z);
      c += "  int zck" + std::to_string(z) + " = kz * args.dilation_z + zc" +
           std::to_string(z) + ";\n";
      if (!src_def.SupportsZeroClamp(Axis::DEPTH)) {
        c += "  bool in_z" + std::to_string(z) + " = " + zck + " >= 0 && " +
             zck + " < args.src_tensor.Depth();\n";
        if (!src_def.CanReadOutOfBorder(Axis::DEPTH)) {
          c += "  " + zck + " = clamp(" + zck +
               ", 0, args.src_tensor.Depth() - 1);\n";
        }
      }
    }
  }
  if (!conv_params_.y_kernel_is_1) {
    c += "  for (int ky = 0; ky < args.kernel_size_y; ++ky) {\n";
    for (int y = 0; y < block_size.y; ++y) {
      const std::string yck = "yck" + std::to_string(y);
      c += "  int " + yck + " = ky * args.dilation_y + yc" + std::to_string(y) +
           ";\n";
      if (!src_def.SupportsZeroClamp(Axis::HEIGHT)) {
        c += "  bool in_y" + std::to_string(y) + " = " + yck + " >= 0 && " +
             yck + " < args.src_tensor.Height();\n";
        if (!src_def.CanReadOutOfBorder(Axis::HEIGHT)) {
          c += "  " + yck + " = clamp(" + yck +
               ", 0, args.src_tensor.Height() - 1);\n";
        }
      }
    }
  }
  if (!conv_params_.x_kernel_is_1) {
    c += "  for (int kx = 0; kx < args.kernel_size_x; ++kx) {\n";
    for (int x = 0; x < block_size.x; ++x) {
      const std::string xck = "xck" + std::to_string(x);
      c += "  int xck" + std::to_string(x) + " = kx * args.dilation_x + xc" +
           std::to_string(x) + ";\n";
      if (!src_def.SupportsZeroClamp(Axis::WIDTH)) {
        c += "  bool in_x" + std::to_string(x) + " = " + xck + " >= 0 && " +
             xck + " < args.src_tensor.Width();\n";
        if (!src_def.CanReadOutOfBorder(Axis::WIDTH)) {
          c += "  " + xck + " = clamp(" + xck +
               ", 0, args.src_tensor.Width() - 1);\n";
        }
      }
    }
  }
  const bool need_multiple_slice_strides =
      src_def.ReturnsZeroForNegOneRead() && !trivial_kernel_size;
  for (int z = 0; z < block_size.z; ++z) {
    const std::string zind = std::to_string(z);
    for (int y = 0; y < block_size.y; ++y) {
      const std::string yind = std::to_string(y);
      for (int x = 0; x < block_size.x; ++x) {
        const std::string xind = std::to_string(x);
        std::string xc = conv_params.x_kernel_is_1 ? "xc" + xind : "xck" + xind;
        std::string yc = conv_params.y_kernel_is_1 ? "yc" + yind : "yck" + yind;
        const std::string id = generate_id(xind, yind, zind);
        std::string coords = "" + xc + ", " + yc;
        if (src_def.HasAxis(Axis::DEPTH)) {
          std::string zc =
              conv_params.z_kernel_is_1 ? "zc" + zind : "zck" + zind;
          coords += ", " + zc;
        }
        if (src_def.IsLinear()) {
          c += "  args.src_tensor.GetAddress(addr" + id + ", " + coords +
               ", 0);\n";
          if (need_multiple_slice_strides) {
            const std::string check = generate_check(xind, yind, zind);
            c += "  addr" + id + " = select(-1, addr" + id + ", (" + check +
                 "));\n";
            c += "  int ds" + id +
                 " = select(0, args.src_tensor.SliceStride(), (" + check +
                 "));\n";
          }
        }
      }
    }
  }
  if (src_def.IsLinear() && !need_multiple_slice_strides) {
    c += "  int ds = args.src_tensor.SliceStride();\n";
  }

  auto declare_src = [&]() {
    for (int z = 0; z < block_size.z; ++z) {
      const std::string zind = std::to_string(z);
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        for (int x = 0; x < block_size.x; ++x) {
          const std::string xind = std::to_string(x);
          const std::string id = generate_id(xind, yind, zind);
          c += "    " + weights_data_type + " src" + id + ";\n";
        }
      }
    }
  };
  const bool conditional_read = gpu_info.IsMali();
  auto read_src = [&]() {
    const std::string cl_type = ToCLDataType(conv_params.weights_data_type);
    for (int z = 0; z < block_size.z; ++z) {
      const std::string zind = std::to_string(z);
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        for (int x = 0; x < block_size.x; ++x) {
          const std::string xind = std::to_string(x);
          std::string id = generate_id(xind, yind, zind);
          const std::string check = generate_check(xind, yind, zind);
          std::string address;
          if (src_def.IsLinear()) {
            address = "addr" + id;
          } else {
            std::string xc =
                conv_params.x_kernel_is_1 ? "xc" + xind : "xck" + xind;
            std::string yc =
                conv_params.y_kernel_is_1 ? "yc" + yind : "yck" + yind;
            address = "" + xc + ", " + yc;
            if (src_def.HasAxis(Axis::DEPTH)) {
              std::string zc =
                  conv_params.z_kernel_is_1 ? "zc" + zind : "zck" + zind;
              address += ", " + zc;
            }
            address += ", s";
          }
          if (src_def.ReturnsZeroForNegOneRead()) {
            c += "    src" + id + " = args.src_tensor.Read<" + cl_type + ">(" +
                 address + ");\n";
            const std::string ds = trivial_kernel_size ? "ds" : "ds" + id;
            c += "    " + address + " += " + ds + ";\n";
          } else {
            if (!check.empty()) {
              if (conditional_read) {
                c += "    src" + id + " = " + check +
                     " ? args.src_tensor.Read<" + cl_type + ">(" + address +
                     ") : INIT_FLT4(0.0f);\n";
              } else {
                c += "    src" + id + " = args.src_tensor.Read<" + cl_type +
                     ">(" + address + ") * INIT_FLT(" + check + ");\n";
              }
            } else {
              c += "    src" + id + " = args.src_tensor.Read<" + cl_type +
                   ">(" + address + ");\n";
            }
            if (src_def.IsLinear()) {
              c += "    " + address + " += ds;\n";
            }
          }
        }
      }
    }
  };
  const bool weights_type_as_accum_type =
      !(op_def.precision == CalculationsPrecision::F32_F16 &&
        conv_params.weights_data_type == DataType::FLOAT16);
  auto conv_core = [&](int shared_offset) {
    const std::string channels[] = {"x", "y", "z", "w"};
    for (int s = 0; s < block_size.w; ++s) {
      const std::string sind = std::to_string(s);
      if (weights_type_as_accum_type) {
        for (int ch = 0; ch < 4; ++ch) {
          for (int z = 0; z < block_size.z; ++z) {
            const std::string zind = std::to_string(z);
            for (int y = 0; y < block_size.y; ++y) {
              const std::string yind = std::to_string(y);
              for (int x = 0; x < block_size.x; ++x) {
                const std::string xind = std::to_string(x);
                std::string R = "r" + generate_id_full(xind, yind, zind, sind);
                std::string S = "src" + generate_id(xind, yind, zind);
                if (use_simd_broadcast) {
                  int simd_id = (s * 4 + ch + shared_offset) / simd_size;
                  int thread_id = (s * 4 + ch + shared_offset) % simd_size;
                  std::string w_val_x = "SUB_GROUP_BROADCAST(simd_w" +
                                        std::to_string(simd_id) + ".x, " +
                                        std::to_string(thread_id) + "u)";
                  std::string w_val_y = "SUB_GROUP_BROADCAST(simd_w" +
                                        std::to_string(simd_id) + ".y, " +
                                        std::to_string(thread_id) + "u)";
                  std::string w_val_z = "SUB_GROUP_BROADCAST(simd_w" +
                                        std::to_string(simd_id) + ".z, " +
                                        std::to_string(thread_id) + "u)";
                  std::string w_val_w = "SUB_GROUP_BROADCAST(simd_w" +
                                        std::to_string(simd_id) + ".w, " +
                                        std::to_string(thread_id) + "u)";
                  if (GetWeightsDescription().IsI4O4()) {
                    c += "    " + R + ".x += " + w_val_x + " * " + S + "." +
                         channels[ch] + ";\n";
                    c += "    " + R + ".y += " + w_val_y + " * " + S + "." +
                         channels[ch] + ";\n";
                    c += "    " + R + ".z += " + w_val_z + " * " + S + "." +
                         channels[ch] + ";\n";
                    c += "    " + R + ".w += " + w_val_w + " * " + S + "." +
                         channels[ch] + ";\n";
                  } else {
                    c += "    " + R + "." + channels[ch] + " += " + w_val_x +
                         " * " + S + ".x;\n";
                    c += "    " + R + "." + channels[ch] + " += " + w_val_y +
                         " * " + S + ".y;\n";
                    c += "    " + R + "." + channels[ch] + " += " + w_val_z +
                         " * " + S + ".z;\n";
                    c += "    " + R + "." + channels[ch] + " += " + w_val_w +
                         " * " + S + ".w;\n";
                  }
                } else {
                  const std::string weight_id =
                      std::to_string(s * 4 + ch + shared_offset);
                  std::string w_val;
                  if (conv_params.AreWeightsBuffer()) {
                    if (gpu_info.SupportsPointersInKernels()) {
                      w_val = "weights_cache[" + weight_id + "]";
                    } else {
                      w_val = "args.weights.Read(filters_offset + " +
                              weight_id + ")";
                    }
                  } else {
                    w_val = "f" + weight_id;
                  }
                  if (GetWeightsDescription().IsI4O4()) {
                    c += "    " + R + " += " + w_val + " * " + S + "." +
                         channels[ch] + ";\n";
                  } else {
                    c += "    " + R + "." + channels[ch] + " += dot(" + w_val +
                         ", " + S + ");\n";
                  }
                }
              }
            }
          }
        }
      } else {  // F32_F16 precision and weights type is float16
        for (int z = 0; z < block_size.z; ++z) {
          const std::string zind = std::to_string(z);
          for (int y = 0; y < block_size.y; ++y) {
            const std::string yind = std::to_string(y);
            for (int x = 0; x < block_size.x; ++x) {
              const std::string xind = std::to_string(x);
              std::string R = "r" + generate_id_full(xind, yind, zind, sind);
              std::string S = "src" + generate_id(xind, yind, zind);
              std::vector<std::string> F(4);
              for (int i = 0; i < 4; ++i) {
                std::string weight_id =
                    std::to_string(s * 4 + i + shared_offset);
                if (conv_params.AreWeightsBuffer()) {
                  if (gpu_info.SupportsPointersInKernels()) {
                    F[i] = "weights_cache[" + weight_id + "]";
                  } else {
                    F[i] =
                        "args.weights.Read(filters_offset + " + weight_id + ")";
                  }
                } else {
                  F[i] = "f" + weight_id;
                }
              }
              if (GetWeightsDescription().IsI4O4()) {
                c += "    " + R + " += TO_ACCUM_TYPE(" + S + ".x * " + F[0] +
                     " + " + S + ".y * " + F[1] + " + " + S + ".z * " + F[2] +
                     " + " + S + ".w * " + F[3] + ");\n";
              } else {
                c += "    " + R + ".x += dot(" + S + ", " + F[0] + ");\n";
                c += "    " + R + ".y += dot(" + S + ", " + F[1] + ");\n";
                c += "    " + R + ".z += dot(" + S + ", " + F[2] + ");\n";
                c += "    " + R + ".w += dot(" + S + ", " + F[3] + ");\n";
              }
            }
          }
        }
      }
    }
  };

  c += "  int s = 0;\n";
  c += "  do {\n";
  declare_src();
  const int total_work_items =
      work_group_size_.x * work_group_size_.y * work_group_size_.z;
  if (conv_params.weights_upload_type ==
      ConvPowerVR::WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP) {
    c += GenerateAsyncUpload("weights_cache", "filters_loc",
                             /*global_offset_name*/ "", local_mem_size);
  } else if (conv_params.weights_upload_type ==
             ConvPowerVR::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    c += "    " + barrier + ";\n";
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
  } else if (conv_params.AreWeightsBuffer()) {  // GLOBAL_MEM/CONSTANT_MEM
    if (gpu_info.SupportsPointersInKernels()) {
      c += "    weights_cache = filters_loc;\n";
    }
  } else {  // TEXTURES_MEM
    for (int dst_s = 0; dst_s < block_size.w; ++dst_s) {
      std::string f_y = trivial_kernel_size ? "s" : "filter_offset";
      if (conv_params.different_weights_for_height) {
        f_y = "DST_Y * args.src_tensor.Slices() + s";
      }
      c += absl::Substitute(
          R"(    FLT4 f$2 = args.weights0.Read(DST_S + $0, $1);
    FLT4 f$3 = args.weights1.Read(DST_S + $0, $1);
    FLT4 f$4 = args.weights2.Read(DST_S + $0, $1);
    FLT4 f$5 = args.weights3.Read(DST_S + $0, $1);
)",
          dst_s, f_y, dst_s * 4 + 0, dst_s * 4 + 1, dst_s * 4 + 2,
          dst_s * 4 + 3);
    }
    if (!trivial_kernel_size) {
      c += "    filter_offset++;\n";
    }
  }
  read_src();
  c += "    s += 1;\n";
  if (conv_params.weights_upload_type ==
      ConvPowerVR::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    c += "    " + barrier + ";\n";
  }
  conv_core(0);
  for (int i = 1; i < conv_params.src_depth_loop_size; ++i) {
    read_src();
    conv_core(i * block_size.w * 4);
    c += "    s += 1;\n";
  }
  if (conv_params.AreWeightsBuffer()) {
    if (gpu_info.SupportsPointersInKernels()) {
      c += "    filters_loc += " + std::to_string(local_mem_size) + ";\n";
    } else {
      c += "    filters_offset += " + std::to_string(local_mem_size) + ";\n";
    }
  }
  c += "  } while (s < args.src_tensor.Slices());\n";
  if (!conv_params.x_kernel_is_1) {
    c += "  };\n";
  }
  if (!conv_params.y_kernel_is_1) {
    c += "  };\n";
  }
  if (src_def.HasAxis(Axis::DEPTH) && !conv_params_.z_kernel_is_1) {
    c += "  };\n";
  }
  if (conv_params.AreWeightsBuffer()) {
    if (conv_params.weights_upload_type ==
        ConvPowerVR::WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP) {
      c += GenerateAsyncUpload("weights_cache", "args.biases.GetPtr()", "DST_S",
                               block_size.w);
    } else if (conv_params.weights_upload_type ==
               ConvPowerVR::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
      c += "  " + barrier + ";\n";
      c += GenerateUploadByThreads("weights_cache", "args.biases.GetPtr()",
                                   "DST_S", "lid", total_work_items,
                                   block_size.w);
      c += "  " + barrier + ";\n";
    } else if (gpu_info.SupportsPointersInKernels()) {
      c += "  weights_cache = args.biases.GetPtr() + DST_S;\n";
    }
  }
  if (late_oob_check) {
    c += "  if (" + dst_oob_check + ") {\n";
    c += "    return;\n";
    c += "  }\n";
  }

  auto generate_dst_check = [&](int x, int y, int z) {
    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH};
    const std::vector<std::string> names{"Width()", "Height()", "Depth()"};
    std::vector<std::string> coords(3);
    coords[0] = "DST_X + " + std::to_string(x);
    coords[1] = "DST_Y + " + std::to_string(y);
    coords[2] = "DST_Z + " + std::to_string(z);
    const std::vector<int> ids{x, y, z};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_def.HasAxis(axis) && ids[i] != 0) {
        if (!check.empty()) {
          check += " && ";
        }
        check += coords[i] + " < args.dst_tensor." + names[i];
      }
    }
    return check;
  };

  for (int s = 0; s < block_size.w; ++s) {
    const std::string sind = std::to_string(s);
    c += "  if (DST_S + " + sind + " >= args.dst_tensor.Slices()) return;\n";
    c += "  {\n";
    if (conv_params.AreWeightsBuffer() &&
        gpu_info.SupportsPointersInKernels()) {
      c += "    FLT4 bias_val = TO_FLT4(weights_cache[" + sind + "]);\n";
    } else {
      c += "    FLT4 bias_val = args.biases.Read(DST_S + " + sind + ");\n";
    }
    for (int z = 0; z < block_size.z; ++z) {
      const std::string zind = std::to_string(z);
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        for (int x = 0; x < block_size.x; ++x) {
          const std::string xind = std::to_string(x);
          const std::string id = generate_id_full(xind, yind, zind, sind);
          const std::string check = generate_dst_check(x, y, z);
          std::string coords = "DST_X + " + xind + ", DST_Y + " + yind;
          if (src_def.HasAxis(Axis::DEPTH)) {
            coords += ", DST_Z + " + zind;
          }
          coords += ", DST_S + " + sind;
          if (!check.empty()) {
            c += "  if (" + check + ") {\n";
          } else {
            c += "  {\n";
          }
          c += "    FLT4 res = TO_FLT4(r" + id + ") + bias_val;\n";
          c += "    args.dst_tensor.Write(res, " + coords + ");\n";
          c += "  }\n";
        }
      }
    }
    c += "  }\n";
  }
  c += "}\n";
  return c;
}

ConvPowerVR::ConvParams ConvPowerVR::GuessBestParams(
    const GpuInfo& gpu_info, const OperationDef& definition, int src_depth,
    int dst_depth, bool x_kernel_is_1, bool y_kernel_is_1,
    bool different_weights_for_height, const BHWC* dst_shape) {
  ConvParams conv_params;
  conv_params.linear_spatial = false;
  conv_params.linear_all = false;
  conv_params.block_size = int4(1, 1, 1, 1);
  conv_params.weights_data_type =
      DeduceDataTypeFromPrecision(definition.precision);
  conv_params.x_kernel_is_1 = x_kernel_is_1;
  conv_params.y_kernel_is_1 = y_kernel_is_1;
  conv_params.different_weights_for_height = different_weights_for_height;
  if (gpu_info.IsNvidia()) {
    if (different_weights_for_height) {
      work_group_size_ = int3(32, 1, 1);
      work_group_launch_order_ = int3(2, 0, 1);
      conv_params.fixed_work_group_size = true;
    } else {
      conv_params.linear_spatial = true;
      work_group_size_ = int3(32, 1, 1);
      work_group_launch_order_ = int3(1, 0, 2);
      conv_params.fixed_work_group_size = true;
    }
    conv_params.block_size = int4(2, 1, 1, 4);
    conv_params.src_depth_loop_size = 1;
    conv_params.weights_upload_type = WeightsUploadType::LOCAL_MEM_BY_THREADS;
    if (dst_depth % 4 == 0 || dst_depth >= 8) {
      conv_params.block_size.w = 4;
    } else if (dst_depth % 2 == 0 || dst_depth >= 4) {
      conv_params.block_size.w = 2;
    } else {
      conv_params.block_size.w = dst_depth;
    }
    if (dst_shape) {
      int task_size = dst_shape->w * dst_shape->b * dst_shape->h * dst_depth;
      float task_size_per_cu =
          static_cast<float>(task_size) / gpu_info.GetComputeUnitsCount();
      int block_size = conv_params.block_size.x * conv_params.block_size.y *
                       conv_params.block_size.w;
      float threads_per_cu = task_size_per_cu / block_size;
      float warps_per_cu = threads_per_cu / 32 /*warp_size*/;
      if (warps_per_cu < 8.0f) {
        conv_params.block_size.x = 1;
      }
      if (warps_per_cu < 4.0f && conv_params.block_size.w >= 4) {
        conv_params.block_size.w /= 2;
      }
      if (warps_per_cu < 2.0f && conv_params.block_size.w >= 2) {
        conv_params.block_size.w /= 2;
      }
    }
    if (src_depth % 2 == 0) {
      conv_params.src_depth_loop_size = 2;
    }
    if (src_depth % 4 == 0 && conv_params.block_size.w <= 2) {
      conv_params.src_depth_loop_size = 4;
    }
  } else if (gpu_info.IsPowerVR()) {
    if (different_weights_for_height) {
      work_group_size_ = int3(32, 1, 1);
      work_group_launch_order_ = int3(2, 0, 1);
      conv_params.fixed_work_group_size = true;
    } else {
      conv_params.linear_spatial = true;
      work_group_size_ = int3(32, 1, 1);
      work_group_launch_order_ = int3(1, 0, 2);
      conv_params.fixed_work_group_size = true;
    }
    conv_params.weights_data_type =
        definition.precision == CalculationsPrecision::F16 ? DataType::FLOAT16
                                                           : DataType::FLOAT32;
    conv_params.block_size = int4(1, 1, 1, 4);
    conv_params.src_depth_loop_size = 1;
    conv_params.weights_upload_type =
        WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP;
    if (dst_depth % 8 == 0 || dst_depth >= 32) {
      conv_params.block_size.w = 8;
    } else if (dst_depth % 4 == 0 || dst_depth >= 8) {
      conv_params.block_size.w = 4;
    } else if (dst_depth % 2 == 0 || dst_depth >= 4) {
      conv_params.block_size.w = 2;
    } else {
      conv_params.block_size.w = dst_depth;
    }
    if (definition.precision == CalculationsPrecision::F16) {
      conv_params.block_size.w = std::min(4, conv_params.block_size.w);
      if (src_depth % 2 == 0) {
        conv_params.src_depth_loop_size = 2;
      }
      if (src_depth % 4 == 0 && conv_params.block_size.w <= 2) {
        conv_params.src_depth_loop_size = 4;
      }
      if (conv_params.block_size.w == 1) {
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
  } else if (gpu_info.IsAMD()) {
    if (different_weights_for_height) {
      work_group_size_ = int3(32, 1, 1);
      work_group_launch_order_ = int3(2, 0, 1);
      conv_params.fixed_work_group_size = true;
    } else {
      work_group_size_ = int3(8, 4, 1);
      work_group_launch_order_ = int3(2, 0, 1);
      conv_params.fixed_work_group_size = true;
    }

    conv_params.block_size = int4(2, 1, 1, 1);
    if (x_kernel_is_1 && y_kernel_is_1) {
      conv_params.block_size.y = 2;
    }
    conv_params.src_depth_loop_size = 1;
    conv_params.weights_upload_type = WeightsUploadType::CONSTANT_MEM;
    if (dst_depth % 8 == 0 || dst_depth >= 32) {
      conv_params.block_size.w = 8;
    } else if (dst_depth % 4 == 0 || dst_depth >= 8) {
      conv_params.block_size.w = 4;
    } else if (dst_depth % 2 == 0 || dst_depth >= 4) {
      conv_params.block_size.w = 2;
    } else {
      conv_params.block_size.w = 1;
    }
    if (src_depth % 2 == 0 && src_depth >= 16) {
      conv_params.src_depth_loop_size = 2;
    }
  } else if (gpu_info.IsMali()) {
    int block_size = 2;
    if (dst_shape) {
      int task_size = dst_shape->w * dst_shape->b * dst_shape->h * dst_depth;
      block_size = GetRecommendedBlockSizeForConv(
          gpu_info, definition.precision, task_size);
    }
    if (!x_kernel_is_1 || !y_kernel_is_1) {
      if (gpu_info.mali_info.IsMidgard() || gpu_info.mali_info.IsBifrost()) {
        block_size = std::min(block_size, 4);
      }
    }
    if (block_size == 8) {
      if (dst_depth == 1 || dst_depth == 3) {
        conv_params.block_size = int4(2, 2, 1, 1);
      } else {
        conv_params.block_size = int4(2, 2, 1, 2);
      }
    } else if (block_size == 4) {
      if (dst_depth == 1 || dst_depth == 3) {
        conv_params.block_size = int4(2, 2, 1, 1);
      } else {
        conv_params.block_size = int4(2, 1, 1, 1);
        if (definition.precision == CalculationsPrecision::F32 &&
            gpu_info.mali_info.IsValhall()) {
          conv_params.block_size.y = 2;
        } else {
          conv_params.block_size.w = 2;
        }
      }
    } else if (block_size == 2) {
      conv_params.block_size = int4(2, 1, 1, 1);
    } else {
      conv_params.block_size = int4(1, 1, 1, 1);
    }
    conv_params.src_depth_loop_size = 1;
    MaliInfo mali_info = gpu_info.mali_info;
    if (src_depth % 2 == 0 && block_size <= 2 && !mali_info.IsMidgard()) {
      conv_params.src_depth_loop_size = 2;
    }
    if (src_depth % 4 == 0 && block_size == 1 && !mali_info.IsMidgard() &&
        definition.precision == CalculationsPrecision::F16) {
      conv_params.src_depth_loop_size = 4;
    }
    work_group_size_ = int3(4, 4, 1);
    work_group_launch_order_ = int3(0, 1, 2);
    conv_params.fixed_work_group_size = false;
    conv_params.weights_upload_type = WeightsUploadType::GLOBAL_MEM;
  } else if (gpu_info.IsAdreno()) {
    conv_params.block_size = int4(2, 2, 1, 2);
    if (gpu_info.adreno_info.IsAdreno3xx()) {
      if (definition.precision == CalculationsPrecision::F16) {
        conv_params.block_size = int4(2, 2, 1, 2);
      } else if (definition.precision == CalculationsPrecision::F32_F16) {
        conv_params.block_size = int4(2, 1, 1, 2);
      } else {  // F32
        conv_params.block_size = int4(2, 2, 1, 1);
      }
    }
    work_group_size_ = int3(8, 2, 1);
    work_group_launch_order_ = int3(0, 1, 2);
    conv_params.fixed_work_group_size = false;
    conv_params.src_depth_loop_size = 1;
    if (definition.src_tensors.size() == 2) {
      // dynamic weights supported only with buffers.
      conv_params.weights_upload_type = WeightsUploadType::GLOBAL_MEM;
    } else {
      conv_params.weights_upload_type = WeightsUploadType::TEXTURES_MEM_X4;
    }
  } else if (gpu_info.IsIntel()) {
    if (different_weights_for_height) {
      work_group_size_ = int3(16, 1, 1);
      work_group_launch_order_ = int3(0, 1, 2);
      conv_params.fixed_work_group_size = true;
    } else {
      conv_params.linear_spatial = true;
      work_group_size_ = int3(16, 1, 1);
      work_group_launch_order_ = int3(0, 1, 2);
      conv_params.fixed_work_group_size = true;
    }
    conv_params.block_size = int4(1, 1, 1, 4);
    conv_params.src_depth_loop_size = 1;
    int sub_group_size = 16;
    const bool supports_subgroups =
        gpu_info.SupportsExtension("cl_khr_subgroups") ||
        gpu_info.SupportsExtension("cl_intel_subgroups");
    if (definition.precision != CalculationsPrecision::F32_F16 &&
        supports_subgroups &&
        gpu_info.SupportsExtension("cl_intel_required_subgroup_size") &&
        gpu_info.SupportsSubGroupWithSize(sub_group_size)) {
      conv_params.weights_upload_type =
          WeightsUploadType::PRIVATE_MEM_SIMD_BROADCAST;
      conv_params.simd_size = sub_group_size;
    } else {
      conv_params.weights_upload_type = WeightsUploadType::LOCAL_MEM_BY_THREADS;
    }
    if (dst_depth % 4 == 0 || dst_depth >= 8) {
      conv_params.block_size.w = 4;
    } else if (dst_depth % 2 == 0 || dst_depth >= 4) {
      conv_params.block_size.w = 2;
    } else {
      conv_params.block_size.w = dst_depth;
    }
    if (src_depth % 2 == 0) {
      conv_params.src_depth_loop_size = 2;
    }
    if (src_depth % 4 == 0 && conv_params.block_size.w <= 2) {
      conv_params.src_depth_loop_size = 4;
    }
  } else if (gpu_info.IsApple()) {
    conv_params.block_size = int4(2, 2, 1, 2);
    work_group_size_ = int3(8, 4, 1);
    work_group_launch_order_ = int3(0, 1, 2);
    conv_params.fixed_work_group_size = true;
    conv_params.src_depth_loop_size = 1;
    conv_params.weights_upload_type = WeightsUploadType::GLOBAL_MEM;
  } else {
    conv_params.block_size = int4(1, 1, 1, 4);
    work_group_size_ = int3(8, 2, 1);
    work_group_launch_order_ = int3(0, 1, 2);
    conv_params.fixed_work_group_size = false;
    conv_params.src_depth_loop_size = 1;
    conv_params.weights_upload_type = WeightsUploadType::GLOBAL_MEM;
    if (dst_depth % 4 == 0 || dst_depth >= 8) {
      conv_params.block_size.w = 4;
    } else if (dst_depth % 2 == 0 || dst_depth >= 4) {
      conv_params.block_size.w = 2;
    } else {
      conv_params.block_size.w = dst_depth;
    }
    if (src_depth % 2 == 0) {
      conv_params.src_depth_loop_size = 2;
    }
    if (src_depth % 4 == 0 && conv_params.block_size.w <= 2) {
      conv_params.src_depth_loop_size = 4;
    }
  }
  if (conv_params.AreWeightsBuffer()) {
    if (gpu_info.IsApple()) {
      conv_params.weights_layout = WeightsLayout::kOSpatialIOGroupO4I4;
    } else {
      conv_params.weights_layout = WeightsLayout::kOSpatialIOGroupI4O4;
    }
  } else {
    if (gpu_info.IsApple()) {
      conv_params.weights_layout =
          WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4;
    } else {
      conv_params.weights_layout =
          WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
    }
  }

  return conv_params;
}

ConvPowerVR::ConvParams ConvPowerVR::GuessBestParams(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Convolution2DAttributes& attr, const BHWC* dst_shape) {
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
  return GuessBestParams(gpu_info, definition, src_depth, dst_depth,
                         x_kernel_is_1, y_kernel_is_1, false, dst_shape);
}

ConvPowerVR::ConvParams ConvPowerVR::GuessBestParams(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Convolution3DAttributes& attr, const BHWDC* dst_shape) {
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
  const bool z_kernel_is_1 = attr.weights.shape.d == 1 && attr.strides.d == 1 &&
                             attr.dilations.d == 1 &&
                             attr.padding.prepended.d == 0 &&
                             attr.padding.appended.d == 0;

  ConvPowerVR::ConvParams result;
  BHWC shape;
  if (dst_shape) {
    shape.b = dst_shape->b;
    shape.h = dst_shape->h * dst_shape->d;
    shape.w = dst_shape->w;
    shape.c = dst_shape->c;
    result = GuessBestParams(gpu_info, definition, src_depth, dst_depth,
                             x_kernel_is_1, y_kernel_is_1, false, &shape);
  } else {
    result = GuessBestParams(gpu_info, definition, src_depth, dst_depth,
                             x_kernel_is_1, y_kernel_is_1, false, nullptr);
  }
  result.z_kernel_is_1 = z_kernel_is_1;
  return result;
}

ConvPowerVR::ConvParams ConvPowerVR::GuessBestParams(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Convolution2DAttributes& attr, const BHWC& weights_shape,
    const BHWC* dst_shape) {
  const int dst_depth = DivideRoundUp(weights_shape.b, 4);
  const int src_depth = DivideRoundUp(weights_shape.c, 4);
  const bool x_kernel_is_1 =
      weights_shape.w == 1 && attr.strides.w == 1 && attr.dilations.w == 1 &&
      attr.padding.prepended.w == 0 && attr.padding.appended.w == 0;
  const bool y_kernel_is_1 =
      weights_shape.h == 1 && attr.strides.h == 1 && attr.dilations.h == 1 &&
      attr.padding.prepended.h == 0 && attr.padding.appended.h == 0;
  return GuessBestParams(gpu_info, definition, src_depth, dst_depth,
                         x_kernel_is_1, y_kernel_is_1, false, dst_shape);
}

ConvPowerVR::ConvParams ConvPowerVR::GuessBestParams(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const FullyConnectedAttributes& attr, const BHWC* dst_shape) {
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  ConvPowerVR::ConvParams params = GuessBestParams(
      gpu_info, definition, src_depth, dst_depth, true, true, false, dst_shape);
  work_group_size_.x *= work_group_size_.y;
  work_group_size_.y = 1;
  params.block_size.x *= params.block_size.y;
  params.block_size.y = 1;
  return params;
}

ConvPowerVR::ConvParams ConvPowerVR::GuessBestParamsWinograd(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Convolution2DAttributes& attr, const BHWC* dst_shape) {
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  ConvPowerVR::ConvParams params = GuessBestParams(
      gpu_info, definition, src_depth, dst_depth, true, true, true, dst_shape);
  params.block_size.x *= params.block_size.y;
  params.block_size.y = 1;
  return params;
}

ConvPowerVR CreateConvPowerVR(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr,
                              const BHWC* dst_shape) {
  ConvPowerVR result(definition, attr, gpu_info, dst_shape);
  result.GenerateCode(gpu_info);
  result.UploadData(attr.weights, attr.bias);
  return result;
}

ConvPowerVR CreateConvPowerVR(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              const FullyConnectedAttributes& attr,
                              const BHWC* dst_shape) {
  ConvPowerVR result(definition, attr, gpu_info, dst_shape);
  result.GenerateCode(gpu_info);
  result.UploadData(attr.weights, attr.bias);
  return result;
}

ConvPowerVR CreateConvPowerVRDynamicWeights(const GpuInfo& gpu_info,
                                            const OperationDef& definition,
                                            const Convolution2DAttributes& attr,
                                            const BHWC& weights_shape,
                                            const BHWC* dst_shape) {
  ConvPowerVR result(definition, attr, weights_shape, gpu_info, dst_shape);
  result.GenerateCode(gpu_info);
  result.UploadBias(attr.bias);
  return result;
}

ConvPowerVR CreateConvPowerVRWino4x4To6x6(const GpuInfo& gpu_info,
                                          const OperationDef& definition,
                                          const Convolution2DAttributes& attr,
                                          const BHWC* dst_shape) {
  ConvPowerVR result(definition);
  result.conv_params_ =
      result.GuessBestParamsWinograd(gpu_info, definition, attr, dst_shape);
  result.GenerateCode(gpu_info);
  result.UploadDataForWinograd4x4To6x6(attr.weights);
  return result;
}

ConvPowerVR CreateConvPowerVR3D(const GpuInfo& gpu_info,
                                const OperationDef& definition,
                                const Convolution3DAttributes& attr,
                                const BHWDC* dst_shape) {
  ConvPowerVR result(definition, attr, gpu_info, dst_shape);
  result.GenerateCode(gpu_info);
  result.UploadWeights(attr.weights);
  result.UploadBias(attr.bias);
  return result;
}

}  // namespace gpu
}  // namespace tflite
