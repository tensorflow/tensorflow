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

#include "tensorflow/lite/delegates/gpu/common/tasks/conv_generic.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
std::string GenerateUploadByThreads(
    const std::string& local_ptr_name, const std::string& name, bool use_ptrs,
    const std::string& global_offset_name, const std::string type_conversion,
    const std::string& lid_name, int total_work_items, int elements_to_upload) {
  std::string c;
  std::string offset =
      global_offset_name.empty() ? "" : global_offset_name + " + ";
  const int groups = elements_to_upload / total_work_items;
  const int reminder = elements_to_upload % total_work_items;
  const std::string access_start = name + (use_ptrs ? "[" : ".Read(");
  const std::string access_end = use_ptrs ? "]" : ")";
  for (int i = 0; i < groups; ++i) {
    const std::string value = access_start + offset + lid_name + " + " +
                              std::to_string(total_work_items * i) + access_end;
    c += "    " + local_ptr_name + "[" + lid_name + " + " +
         std::to_string(total_work_items * i) +
         "] = " + absl::Substitute(type_conversion, value) + ";\n";
  }
  if (reminder != 0) {
    const std::string value = access_start + offset + lid_name + " + " +
                              std::to_string(total_work_items * groups) +
                              access_end;
    c += "    if (" + lid_name + " < " + std::to_string(reminder) + ") {\n";
    c += "      " + local_ptr_name + "[" + lid_name + " + " +
         std::to_string(total_work_items * groups) +
         "] = " + absl::Substitute(type_conversion, value) + ";\n";
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
                                bool need_depth, bool need_batch) {
  std::string c;
  int3 launch_remap;
  launch_remap[work_group_launch_order.x] = 0;
  launch_remap[work_group_launch_order.y] = 1;
  launch_remap[work_group_launch_order.z] = 2;
  if (linear_all) {
    c += "  int linear_all = GLOBAL_ID_0;\n";
    if (need_batch) {
      c += "  int B = linear_all % args.task_size_b;\n";
      c += "  linear_all = linear_all / args.task_size_b;\n";
    }
    c += "  int DST_X = linear_all % args.task_size_x;\n";
    c += "  linear_all = linear_all / args.task_size_x;\n";
    c += "  int DST_Y = linear_all % args.task_size_y;\n";
    c += "  linear_all = linear_all / args.task_size_y;\n";
    if (need_depth) {
      c += "  int DST_Z = linear_all % args.task_size_z;\n";
      c += "  linear_all = linear_all / args.task_size_z;\n";
    }
    c += "  int DST_S = linear_all;\n";
  } else if (linear_spatial) {
    if (work_group_launch_order[0] == 0) {
      c += "  int linear_spatial = GLOBAL_ID_0;\n";
    } else {
      c += "  int linear_spatial = GROUP_ID_" +
           std::to_string(launch_remap[0]) + " * GROUP_SIZE_0 + LOCAL_ID_0;\n";
    }
    if (need_batch) {
      c += "  int B = linear_spatial % args.task_size_b;\n";
      c += "  linear_spatial = linear_spatial / args.task_size_b;\n";
    }
    c += "  int DST_X = linear_spatial % args.task_size_x;\n";
    c += "  linear_spatial = linear_spatial / args.task_size_x;\n";
    c += "  int DST_Y = linear_spatial % args.task_size_y;\n";
    c += "  linear_spatial = linear_spatial / args.task_size_y;\n";
    if (need_depth) {
      c += "  int DST_Z = linear_spatial;\n";
    }
    if (work_group_launch_order[1] == 1) {
      c += "  int DST_S = GLOBAL_ID_1;\n";
    } else {
      c += "  int DST_S = GROUP_ID_" + std::to_string(launch_remap[1]) +
           " * GROUP_SIZE_1 + LOCAL_ID_1;\n";
    }
  } else {
    if (work_group_launch_order[0] == 0) {
      c += "  int DST_X = GLOBAL_ID_0;\n";
    } else {
      c += "  int DST_X = GROUP_ID_" + std::to_string(launch_remap[0]) +
           " * GROUP_SIZE_0 + LOCAL_ID_0;\n";
    }
    if (need_batch) {
      c += "  int B = DST_X % args.task_size_b;\n";
      c += "  DST_X = DST_X / args.task_size_b;\n";
    }
    std::string global_id_1;
    if (work_group_launch_order[1] == 1) {
      global_id_1 = "GLOBAL_ID_1";
    } else {
      global_id_1 = "GROUP_ID_" + std::to_string(launch_remap[1]) +
                    " * GROUP_SIZE_1 + LOCAL_ID_1";
    }
    if (need_depth) {
      c += "  int linear_id_1 = " + global_id_1 + ";\n";
      c += "  int DST_Y = linear_id_1 % args.task_size_y;\n";
      c += "  int DST_Z = linear_id_1 / args.task_size_y;\n";
    } else {
      c += "  int DST_Y = " + global_id_1 + ";\n";
    }
    if (work_group_launch_order[2] == 2) {
      c += "  int DST_S = GLOBAL_ID_2;\n";
    } else {
      c += "  int DST_S = GROUP_ID_" + std::to_string(launch_remap[2]) +
           " * GROUP_SIZE_2 + LOCAL_ID_2;\n";
    }
  }
  if (block_size.x != 1) {
    c += "  DST_X *= " + std::to_string(block_size.x) + ";\n";
  }
  if (block_size.y != 1) {
    c += "  DST_Y *= " + std::to_string(block_size.y) + ";\n";
  }
  if (need_depth && block_size.z != 1) {
    c += "  DST_Z *= " + std::to_string(block_size.z) + ";\n";
  }
  if (block_size.w != 1) {
    c += "  DST_S *= " + std::to_string(block_size.w) + ";\n";
  }

  return c;
}
}  // namespace

ConvGeneric::ConvGeneric(const OperationDef& definition,
                         const Convolution2DAttributes& attr,
                         const GpuInfo& gpu_info, const BHWC* dst_shape)
    : GPUOperation(definition),
      stride_(attr.strides.w, attr.strides.h, 1, 1),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h, 0, 0),
      kernel_size_(attr.weights.shape.w, attr.weights.shape.h, 1, 1),
      dilation_(attr.dilations.w, attr.dilations.h, 1, 1),
      conv_params_(GuessBestParams(gpu_info, definition, attr, dst_shape)) {
  const int src_slices = DivideRoundUp(attr.weights.shape.i, 4);
  const int dst_slices = DivideRoundUp(attr.weights.shape.o, 4);
  if (attr.groups != 1) {
    conv_params_.groups_support = true;
    const int dst_group_slices = dst_slices / attr.groups;
    if (dst_group_slices % conv_params_.block_size.w != 0) {
      if (conv_params_.block_size.w == 4 && dst_group_slices % 2 == 0) {
        conv_params_.block_size.w = 2;
      } else {
        conv_params_.block_size.w = 1;
      }
    }
    args_.AddInt("src_group_size", src_slices);
    args_.AddInt("dst_group_size", dst_slices / attr.groups);
  }
}

ConvGeneric::ConvGeneric(const OperationDef& definition,
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

ConvGeneric::ConvGeneric(const OperationDef& definition,
                         const FullyConnectedAttributes& attr,
                         const GpuInfo& gpu_info, const BHWC* dst_shape)
    : GPUOperation(definition),
      stride_(1, 1, 1, 1),
      padding_(0, 0, 0, 0),
      kernel_size_(1, 1, 1, 1),
      dilation_(1, 1, 1, 1),
      conv_params_(GuessBestParams(gpu_info, definition, attr, dst_shape)) {}

ConvGeneric::ConvGeneric(const OperationDef& definition)
    : GPUOperation(definition),
      stride_(1, 1, 1, 1),
      padding_(0, 0, 0, 0),
      kernel_size_(1, 1, 1, 1),
      dilation_(1, 1, 1, 1) {}

ConvGeneric::ConvGeneric(ConvGeneric&& operation)
    : GPUOperation(std::move(operation)),
      stride_(operation.stride_),
      padding_(operation.padding_),
      kernel_size_(operation.kernel_size_),
      dilation_(operation.dilation_),
      conv_params_(operation.conv_params_) {}

ConvGeneric::ConvGeneric(const OperationDef& definition,
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

ConvGeneric& ConvGeneric::operator=(ConvGeneric&& operation) {
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

void ConvGeneric::GenerateCode(const GpuInfo& gpu_info) {
  if (conv_params_.linear_all) {
    grid_dimension_ = 1;
  } else if (conv_params_.linear_spatial) {
    grid_dimension_ = 2;
  }

  AddSrcTensor("src_tensor", definition_.src_tensors[0]);
  AddDstTensor("dst_tensor", definition_.dst_tensors[0]);
  if (definition_.src_tensors.size() == 2) {  // dynamic weights
    const DataType weights_type = definition_.GetDataType();
    if (conv_params_.weights_layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
        conv_params_.weights_layout == WeightsLayout::kOSpatialIOGroupO4I4) {
      definition_.src_tensors[1] = {weights_type, TensorStorageType::BUFFER,
                                    Layout::HWC};
      BufferDescriptor desc;
      desc.element_type = weights_type;
      desc.element_size = 4;
      desc.memory_type = conv_params_.weights_upload_type ==
                                 ConvGeneric::WeightsUploadType::CONSTANT_MEM
                             ? MemoryType::CONSTANT
                             : MemoryType::GLOBAL;

      AddSrcBuffer("weights", desc);
    } else {
      TensorDescriptor desc{weights_type, TensorStorageType::TEXTURE_2D,
                            Layout::HW};
      definition_.src_tensors[1] = desc;
      definition_.src_tensors.push_back(desc);
      definition_.src_tensors.push_back(desc);
      definition_.src_tensors.push_back(desc);
      for (int i = 0; i < 4; ++i) {
        const std::string name = "weights" + std::to_string(i);
        AddSrcTensor(name, definition_.src_tensors[1 + i]);
      }
    }
  }

  code_ = GenerateConv(gpu_info, definition_, conv_params_);
  if (definition_.precision == CalculationsPrecision::F16 &&
      gpu_info.IsPowerVR()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  if (gpu_info.IsMali()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  if (conv_params_.IsPrivateMemBroadcast() &&
      (gpu_info.IsCL20OrHigher() || gpu_info.opencl_info.IsCLVK())) {
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

absl::Status ConvGeneric::BindArguments(ArgumentsBinder* args) {
  const int task_size_b = dst_[0]->Batch();
  const int task_size_x =
      DivideRoundUp(dst_[0]->Width(), conv_params_.block_size.x);
  const int task_size_y =
      DivideRoundUp(dst_[0]->Height(), conv_params_.block_size.y);
  const int task_size_z =
      DivideRoundUp(dst_[0]->Depth(), conv_params_.block_size.z);
  RETURN_IF_ERROR(args->SetInt("task_size_b", task_size_b));
  RETURN_IF_ERROR(args->SetInt("task_size_x", task_size_x));
  RETURN_IF_ERROR(args->SetInt("task_size_y", task_size_y));
  RETURN_IF_ERROR(args->SetInt("task_size_z", task_size_z));
  return absl::OkStatus();
}

int3 ConvGeneric::GetGridSize() const {
  const int task_size_b = dst_[0]->Batch();
  const int task_size_x =
      DivideRoundUp(dst_[0]->Width(), conv_params_.block_size.x);
  const int task_size_y =
      DivideRoundUp(dst_[0]->Height(), conv_params_.block_size.y);
  const int task_size_z =
      DivideRoundUp(dst_[0]->Depth(), conv_params_.block_size.z);
  const int task_size_s =
      DivideRoundUp(dst_[0]->Slices(), conv_params_.block_size.w);
  int3 wg;

  if (conv_params_.linear_all) {
    return int3(
        task_size_x * task_size_b * task_size_y * task_size_z * task_size_s, 1,
        1);
  } else if (conv_params_.linear_spatial) {
    return int3(task_size_x * task_size_b * task_size_y * task_size_z,
                task_size_s, 1);
  } else {
    return int3(task_size_x * task_size_b, task_size_y * task_size_z,
                task_size_s);
  }
}

void ConvGeneric::GetPossibleKernelWorkGroups(
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

std::string ConvGeneric::GenerateConv(const GpuInfo& gpu_info,
                                      const OperationDef& op_def,
                                      const ConvParams& conv_params) {
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
      if (src_def.HasAxis(axis) && !src_def.SupportsZeroClamp(axis, gpu_info) &&
          !is_1[i]) {
        if (!check.empty()) {
          check += " && ";
        }
        check += names[i] + coords[i];
      }
    }
    return check;
  };

  if (!conv_params_.x_kernel_is_1) {
    args_.AddInt("stride_x", stride_.x);
    args_.AddInt("padding_x", padding_.x);
    args_.AddInt("kernel_size_x", kernel_size_.x);
    args_.AddInt("dilation_x", dilation_.x);
  }
  if (!conv_params_.y_kernel_is_1) {
    args_.AddInt("stride_y", stride_.y);
    args_.AddInt("padding_y", padding_.y);
    args_.AddInt("kernel_size_y", kernel_size_.y);
    args_.AddInt("dilation_y", dilation_.y);
  }
  if (src_def.HasAxis(Axis::DEPTH) && !conv_params_.z_kernel_is_1) {
    args_.AddInt("stride_z", stride_.z);
    args_.AddInt("padding_z", padding_.z);
    args_.AddInt("kernel_size_z", kernel_size_.z);
    args_.AddInt("dilation_z", dilation_.z);
  }
  args_.AddInt("task_size_b");
  args_.AddInt("task_size_x");
  args_.AddInt("task_size_y");
  args_.AddInt("task_size_z");

  const int wg_total_size =
      work_group_size_.x * work_group_size_.y * work_group_size_.z;
  const std::string barrier =
      wg_total_size == 32 && gpu_info.IsWaveSizeEqualTo32()
          ? "SIMD_LOCAL_MEM_BARRIER"
          : "LOCAL_MEM_BARRIER";

  const bool need_local_mem =
      conv_params.weights_upload_type ==
          ConvGeneric::WeightsUploadType::LOCAL_MEM_BY_THREADS ||
      conv_params.weights_upload_type ==
          ConvGeneric::WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP;

  const int local_mem_size =
      conv_params.block_size.w * 4 * conv_params.src_depth_loop_size;

  const bool use_simd_broadcast = conv_params.IsPrivateMemBroadcast();
  const int simd_size = conv_params.simd_size;

  const bool late_oob_check = need_local_mem || use_simd_broadcast;

  const std::string weights_space =
      conv_params.weights_upload_type ==
              ConvGeneric::WeightsUploadType::CONSTANT_MEM
          ? "__constant"
          : "__global";

  std::string c;
  if (use_simd_broadcast && gpu_info.IsApiOpenCl()) {
    if (gpu_info.opencl_info.cl_version == OpenClVersion::kCl2_0 ||
        gpu_info.SupportsExtension("cl_khr_subgroups")) {
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
  if (use_simd_broadcast && gpu_info.IsApiOpenCl() &&
      gpu_info.SupportsExtension("cl_intel_required_subgroup_size")) {
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
                           src_def.HasAxis(Axis::DEPTH),
                           src_def.HasAxis(Axis::BATCH));
  if (src_def.HasAxis(Axis::BATCH)) {
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  }
  if (!conv_params.need_dst_loop) {
    c += "  DST_S = 0;\n";
  }
  c += "  if (DST_S >= args.dst_tensor.Slices()) return;\n";
  if (!late_oob_check) {
    c += "  if (" + dst_oob_check + ") {\n";
    c += "    return;\n";
    c += "  }\n";
  }
  if (conv_params.groups_support) {
    c += "      int conv_group_id = DST_S / args.dst_group_size;\n";
    c += "      int src_start_slice = conv_group_id * args.src_group_size;\n";
    c += "      int src_end_slice = src_start_slice + args.src_group_size;\n";
  }
  const std::string src_group_start_slice =
      conv_params.groups_support ? "src_start_slice" : "0";
  const std::string src_group_end_slice =
      conv_params.groups_support ? "src_end_slice" : "args.src_tensor.Slices()";
  const std::string src_group_slices = conv_params.groups_support
                                           ? "args.src_group_size"
                                           : "args.src_tensor.Slices()";
  if (conv_params.weights_upload_type ==
      ConvGeneric::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
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
      c += "  int xc" + xind + " = " + xc +
           " * args.stride_x + args.padding_x;\n";
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
  const std::string weights_global_ptr =
      weights_space + " " + ToCLDataType(conv_params.weights_data_type, 4) +
      "*";
  DataType summable_data_type = conv_params.weights_data_type;
  if (gpu_info.IsPowerVR() &&
      op_def.precision == CalculationsPrecision::F32_F16 &&
      conv_params.weights_upload_type ==
          ConvGeneric::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    summable_data_type = DataType::FLOAT32;
  }
  if (need_local_mem) {
    c += "  __local " + ToCLDataType(summable_data_type, 4) +
         " weights_cache[" + std::to_string(local_mem_size) + "];\n";
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
      offset = "DST_S * 4 * " + src_group_slices + kernel_spatial_offset;
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
      if (!src_def.SupportsZeroClamp(Axis::DEPTH, gpu_info)) {
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
      if (!src_def.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
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
      if (!src_def.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
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
      src_def.ReturnsZeroForNegOneRead(gpu_info) && !trivial_kernel_size;
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
          c += "  int addr" + id + " = args.src_tensor.GetAddress(" + coords +
               ", " + src_group_start_slice + ");\n";
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
          c += "    " + ToCLDataType(summable_data_type, 4) + " src" + id +
               ";\n";
        }
      }
    }
  };
  const bool conditional_read = gpu_info.IsMali();
  auto read_src = [&]() {
    const std::string read_as_type = ToCLDataType(summable_data_type);
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
          if (src_def.ReturnsZeroForNegOneRead(gpu_info)) {
            c += "    src" + id + " = args.src_tensor.Read<" + read_as_type +
                 ">(" + address + ");\n";
            const std::string ds = trivial_kernel_size ? "ds" : "ds" + id;
            c += "    " + address + " += " + ds + ";\n";
          } else {
            if (!check.empty()) {
              if (conditional_read) {
                c += "    src" + id + " = " + check +
                     " ? args.src_tensor.Read<" + read_as_type + ">(" +
                     address + ") : INIT_FLT4(0.0f);\n";
              } else {
                c += "    src" + id + " = args.src_tensor.Read<" +
                     read_as_type + ">(" + address + ") * INIT_FLT(" + check +
                     ");\n";
              }
            } else {
              c += "    src" + id + " = args.src_tensor.Read<" + read_as_type +
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
  bool use_fma = gpu_info.IsAMD() && gpu_info.IsApiOpenCl();
  auto conv_core = [&](int shared_offset) {
    const std::string channels[] = {"x", "y", "z", "w"};
    for (int s = 0; s < block_size.w; ++s) {
      const std::string sind = std::to_string(s);
      if (op_def.precision != CalculationsPrecision::F32_F16 ||
          summable_data_type == DataType::FLOAT32) {
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
                    if (need_local_mem ||
                        gpu_info.SupportsPointersInKernels()) {
                      w_val = "weights_cache[" + weight_id + "]";
                    } else {
                      w_val = "args.weights.Read(filters_offset + " +
                              weight_id + ")";
                    }
                  } else {
                    w_val = "f" + weight_id;
                  }
                  if (GetWeightsDescription().IsI4O4()) {
                    if (use_fma) {
                      c += "    " + R + " = fma(" + w_val + ", " + S + "." +
                           channels[ch] + ", " + R + ");\n";
                    } else {
                      c += "    " + R + " += " + w_val + " * " + S + "." +
                           channels[ch] + ";\n";
                    }
                  } else {
                    c += "    " + R + "." + channels[ch] + " += dot(" + w_val +
                         ", " + S + ");\n";
                  }
                }
              }
            }
          }
        }
      } else {  // F32_F16 precision
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
                  if (need_local_mem || gpu_info.SupportsPointersInKernels()) {
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

  c += "  int s = " + src_group_start_slice + ";\n";
  if (conv_params.need_src_loop) {
    c += "  do {\n";
  }
  declare_src();
  const int total_work_items =
      work_group_size_.x * work_group_size_.y * work_group_size_.z;
  const std::string type_conversion = GetTypeConversion(
      gpu_info, conv_params.weights_data_type, summable_data_type, 4);
  if (conv_params.weights_upload_type ==
      ConvGeneric::WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP) {
    c += GenerateAsyncUpload("weights_cache", "filters_loc",
                             /*global_offset_name*/ "", local_mem_size);
  } else if (conv_params.weights_upload_type ==
             ConvGeneric::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    if (gpu_info.IsApiMetal() && wg_total_size == 32 &&
        gpu_info.IsWaveSizeEqualTo32()) {
      c += "    SIMDGROUP_BARRIER(mem_flags::mem_none);\n";
    } else {
      c += "    " + barrier + ";\n";
    }
    if (gpu_info.SupportsPointersInKernels()) {
      c += GenerateUploadByThreads("weights_cache", "filters_loc",
                                   /*use_ptrs*/ true,
                                   /*global_offset_name*/ "", type_conversion,
                                   "lid", total_work_items, local_mem_size);
    } else {
      c += GenerateUploadByThreads("weights_cache", "args.weights",
                                   /*use_ptrs*/ false, "filters_offset",
                                   type_conversion, "lid", total_work_items,
                                   local_mem_size);
    }
  } else if (use_simd_broadcast) {
    int parts = local_mem_size / simd_size;
    int reminder = local_mem_size % simd_size;
    const std::string read_start = gpu_info.SupportsPointersInKernels()
                                       ? "filters_loc["
                                       : "args.weights.Read(filters_offset + ";
    const std::string read_end =
        gpu_info.SupportsPointersInKernels() ? "]" : ")";
    for (int i = 0; i < parts; ++i) {
      const std::string weights_index =
          "simd_id + " + std::to_string(i * simd_size);
      c += "    FLT4 simd_w" + std::to_string(i) + " = " + read_start +
           weights_index + read_end + ";\n";
    }
    if (reminder) {
      const std::string weights_index =
          "simd_id + " + std::to_string(parts * simd_size);
      c += "    FLT4 simd_w" + std::to_string(parts) + ";\n";
      c += "    if (simd_id < " + std::to_string(reminder) + ") {\n";
      c += "      simd_w" + std::to_string(parts) + " = " + read_start +
           weights_index + read_end + ";\n";
      c += "    }\n";
    }
  } else if (conv_params.AreWeightsBuffer()) {  // GLOBAL_MEM/CONSTANT_MEM
    if (gpu_info.SupportsPointersInKernels()) {
      c += "    weights_cache = filters_loc;\n";
    }
  } else {  // TEXTURES_MEM
    for (int dst_s = 0; dst_s < block_size.w; ++dst_s) {
      std::string f_y = trivial_kernel_size ? "s" : "filter_offset";
      if (trivial_kernel_size && conv_params.groups_support) {
        f_y = "s - src_start_slice";
      }
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
      ConvGeneric::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
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
  if (conv_params.need_src_loop) {
    c += "  } while (s < " + src_group_end_slice + ");\n";
  }
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
        ConvGeneric::WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP) {
      c += GenerateAsyncUpload("weights_cache", "args.biases.GetPtr()", "DST_S",
                               block_size.w);
    } else if (conv_params.weights_upload_type ==
               ConvGeneric::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
      c += "  " + barrier + ";\n";
      c += GenerateUploadByThreads("weights_cache", "args.biases",
                                   /*use_ptrs*/ false, "DST_S", type_conversion,
                                   "lid", total_work_items, block_size.w);
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
        (need_local_mem || gpu_info.SupportsPointersInKernels())) {
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

int GetGroupsCount(const BHWC& dst_shape, const int3& wg_size,
                   const int4& block_size) {
  const int dst_slices = DivideRoundUp(dst_shape.c, 4);

  int grid_x = DivideRoundUp(dst_shape.w, block_size.x) * dst_shape.b;
  int grid_y = DivideRoundUp(dst_shape.h, block_size.y);
  int grid_z = DivideRoundUp(dst_slices, block_size.w);

  return DivideRoundUp(grid_x, wg_size.x) * DivideRoundUp(grid_y, wg_size.y) *
         DivideRoundUp(grid_z, wg_size.z);
}

int GetGroupsCountForLinearWH(const BHWC& dst_shape, const int3& wg_size,
                              const int4& block_size) {
  const int dst_slices = DivideRoundUp(dst_shape.c, 4);

  int grid_x = DivideRoundUp(dst_shape.w, block_size.x) * dst_shape.b;
  int grid_y = DivideRoundUp(dst_shape.h, block_size.y);
  int grid_z = DivideRoundUp(dst_slices, block_size.w);

  return DivideRoundUp(grid_x * grid_y, wg_size.x) *
         DivideRoundUp(grid_z, wg_size.y);
}

int GetGroupsCountForLinearWHS(const BHWC& dst_shape, const int3& wg_size,
                               const int4& block_size) {
  const int dst_slices = DivideRoundUp(dst_shape.c, 4);

  int grid_x = DivideRoundUp(dst_shape.w, block_size.x) * dst_shape.b;
  int grid_y = DivideRoundUp(dst_shape.h, block_size.y);
  int grid_z = DivideRoundUp(dst_slices, block_size.w);

  return DivideRoundUp(grid_x * grid_y * grid_z, wg_size.x);
}

bool IsKernelXIs1(const Convolution2DAttributes& attr) {
  return attr.weights.shape.w == 1 && attr.strides.w == 1 &&
         attr.dilations.w == 1 && attr.padding.prepended.w == 0 &&
         attr.padding.appended.w == 0;
}

bool IsKernelYIs1(const Convolution2DAttributes& attr) {
  return attr.weights.shape.h == 1 && attr.strides.h == 1 &&
         attr.dilations.h == 1 && attr.padding.prepended.h == 0 &&
         attr.padding.appended.h == 0;
}

int GetMaximumPossibleWavesCount(const AppleInfo& apple_info,
                                 const BHWC& dst_shape) {
  if (apple_info.IsLocalMemoryPreferredOverGlobal()) {
    return GetGroupsCountForLinearWH(dst_shape, {32, 1, 1}, int4(1, 1, 1, 1));
  } else {
    return GetGroupsCountForLinearWHS(dst_shape, {32, 1, 1}, int4(1, 1, 1, 1));
  }
}

int GetRecommendedBlockSize(const AppleInfo& apple_info,
                            const BHWC& dst_shape) {
  const int max_waves = GetMaximumPossibleWavesCount(apple_info, dst_shape);
  const int cu_count = apple_info.GetComputeUnitsCount();
  if (max_waves >= cu_count * 64) {
    return 8;
  } else if (max_waves >= cu_count * 32) {
    return 4;
  } else if (max_waves >= cu_count * 16) {
    return 2;
  } else {
    return 1;
  }
}

struct WorkGroupSizeOption {
  enum class ThreadMapping { kDefault, kLinearSpatial, kLinearAll };
  int3 work_group_size;
  int work_groups_count;
  ThreadMapping thread_mapping;
  float penalty = 1.0f;
};

WorkGroupSizeOption CreateWorkGroupSizeOption(
    const int3& work_group_size,
    WorkGroupSizeOption::ThreadMapping mapping_type, float penalty,
    const BHWC& dst_shape, const int4& block_size) {
  WorkGroupSizeOption wg;
  wg.work_group_size = work_group_size;
  wg.thread_mapping = mapping_type;
  wg.penalty = penalty;
  if (mapping_type == WorkGroupSizeOption::ThreadMapping::kDefault) {
    wg.work_groups_count =
        GetGroupsCount(dst_shape, work_group_size, block_size);
  } else if (mapping_type ==
             WorkGroupSizeOption::ThreadMapping::kLinearSpatial) {
    wg.work_groups_count =
        GetGroupsCountForLinearWH(dst_shape, work_group_size, block_size);
  } else if (mapping_type == WorkGroupSizeOption::ThreadMapping::kLinearAll) {
    wg.work_groups_count =
        GetGroupsCountForLinearWHS(dst_shape, work_group_size, block_size);
  }
  return wg;
}

ConvGeneric::ConvParams GetConvParamsForA7A8(const AppleInfo& apple_info,
                                             bool x_kernel_is_1,
                                             bool y_kernel_is_1, int src_slices,
                                             const BHWC& dst_shape) {
  const int dst_slices = DivideRoundUp(dst_shape.c, 4);
  int blk_total_size = GetRecommendedBlockSize(apple_info, dst_shape);
  int3 block_size = int3(1, 1, 1);
  if (blk_total_size >= 4 && (dst_slices % 4 == 0 || dst_slices >= 16)) {
    block_size.z = 4;
    blk_total_size /= 4;
  } else if (blk_total_size >= 2 && (dst_slices % 2 == 0 || dst_slices >= 4)) {
    block_size.z = 2;
    blk_total_size /= 2;
  }
  if (blk_total_size >= 4) {
    block_size.x = 2;
    block_size.y = 2;
    blk_total_size /= 4;
  } else if (blk_total_size >= 2) {
    if (dst_shape.w % 2 != 0 && dst_shape.h % 2 == 0) {
      block_size.y = 2;
    } else {
      block_size.x = 2;
    }
    blk_total_size /= 2;
  }

  ConvGeneric::ConvParams params;
  params.weights_upload_type =
      ConvGeneric::WeightsUploadType::LOCAL_MEM_BY_THREADS;
  params.x_kernel_is_1 = x_kernel_is_1;
  params.y_kernel_is_1 = y_kernel_is_1;
  params.src_depth_loop_size = 1;
  params.block_size.x = block_size.x;
  params.block_size.y = block_size.y;
  params.block_size.z = 1;
  params.block_size.w = block_size.z;
  params.weights_layout = WeightsLayout::kOSpatialIOGroupO4I4;

  std::vector<WorkGroupSizeOption> options;
  options.push_back(CreateWorkGroupSizeOption(
      {8, 4, 1}, WorkGroupSizeOption::ThreadMapping::kDefault, 1.0f, dst_shape,
      params.block_size));
  if (!apple_info.IsFamilyApple1()) {
    options.push_back(CreateWorkGroupSizeOption(
        {4, 4, 1}, WorkGroupSizeOption::ThreadMapping::kDefault, 1.01f,
        dst_shape, params.block_size));
    options.push_back(CreateWorkGroupSizeOption(
        {4, 2, 1}, WorkGroupSizeOption::ThreadMapping::kDefault, 1.25f,
        dst_shape, params.block_size));
  }
  options.push_back(CreateWorkGroupSizeOption(
      {32, 1, 1}, WorkGroupSizeOption::ThreadMapping::kLinearSpatial, 1.0f,
      dst_shape, params.block_size));
  if (!apple_info.IsFamilyApple1()) {
    options.push_back(CreateWorkGroupSizeOption(
        {16, 1, 1}, WorkGroupSizeOption::ThreadMapping::kLinearSpatial, 1.01f,
        dst_shape, params.block_size));
    options.push_back(CreateWorkGroupSizeOption(
        {8, 1, 1}, WorkGroupSizeOption::ThreadMapping::kLinearSpatial, 1.25f,
        dst_shape, params.block_size));
    options.push_back(CreateWorkGroupSizeOption(
        {32, 1, 1}, WorkGroupSizeOption::ThreadMapping::kLinearAll, 3.1 * 1.0f,
        dst_shape, params.block_size));
    options.push_back(CreateWorkGroupSizeOption(
        {16, 1, 1}, WorkGroupSizeOption::ThreadMapping::kLinearAll, 3.1 * 1.01f,
        dst_shape, params.block_size));
    options.push_back(CreateWorkGroupSizeOption(
        {8, 1, 1}, WorkGroupSizeOption::ThreadMapping::kLinearAll, 3.1 * 1.25f,
        dst_shape, params.block_size));
  }

  float optimum = options[0].work_groups_count * options[0].penalty *
                  options[0].work_group_size.x * options[0].work_group_size.y *
                  options[0].work_group_size.z;
  int optimum_index = 0;
  for (int i = 1; i < options.size(); ++i) {
    float local_optimum = options[i].work_groups_count * options[i].penalty *
                          options[i].work_group_size.x *
                          options[i].work_group_size.y *
                          options[i].work_group_size.z;
    if (local_optimum < optimum) {
      optimum = local_optimum;
      optimum_index = i;
    }
  }

  WorkGroupSizeOption optimum_wg = options[optimum_index];
  if (optimum_wg.thread_mapping ==
      WorkGroupSizeOption::ThreadMapping::kLinearSpatial) {
    params.linear_spatial = true;
    params.linear_all = false;
    params.work_group_size = optimum_wg.work_group_size;
    params.work_group_launch_order = int3(1, 0, 2);
  } else if (optimum_wg.thread_mapping ==
             WorkGroupSizeOption::ThreadMapping::kLinearAll) {
    params.linear_spatial = false;
    params.linear_all = true;
    params.work_group_size = optimum_wg.work_group_size;
    params.work_group_launch_order = int3(0, 1, 2);
    params.weights_upload_type = ConvGeneric::WeightsUploadType::GLOBAL_MEM;
  } else {
    // default 3D workgroup
    params.linear_spatial = false;
    params.linear_all = false;
    params.work_group_size = optimum_wg.work_group_size;
    params.work_group_launch_order = int3(2, 0, 1);
  }
  int total_elements = params.block_size.x * params.block_size.y *
                       params.block_size.z * params.block_size.w;
  if (total_elements == 1) {
    if (src_slices % 4 == 0) {
      params.src_depth_loop_size = 4;
    } else if (src_slices % 2 == 0) {
      params.src_depth_loop_size = 2;
    }
  } else if (total_elements == 2) {
    if (src_slices % 2 == 0) {
      params.src_depth_loop_size = 2;
    }
  }
  if (params.src_depth_loop_size == src_slices) {
    params.need_src_loop = false;
  }
  if (params.block_size.w == dst_slices) {
    params.need_dst_loop = false;
  }
  const bool use_filters_constants =
      !params.need_dst_loop && !params.need_src_loop && params.x_kernel_is_1 &&
      params.y_kernel_is_1;
  if (use_filters_constants) {
    params.weights_upload_type = ConvGeneric::WeightsUploadType::CONSTANT_MEM;
  }

  return params;
}

ConvGeneric::ConvParams GetConvParamsForA9AndHigher(const AppleInfo& apple_info,
                                                    bool x_kernel_is_1,
                                                    bool y_kernel_is_1,
                                                    int src_slices,
                                                    const BHWC& dst_shape) {
  const int dst_slices = DivideRoundUp(dst_shape.c, 4);
  int blk_total_size = GetRecommendedBlockSize(apple_info, dst_shape);
  int3 block_size = int3(1, 1, 1);
  if (blk_total_size >= 2 && apple_info.IsBionic()) {
    if (dst_shape.h % 2 != 0 && dst_shape.w % 2 == 0) {
      block_size.x = 2;
    } else {
      block_size.y = 2;
    }
    blk_total_size /= 2;
  }
  if (blk_total_size >= 4 && (dst_slices % 4 == 0 || dst_slices >= 16)) {
    block_size.z = 4;
    blk_total_size /= 4;
  } else if (blk_total_size >= 2 && (dst_slices % 2 == 0 || dst_slices >= 4)) {
    block_size.z = 2;
    blk_total_size /= 2;
  }
  if (blk_total_size >= 4 && dst_slices == 3) {
    block_size.z = 3;
    blk_total_size /= 4;
  }

  ConvGeneric::ConvParams params;
  params.weights_upload_type = ConvGeneric::WeightsUploadType::GLOBAL_MEM;
  params.x_kernel_is_1 = x_kernel_is_1;
  params.y_kernel_is_1 = y_kernel_is_1;
  params.src_depth_loop_size = 1;
  params.block_size.x = block_size.x;
  params.block_size.y = block_size.y;
  params.block_size.z = 1;
  params.block_size.w = block_size.z;
  params.linear_spatial = false;
  params.linear_all = false;
  params.work_group_size = int3(8, 4, 1);
  params.work_group_launch_order = int3(2, 0, 1);
  params.weights_layout = WeightsLayout::kOSpatialIOGroupO4I4;
  int g1 = GetGroupsCount(dst_shape, params.work_group_size, params.block_size);
  int g2 = GetGroupsCountForLinearWH(dst_shape, {32, 1, 1}, params.block_size);
  int g3 = GetGroupsCountForLinearWHS(dst_shape, {32, 1, 1}, params.block_size);
  if (g2 < g1) {
    params.linear_spatial = true;
    params.work_group_size = int3(32, 1, 1);
    params.work_group_launch_order = int3(0, 1, 2);
  }
  float precise_threshold = apple_info.IsBionic() ? 1.0f : 1.04f;
  float precise_ratio = static_cast<float>(g2) / static_cast<float>(g3);
  if (precise_ratio > precise_threshold) {
    params.linear_spatial = false;
    params.linear_all = true;
    params.work_group_size = int3(32, 1, 1);
  }
  int total_elements = params.block_size.x * params.block_size.y *
                       params.block_size.z * params.block_size.w;
  if (total_elements == 1) {
    if (src_slices % 4 == 0) {
      params.src_depth_loop_size = 4;
    } else if (src_slices % 2 == 0) {
      params.src_depth_loop_size = 2;
    }
  } else if (total_elements == 2) {
    if (src_slices % 2 == 0) {
      params.src_depth_loop_size = 2;
    }
  }
  if (params.src_depth_loop_size == src_slices) {
    params.need_src_loop = false;
  }
  if (params.block_size.w == dst_slices) {
    params.need_dst_loop = false;
  }
  const bool use_filters_constants =
      !params.need_dst_loop && !params.need_src_loop && params.x_kernel_is_1 &&
      params.y_kernel_is_1;
  if (use_filters_constants) {
    params.weights_upload_type = ConvGeneric::WeightsUploadType::CONSTANT_MEM;
  }

  return params;
}

ConvGeneric::ConvParams ConvGeneric::GuessBestParamsApple(
    const GpuInfo& gpu_info, const OperationDef& definition, int src_depth,
    int dst_depth, bool x_kernel_is_1, bool y_kernel_is_1,
    bool different_weights_for_height, const BHWC& dst_shape) {
  if (gpu_info.apple_info.IsLocalMemoryPreferredOverGlobal()) {
    return GetConvParamsForA7A8(gpu_info.apple_info, x_kernel_is_1,
                                y_kernel_is_1, src_depth, dst_shape);
  } else {
    return GetConvParamsForA9AndHigher(gpu_info.apple_info, x_kernel_is_1,
                                       y_kernel_is_1, src_depth, dst_shape);
  }
}

ConvGeneric::ConvParams ConvGeneric::GuessBestParams(
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
    if (gpu_info.IsCL30OrHigher()) {
      work_group_size_ =
          int3(gpu_info.opencl_info.preferred_work_group_size_multiple, 1, 1);
    } else {
      work_group_size_ = int3(32, 1, 1);
    }
    if (different_weights_for_height) {
      work_group_launch_order_ = int3(2, 0, 1);
      conv_params.fixed_work_group_size = true;
    } else {
      conv_params.linear_spatial = true;
      work_group_launch_order_ = int3(1, 0, 2);
      conv_params.fixed_work_group_size = true;
    }
    conv_params.block_size = int4(1, 1, 1, 4);
    conv_params.src_depth_loop_size = 1;
    if (!gpu_info.IsApiOpenCl() ||
        (gpu_info.IsApiOpenCl() &&
         gpu_info.opencl_info.dedicated_local_memory)) {
      if (definition.precision == CalculationsPrecision::F32_F16) {
        conv_params.weights_upload_type =
            WeightsUploadType::LOCAL_MEM_BY_THREADS;
      } else {
        conv_params.weights_upload_type =
            WeightsUploadType::LOCAL_MEM_ASYNC_SUBGROUP;
      }
    } else {
      conv_params.weights_upload_type = WeightsUploadType::GLOBAL_MEM;
    }
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
    work_group_size_ = int3(8, 4, 1);
    work_group_launch_order_ = int3(0, 1, 2);
    conv_params.fixed_work_group_size = false;

    if (gpu_info.IsApiOpenCl()) {
      conv_params.weights_upload_type = WeightsUploadType::CONSTANT_MEM;
    } else {
      conv_params.weights_upload_type = WeightsUploadType::GLOBAL_MEM;
    }
    if (dst_depth % 4 == 0 || dst_depth >= 8) {
      conv_params.block_size = int4(2, 2, 1, 4);
    } else if (dst_depth % 2 == 0 || dst_depth >= 4) {
      conv_params.block_size = int4(4, 2, 1, 2);
    } else {
      conv_params.block_size = int4(4, 4, 1, 1);
    }
    auto reduce_block_size_wzyx = [](int4* block_size) {
      if (block_size->w % 2 == 0) {
        block_size->w /= 2;
      } else if (block_size->z % 2 == 0) {
        block_size->z /= 2;
      } else if (block_size->y % 2 == 0) {
        block_size->y /= 2;
      } else if (block_size->x % 2 == 0) {
        block_size->x /= 2;
      }
    };
    if (definition_.precision != CalculationsPrecision::F16) {
      reduce_block_size_wzyx(&conv_params.block_size);
    }
    if (dst_shape) {
      int task_size = dst_shape->w * dst_shape->b * dst_shape->h * dst_depth;
      float task_size_per_cu =
          static_cast<float>(task_size) / gpu_info.GetComputeUnitsCount();
      int block_size = conv_params.block_size.x * conv_params.block_size.y *
                       conv_params.block_size.w;
      float threads_per_cu = task_size_per_cu / block_size;
      float warps_per_cu = threads_per_cu / 64;
      if (warps_per_cu < 4.0f) {
        reduce_block_size_wzyx(&conv_params.block_size);
      }
      if (warps_per_cu < 2.0f) {
        reduce_block_size_wzyx(&conv_params.block_size);
      }
      if (warps_per_cu < 1.0f) {
        reduce_block_size_wzyx(&conv_params.block_size);
      }
      if (warps_per_cu < 0.5f) {
        reduce_block_size_wzyx(&conv_params.block_size);
      }
    }
    int block_size = conv_params.block_size.x * conv_params.block_size.y *
                     conv_params.block_size.w;
    conv_params.src_depth_loop_size = 1;
    if (block_size <= 4 && src_depth % 2 == 0) {
      conv_params.src_depth_loop_size = 2;
    }
    if (block_size <= 2 && src_depth % 4 == 0) {
      conv_params.src_depth_loop_size = 4;
    }
    if (block_size <= 1 && src_depth % 8 == 0) {
      conv_params.src_depth_loop_size = 8;
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
    if (dst_shape) {
      if (dst_shape->w == 1) {
        conv_params.block_size.y *= conv_params.block_size.x;
        conv_params.block_size.x = 1;
      }
      if (dst_shape->h == 1) {
        conv_params.block_size.x *= conv_params.block_size.y;
        conv_params.block_size.y = 1;
      }
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
    if (dst_shape) {
      const int wave_size = gpu_info.adreno_info.GetWaveSize(
          definition.precision == CalculationsPrecision::F16);
      const double task_size =
          1.0 * dst_shape->w * dst_shape->b * dst_shape->h * dst_depth;
      const double waves =
          task_size / gpu_info.GetComputeUnitsCount() / wave_size;
      if (waves <= 6.0f) {
        conv_params.block_size = int4(1, 1, 1, 1);
      } else if (waves <= 12.0f) {
        conv_params.block_size = int4(2, 1, 1, 1);
      } else if (waves <= 24.0f) {
        conv_params.block_size = int4(2, 1, 1, 2);
      } else {
        conv_params.block_size = int4(2, 2, 1, 2);
      }
    } else {
      conv_params.block_size = int4(2, 2, 1, 2);
    }
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
    conv_params.weights_upload_type = WeightsUploadType::TEXTURES_MEM_X4;
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
    conv_params.weights_upload_type = WeightsUploadType::LOCAL_MEM_BY_THREADS;
    if (gpu_info.IsApiMetal() &&
        definition.precision != CalculationsPrecision::F32_F16 &&
        gpu_info.metal_info.IsMslVersionEqualOrHigher(2)) {
      conv_params.weights_upload_type =
          WeightsUploadType::PRIVATE_MEM_SIMD_BROADCAST;
      conv_params.simd_size = 8;
    }
    if (gpu_info.IsApiOpenCl() &&
        definition.precision != CalculationsPrecision::F32_F16) {
      const bool supports_subgroups =
          gpu_info.SupportsExtension("cl_khr_subgroups") ||
          gpu_info.SupportsExtension("cl_intel_subgroups") ||
          gpu_info.opencl_info.IsCLVK();
      if (supports_subgroups) {
        const int kSubGroupSize = 16;
        const bool supports_subgroup_size_control =
            gpu_info.SupportsExtension("cl_intel_required_subgroup_size");
        int min_subgroup_size;
        auto min_subgroup_size_status =
            gpu_info.GetMinSubGroupSize(min_subgroup_size);
        if (supports_subgroup_size_control &&
            gpu_info.SupportsSubGroupWithSize(kSubGroupSize)) {
          conv_params.weights_upload_type =
              WeightsUploadType::PRIVATE_MEM_SIMD_BROADCAST;
          conv_params.simd_size = kSubGroupSize;
        } else if (supports_subgroup_size_control &&
                   min_subgroup_size_status.ok()) {
          conv_params.weights_upload_type =
              WeightsUploadType::PRIVATE_MEM_SIMD_BROADCAST;
          conv_params.simd_size = min_subgroup_size;
          work_group_size_ = int3(min_subgroup_size, 1, 1);
        } else {
          // no support of subgroup size control
          // only smallest subgroup size (8) can be used safely, otherwise
          // correctness can not be guaranteed
          // conv_params.weights_upload_type =
          //    WeightsUploadType::PRIVATE_MEM_SIMD_BROADCAST;
          // conv_params.simd_size = 8;
        }
      }
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
    BHWC output_shape = BHWC(1, 32, 32, 128);
    if (dst_shape) {
      output_shape = *dst_shape;
    }
    conv_params = GuessBestParamsApple(
        gpu_info, definition, src_depth, dst_depth, x_kernel_is_1,
        y_kernel_is_1, different_weights_for_height, output_shape);
    conv_params.fixed_work_group_size = true;
    work_group_size_ = conv_params.work_group_size;
    work_group_launch_order_ = conv_params.work_group_launch_order;
    conv_params.weights_data_type =
        DeduceDataTypeFromPrecision(definition.precision);
    conv_params.x_kernel_is_1 = x_kernel_is_1;
    conv_params.y_kernel_is_1 = y_kernel_is_1;
    conv_params.different_weights_for_height = different_weights_for_height;
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

ConvGeneric::ConvParams ConvGeneric::GuessBestParams(
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

ConvGeneric::ConvParams ConvGeneric::GuessBestParams(
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

  ConvGeneric::ConvParams result;
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

ConvGeneric::ConvParams ConvGeneric::GuessBestParams(
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

ConvGeneric::ConvParams ConvGeneric::GuessBestParams(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const FullyConnectedAttributes& attr, const BHWC* dst_shape) {
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  ConvGeneric::ConvParams params = GuessBestParams(
      gpu_info, definition, src_depth, dst_depth, true, true, false, dst_shape);
  work_group_size_.x *= work_group_size_.y;
  work_group_size_.y = 1;
  params.block_size.x *= params.block_size.y;
  params.block_size.y = 1;
  return params;
}

ConvGeneric::ConvParams ConvGeneric::GuessBestParamsPointwise(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OHWI& weights_shape, const BHWC* dst_shape) {
  const int dst_depth = DivideRoundUp(weights_shape.o, 4);
  const int src_depth = DivideRoundUp(weights_shape.i, 4);
  ConvGeneric::ConvParams params = GuessBestParams(
      gpu_info, definition, src_depth, dst_depth, true, true, true, dst_shape);
  params.block_size.x *= params.block_size.y;
  params.block_size.y = 1;
  work_group_size_.x *= work_group_size_.y;
  work_group_size_.y = 1;
  return params;
}

ConvGeneric CreateConvGeneric(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr,
                              const BHWC* dst_shape) {
  ConvGeneric result(definition, attr, gpu_info, dst_shape);
  result.GenerateCode(gpu_info);
  result.UploadData(attr.weights, attr.bias);
  return result;
}

ConvGeneric CreateConvGeneric(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              const FullyConnectedAttributes& attr,
                              const BHWC* dst_shape) {
  ConvGeneric result(definition, attr, gpu_info, dst_shape);
  result.GenerateCode(gpu_info);
  result.UploadData(attr.weights, attr.bias);
  return result;
}

ConvGeneric CreateConvGenericDynamicWeights(const GpuInfo& gpu_info,
                                            const OperationDef& definition,
                                            const Convolution2DAttributes& attr,
                                            const BHWC& weights_shape,
                                            const BHWC* dst_shape) {
  ConvGeneric result(definition, attr, weights_shape, gpu_info, dst_shape);
  result.GenerateCode(gpu_info);
  result.UploadBias(attr.bias);
  return result;
}

ConvGeneric CreateConvGenericBatchedMatMul(const GpuInfo& gpu_info,
                                           const OperationDef& definition,
                                           const OHWI& weights_shape,
                                           const BHWC* dst_shape) {
  ConvGeneric result(definition);
  result.conv_params_ = result.GuessBestParamsPointwise(
      gpu_info, definition, weights_shape, dst_shape);
  result.GenerateCode(gpu_info);
  tflite::gpu::Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(weights_shape.o);
  biases.data.resize(weights_shape.o, 0.0f);
  result.UploadBias(biases);
  return result;
}

ConvGeneric CreateConvGenericWino4x4To6x6(const GpuInfo& gpu_info,
                                          const OperationDef& definition,
                                          const Convolution2DAttributes& attr,
                                          const BHWC* dst_shape) {
  ConvGeneric result(definition);
  result.conv_params_ = result.GuessBestParamsPointwise(
      gpu_info, definition, attr.weights.shape, dst_shape);
  result.GenerateCode(gpu_info);
  result.UploadDataForWinograd4x4To6x6(attr.weights);
  return result;
}

ConvGeneric CreateConvGeneric3D(const GpuInfo& gpu_info,
                                const OperationDef& definition,
                                const Convolution3DAttributes& attr,
                                const BHWDC* dst_shape) {
  ConvGeneric result(definition, attr, gpu_info, dst_shape);
  result.GenerateCode(gpu_info);
  result.UploadWeights(attr.weights);
  result.UploadBias(attr.bias);
  return result;
}

}  // namespace gpu
}  // namespace tflite
