/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/kernels/mean.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
// total_wg_size is pot, dimensions is {1, 2, 3}
int3 GetWGSizeFromTotalSize(int total_wg_size, int dimensions) {
  if (dimensions == 1) {
    return {total_wg_size, 1, 1};
  } else if (dimensions == 2) {
    int3 wg_size = int3(1, 1, 1);
    while (total_wg_size != 1) {
      if (total_wg_size >= 4) {
        wg_size.x *= 2;
        wg_size.y *= 2;
        total_wg_size /= 4;
      } else {
        // total_wg_size == 2
        wg_size.x *= 2;
        total_wg_size /= 2;
      }
    }
    return wg_size;
  } else {
    // dimensions == 3
    int3 wg_size = int3(1, 1, 1);
    while (total_wg_size != 1) {
      if (total_wg_size >= 8) {
        wg_size.x *= 2;
        wg_size.y *= 2;
        wg_size.z *= 2;
        total_wg_size /= 8;
      } else if (total_wg_size == 4) {
        wg_size.x *= 2;
        wg_size.y *= 2;
        total_wg_size /= 4;
      } else {
        // total_wg_size == 2
        wg_size.x *= 2;
        total_wg_size /= 2;
      }
    }
    return wg_size;
  }
}

int GetWGTotalSize(const GpuInfo& gpu_info) {
  // total_wg_size must be power of 2 and >= 4;
  int total_wg_size = 256;
  if (gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno3xx()) {
    total_wg_size = 128;
  }
  if (gpu_info.IsMali()) {
    const MaliInfo& mali_info = gpu_info.mali_info;
    if (mali_info.IsMaliT6xx() || mali_info.IsMaliT7xx() ||
        mali_info.IsMaliT8xx()) {
      total_wg_size = 32;
    } else {
      total_wg_size = 64;
    }
  }
  return total_wg_size;
}

bool HasAxis(const std::vector<Axis>& axis, Axis a) {
  for (const auto& a2 : axis) {
    if (a2 == a) {
      return true;
    }
  }
  return false;
}

}  // namespace

Reduce::Reduce(const std::set<Axis>& axis_to_reduce, OperationType op_type,
               const OperationDef& definition, const GpuInfo& gpu_info)
    : GPUOperation(definition) {
  std::vector<Axis> ordered_axis_to_reduce;
  for (const auto& a :
       {Axis::CHANNELS, Axis::DEPTH, Axis::HEIGHT, Axis::WIDTH, Axis::BATCH}) {
    if (axis_to_reduce.count(a)) {
      ordered_axis_to_reduce.push_back(a);
    }
  }
  int wg_dims = std::min(3, static_cast<int>(ordered_axis_to_reduce.size()));
  const int total_wg_size = GetWGTotalSize(gpu_info);
  work_group_size_ = GetWGSizeFromTotalSize(total_wg_size, wg_dims);
  code_ = GetReduceKernelCode(definition_, work_group_size_,
                              ordered_axis_to_reduce, op_type);
}

Reduce::Reduce(Reduce&& operation) : GPUOperation(std::move(operation)) {}

Reduce& Reduce::operator=(Reduce&& operation) {
  if (this != &operation) {
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string Reduce::GetReduceKernelCode(const OperationDef& op_def,
                                        const int3& work_group_size,
                                        const std::vector<Axis>& axis_to_reduce,
                                        OperationType op_type) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddFloat("inv_multiplier_1");
  args_.AddFloat("inv_multiplier_2");
  args_.AddFloat("mask_x");
  args_.AddFloat("mask_y");
  args_.AddFloat("mask_z");
  args_.AddFloat("mask_w");

  std::set<Axis> axis_to_leave;
  const std::vector<Axis> all_axis = {Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH,
                                      Axis::CHANNELS, Axis::BATCH};
  for (const auto& a : all_axis) {
    if (op_def.dst_tensors[0].HasAxis(a)) {
      if (!HasAxis(axis_to_reduce, a)) {
        axis_to_leave.insert(a);
      }
    }
  }
  int wg_dims = std::min(3, static_cast<int>(axis_to_reduce.size()));
  const bool channels_reductin = HasAxis(axis_to_reduce, Axis::CHANNELS);

  std::string c = GetCommonDefines(op_def.precision);
  const std::string wg_x = std::to_string(work_group_size.x);
  const std::string wg_y = std::to_string(work_group_size.y);
  const std::string wg_z = std::to_string(work_group_size.z);
  const int wg_total_size =
      work_group_size.x * work_group_size.y * work_group_size.z;
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  __local float4 accum[" + std::to_string(wg_total_size) + "];\n";
  if (wg_dims == 1) {
    c += "  int local_x = get_local_id(0);\n";
    c += "  int local_id = local_x;\n";
  } else if (wg_dims == 2) {
    c += "  int local_x = get_local_id(0);\n";
    c += "  int local_y = get_local_id(1);\n";
    c += "  int local_id = local_y * " + wg_x + " + local_x;\n";
  } else if (wg_dims == 3) {
    c += "  int local_x = get_local_id(0);\n";
    c += "  int local_y = get_local_id(1);\n";
    c += "  int local_z = get_local_id(2);\n";
    c += "  int local_id = (local_z * " + wg_y + " + local_y) * " + wg_x +
         " + local_x;\n";
  }
  if (axis_to_leave.count(Axis::WIDTH)) {
    if (axis_to_leave.count(Axis::BATCH)) {
      c += "  int linear_id = get_group_id(0);\n";
      c += "  int DST_X = linear_id / args.dst_tensor.Batch();\n";
      c += "  int DST_B = linear_id % args.dst_tensor.Batch();\n";
    } else {
      c += "  int DST_X = get_group_id(0);\n";
    }
  } else if (axis_to_leave.count(Axis::BATCH)) {
    c += "  int DST_B = get_group_id(0);\n";
  }
  if (axis_to_leave.count(Axis::HEIGHT)) {
    if (axis_to_leave.count(Axis::DEPTH)) {
      c += "  int linear_id = get_group_id(1);\n";
      c += "  int DST_Y = linear_id % args.dst_tensor.Height();\n";
      c += "  int DST_Z = linear_id / args.dst_tensor.Height();\n";
    } else {
      c += "  int DST_Y = get_group_id(1);\n";
    }
  } else if (axis_to_leave.count(Axis::DEPTH)) {
    c += "  int DST_Z = get_group_id(1);\n";
  }
  if (axis_to_leave.count(Axis::CHANNELS)) {
    c += "  int DST_S = get_group_id(2);\n";
  }
  std::map<Axis, std::string> axis_to_selector = {
      {Axis::BATCH, "Batch()"},     {Axis::WIDTH, "Width()"},
      {Axis::HEIGHT, "Height()"},   {Axis::DEPTH, "Depth()"},
      {Axis::CHANNELS, "Slices()"},
  };
  std::map<Axis, std::string> axis_to_coord = {
      {Axis::BATCH, "B"}, {Axis::WIDTH, "X"},    {Axis::HEIGHT, "Y"},
      {Axis::DEPTH, "Z"}, {Axis::CHANNELS, "S"},
  };
  std::string dst_check;
  for (auto& axis : axis_to_leave) {
    if (!dst_check.empty()) {
      dst_check += " || ";
    }
    dst_check += "DST_" + axis_to_coord[axis] + " >= args.dst_tensor." +
                 axis_to_selector[axis];
  }
  if (!dst_check.empty()) {
    c += "  if (" + dst_check + ") return;\n";
  }
  c += "  float4 reducer = (float4)(0.0f);\n";
  const std::vector<std::string> local_ids = {"local_x", "local_y", "local_z"};
  const std::vector<std::string> local_sizes = {wg_x, wg_y, wg_z};
  std::map<Axis, std::string> src_coords;
  for (const auto& a : all_axis) {
    if (op_def.dst_tensors[0].HasAxis(a)) {
      src_coords[a] = "DST_" + axis_to_coord[a];
    } else {
      src_coords[a] = "0";
    }
  }
  for (int i = 0; i < axis_to_reduce.size(); ++i) {
    const auto& axis = axis_to_reduce[i];
    const int index = axis_to_reduce.size() - 1 - i;
    const std::string first = index < wg_dims ? local_ids[index] : "0";
    const std::string step = index < wg_dims ? local_sizes[index] : "1";
    const std::string src_coord = "SRC_" + axis_to_coord[axis];
    src_coords[axis] = src_coord;
    c += "  for (int " + src_coord + " = " + first + "; " + src_coord +
         " < args.src_tensor." + axis_to_selector[axis] + "; " + src_coord +
         " += " + step + ") {\n";
    if (axis == Axis::CHANNELS) {
      c += "    bool last = SRC_S == args.src_tensor.Slices() - 1;\n";
      c += "    float4 mask_a = last ? (float4)(args.mask_x, args.mask_y, "
           "args.mask_z, args.mask_w) : (float4)(1.0f);\n";
      c += "    float4 mask_b = last ? (float4)(0.0f) : (float4)(0.0f);\n";
    }
  }
  std::string src_coordinates;
  for (const auto& a : all_axis) {
    if (op_def.src_tensors[0].HasAxis(a)) {
      if (!src_coordinates.empty()) {
        src_coordinates += ", ";
      }
      src_coordinates += src_coords[a];
    }
  }
  c += "    float4 src_val = args.src_tensor.Read<float>(" + src_coordinates +
       ");\n";
  if (channels_reductin) {
    c += "    reducer += src_val * mask_a + mask_b;\n";
  } else {
    c += "    reducer += src_val;\n";
  }
  for (int i = 0; i < axis_to_reduce.size(); ++i) {
    c += "  }\n";
  }
  if (op_type == OperationType::MEAN) {
    c += "  reducer *= args.inv_multiplier_1;\n";
  }
  c += "  accum[local_id] = reducer;\n";
  c += "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  const int total_size =
      work_group_size.x * work_group_size.y * work_group_size.z;
  int offset = 1;
  int reminder = total_size / 4;
  for (; reminder >= 8; reminder /= 4, offset *= 4) {
    c += "  if (local_id < " + std::to_string(reminder) + ") {\n";
    c += "    int t = local_id * " + std::to_string(offset * 4) + ";\n";
    c += "    float4 sum = accum[t + " + std::to_string(offset) + "];\n";
    c += "    sum += accum[t + " + std::to_string(offset * 2) + "];\n";
    c += "    sum += accum[t + " + std::to_string(offset * 3) + "];\n";
    c += "    accum[t] += sum;\n";
    c += "  }\n";
    c += "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  }
  c += "  float4 sum = accum[0];\n";
  reminder *= 4;
  for (int i = 1; i < reminder; ++i) {
    c += "  sum += accum[" + std::to_string(offset * i) + "];\n";
  }
  if (channels_reductin) {
    c += "  sum.x += sum.y + sum.z + sum.w;\n";
  }
  if (op_type == OperationType::MEAN) {
    c += "  sum *= args.inv_multiplier_2;\n";
  }
  c += "  FLT4 result = TO_FLT4(sum);\n";
  std::string dst_coordinates;
  for (const auto& a : all_axis) {
    if (op_def.dst_tensors[0].HasAxis(a)) {
      if (!dst_coordinates.empty()) {
        dst_coordinates += ", ";
      }
      if (axis_to_leave.count(a)) {
        dst_coordinates += "DST_" + axis_to_coord[a];
      } else {
        dst_coordinates += "0";
      }
    }
  }
  c += "  args.dst_tensor.Write(result, " + dst_coordinates + ");\n";
  c += "}\n";
  return c;
}

absl::Status Reduce::BindArguments(ArgumentsBinder* args) {
  const double total_src_elements = 1.0 * src_[0]->Batch() * src_[0]->Width() *
                                    src_[0]->Height() * src_[0]->Depth() *
                                    src_[0]->Channels();
  const double total_dst_elements = 1.0 * dst_[0]->Batch() * dst_[0]->Width() *
                                    dst_[0]->Height() * dst_[0]->Depth() *
                                    dst_[0]->Channels();
  const double reduction_size = total_src_elements / total_dst_elements;
  const double size_0 =
      work_group_size_.x * work_group_size_.y * work_group_size_.z;
  const double size_1 = reduction_size / size_0;
  RETURN_IF_ERROR(args->SetFloat("inv_multiplier_1", 1.0 / size_1));
  RETURN_IF_ERROR(args->SetFloat("inv_multiplier_2", 1.0 / size_0));
  float4 mask = GetMaskForLastPlane(src_[0]->Channels());
  RETURN_IF_ERROR(args->SetFloat("mask_x", mask.x));
  RETURN_IF_ERROR(args->SetFloat("mask_y", mask.y));
  RETURN_IF_ERROR(args->SetFloat("mask_z", mask.z));
  RETURN_IF_ERROR(args->SetFloat("mask_w", mask.w));
  return absl::OkStatus();
}

int3 Reduce::GetGridSize() const {
  const int grid_x = work_group_size_.x * dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = work_group_size_.y * dst_[0]->Height() * dst_[0]->Depth();
  const int grid_z = work_group_size_.z * dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

Reduce CreateReduce(const std::set<Axis>& axis_to_reduce, OperationType op_type,
                    const OperationDef& definition, const GpuInfo& gpu_info) {
  return Reduce(axis_to_reduce, op_type, definition, gpu_info);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
