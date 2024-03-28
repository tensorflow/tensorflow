/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/group_norm_var.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

namespace {
int GetMaximumWGTotalSize(const GpuInfo& gpu_info) {
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

//max_total_wg_size is pot
int3 GetMaximumPossibleWGSize(const std::vector<int>& ordered_sizes,
                              int max_total_wg_size) {
  int3 wg_size = int3(1, 1, 1);
  int wg_size_total = 1;
  // Make sure that a minimum number of reductions happens inside the loop over
  // reduction dims. Otherwise, the reduction size could equal the number of
  // workgroups and the inner loop would just copy the values to the reducer,
  // which is inefficient.
  const int minimum_loop_reductions = 4;
  int total_loop_reductions = 1;
  for (int i = ordered_sizes.size() - 1; i >= 0; i--) {
    const int wg_index = ordered_sizes.size() - 1 - i;
    if (wg_index >= 3) {
      return wg_size;
    }
    int loop_reductions_dim = 1;
    while (ordered_sizes[i] >= wg_size[wg_index] * 2 * loop_reductions_dim) {
      // Don't increase the work group size of this dim until we have at least
      // 'minimum_loop_reductions' reductions.
      if (total_loop_reductions < minimum_loop_reductions) {
        total_loop_reductions *= 2;
        loop_reductions_dim *= 2;
        continue;
      }
      wg_size_total *= 2;
      if (wg_size_total > max_total_wg_size) {
        return wg_size;
      }
      wg_size[wg_index] *= 2;
    }
  }
  return wg_size;
}

DataType GetAccumType(DataType src_type) {
  if (src_type == DataType::FLOAT32 || src_type == DataType::FLOAT16) {
    return DataType::FLOAT32;
  } else if (src_type == DataType::INT32 || src_type == DataType::INT16 ||
             src_type == DataType::INT8) {
    return DataType::INT32;
  } else if (src_type == DataType::UINT32 || src_type == DataType::UINT16 ||
             src_type == DataType::UINT8) {
    return DataType::UINT32;
  } else {
    return src_type;
  }
}


}  // namespace

std::string GroupNormVar::GetGroupNormVarCode(const OperationDef& op_def, 
                                              const GpuInfo& gpu_info,
                                              const int3& work_group_size,
                                              std::vector<int>& axis_to_reduce) {
                                                  
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddSrcTensor("mean_tensor", op_def.src_tensors[1]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);


  std::vector<int> axis_to_leave{3};
  const bool channels_reduction = false;
  
  int wg_dims = 0;
  if(use_wg_reduction_) {
    if(work_group_size.y == 1 && work_group_size.z == 1) {
      wg_dims = 1;
    } else if(work_group_size.z == 1) {
      wg_dims = 2;
    } else {
      wg_dims = 3;
    }
  }

  auto get_global_id = [&](int i) {
    if(use_wg_reduction_) {
      return "GROUP_ID_" + std::to_string(i);
    }else{
      return "GLOBAL_ID_" + std::to_string(i);
    }
  };

  const std::string accum_type_decl = "FLT4";


  std::string c;
  const std::string wg_x = std::to_string(work_group_size.x);
  const std::string wg_y = std::to_string(work_group_size.y);
  const std::string wg_z = std::to_string(work_group_size.z);

  const int wg_total_size = 
    work_group_size.x * work_group_size.y * work_group_size.z;
   
  c += "MAIN_FUNCTION($0) {\n";
  if (use_wg_reduction_) {
    c+= "  __local " + accum_type_decl + " accum[" + std::to_string(wg_total_size) + "];\n";
    if (wg_dims == 1) {
      c += "  int local_x = LOCAL_ID_0;\n";
      c += "  int local_id = local_x;\n";
    } else if (wg_dims == 2) {
      c += "  int local_x = LOCAL_ID_0;\n";
      c += "  int local_y = LOCAL_ID_1;\n";
      c += "  int local_id = local_y * " + wg_x + " + local_x;\n";
    } else if (wg_dims == 3) {
      c += "  int local_x = LOCAL_ID_0;\n";
      c += "  int local_y = LOCAL_ID_1;\n";
      c += "  int local_z = LOCAL_ID_2;\n";
      c += "  int local_id = (local_z * " + wg_y + " + local_y) * " + wg_x +
           " + local_x;\n";
    }
  }

  c += "  int DST_S = " + get_global_id(2) + ";\n";
  c += "  if (DST_S >= args.dst_tensor.Slices()) return;\n";

  std::map<int, std::string> axis_to_coord = {
      {0, "B"}, {2, "X"}, {1, "Y"}, {3, "S"},
  };

  std::map<int, std::string> axis_to_selector = {
      {0, "Batch()"}, {2, "Width()"}, {1, "Height()"}, {3, "Slices()"},
  };

  // calculate the two means (i'm assuming channels/groups is more that 4);
  c += "  int group_size = args.src_tensor.Channels() / args.num_groups;\n";
  c += "  int divisor = group_size * args.src_tensor.Height() * args.src_tensor.Width();\n";

  c += "  int group_id_1 = (DST_S*4) / group_size;\n";
  c += "  int group_id_2 = group_id_1 + 1;\n";
  c += "  float mean_array[4];\n";

  // calculating the mean for group1
  c += "  int left_index = group_id_1 * group_size;\n";
  c += "  int right_index = left_index + group_size;\n"; 
  c += "  float mean1 = 0.0f;\n";
  c += "  for(int i=left_index; i < right_index; i++){\n";
  c += "    int s_idx = i/4;\n";
  c += "    FLT4 value = args.mean_tensor.Read(0,0,s_idx);\n";
  c += "    mean_array[0] = value.x;\n";
  c += "    mean_array[1] = value.y;\n";
  c += "    mean_array[2] = value.z;\n";
  c += "    mean_array[3] = value.w;\n";
  c += "    mean1 = mean1 + mean_array[i % 4];\n";
  c += "  }\n";
  c += "  mean1 = mean1 / divisor;\n";

  // calculating the mean for group2
  c += "  left_index = group_id_2 * group_size;\n";
  c += "  right_index = left_index + group_size;\n"; 
  c += "  float mean2 = 0.0f;\n";
  c += "  for(int i=left_index; i < right_index; i++){\n";
  c += "    int s_idx = i/4;\n";
  c += "    FLT4 value = args.mean_tensor.Read(0,0,s_idx);\n";
  c += "    mean_array[0] = value.x;\n";
  c += "    mean_array[1] = value.y;\n";
  c += "    mean_array[2] = value.z;\n";
  c += "    mean_array[3] = value.w;\n";
  c += "    mean2 = mean2 + mean_array[i % 4];\n";
  c += "  }\n";
  c += "  mean2 = mean2 / divisor;\n";

  // now i have to calculate the mean for the 4 index;
  c += " float mean_group[2] = {mean1, mean2};\n";
  c += " int idx1 = ((DST_S*4)/group_size) - group_id_1;\n";
  c += " int idx2 = ((DST_S*4+1)/group_size) - group_id_1;\n";
  c += " int idx3 = ((DST_S*4+2)/group_size) - group_id_1;\n";
  c += " int idx4 = ((DST_S*4+3)/group_size) - group_id_1;\n";
  c += " FLT4 mean = (FLT4){mean_group[idx1], mean_group[idx2], mean_group[idx3], mean_group[idx4]};\n";

  c += "  " + accum_type_decl + " reducer = " + "INIT_FLT4(0);\n";
  // no channel reduction
  const std::vector<std::string> local_ids = {"local_x", "local_y", "local_z"};
  const std::vector<std::string> local_sizes = {wg_x, wg_y, wg_z};

  axis_to_reduce = {1, 2, 0};

  for (int i=0; i< axis_to_reduce.size(); i++) {
    const auto axis = axis_to_reduce[i];
    const int index = axis_to_reduce.size() - 1 - i;
    const std::string first = index < wg_dims ? local_ids[index] : "0";
    const std::string step = index < wg_dims ? local_sizes[index] : "1";
    const std::string src_coord = "SRC_" + axis_to_coord[axis];

    c += "  for (int " + src_coord + " = " + first + "; " + src_coord + 
         " < args.src_tensor." + axis_to_selector[axis] + "; " + src_coord + 
         " += " + step + ") {\n";
  }

  c += "    FLT4 src_val = args.src_tensor.Read(SRC_X, SRC_Y, DST_S);\n";
  c += "    reducer = reducer + ((src_val - mean)*(src_val - mean));\n";

  for (int i = 0; i < axis_to_reduce.size(); i++) {
    c += "  }\n";
  }

  if(use_wg_reduction_) {
    c += "  accum[local_id] = reducer;\n";
    c += "  LOCAL_MEM_BARRIER;\n";

    const int total_size = 
        work_group_size.x * work_group_size.y * work_group_size.z;
    int offset = 1;
    int remainder = total_size / 4;

    for(; remainder>=8; remainder /= 4, offset *= 4) {
      c += "  if(local_id < " + std::to_string(remainder) + ") {\n";
      c += "    int t = local_id * " + std::to_string(offset*4) + ";\n";
      c += "    " + accum_type_decl + "  reduced = accum[t + " + 
           std::to_string(offset) + "];\n";
      c += "    reduced = reduced + accum[t + " + std::to_string(offset*2) + "];\n";
      c += "    reduced = reduced + accum[t + " + std::to_string(offset*3) + "];\n";
      c += "    accum[t] = accum[t] + reduced;\n";
      c += "  }\n";
      c += "  LOCAL_MEM_BARRIER;\n";   
    }
    c += "  if (local_id != 0) return;\n";
    c += "  reducer = accum[0];\n";
    remainder *= 4;
    for(int i=1; i< remainder; ++i) {
      c += "  reducer = reducer + accum[" + std::to_string(offset*i) + "];\n";
    }

  }
  c += "  args.src_tensor::type result = reducer;\n";

  c += "  args.dst_tensor.Write(result, 0, 0, DST_S);\n";
  c += "}\n";
  return c;
}

GroupNormVar::GroupNormVar(const BHWC& shape, 
                           const OperationDef& definition,
                           const GpuInfo& gpu_info) : GPUOperation(definition) {

  std::vector<int> axis_to_reduce{0, 1, 2};
  std::vector<int> axis_sizes{shape.b, shape.h, shape.w};

  const int max_total_wg_size = GetMaximumWGTotalSize(gpu_info);
  int3 current_wg_size = GetMaximumPossibleWGSize(axis_sizes, max_total_wg_size);

  int current_wg_size_total = 
      current_wg_size.x * current_wg_size.y * current_wg_size.z;

  int threshold = max_total_wg_size / 4;
  if (gpu_info.IsApple()) {
    threshold = 16;
  }

  if (current_wg_size_total < threshold) {
    use_wg_reduction_ = false;
  }else{
    use_wg_reduction_ = true;
    work_group_size_ = {current_wg_size.z, current_wg_size.y, current_wg_size.x};
  }

  code_ = GetGroupNormVarCode(definition_, gpu_info, 
                               work_group_size_, axis_to_reduce);
}

absl::Status GroupNormVar::BindArguments(ArgumentsBinder* args) {
  //needed for reduce mean;
  const double total_src_elements = 1.0 * src_[0]->Batch() * src_[0]->Width() *
                                    src_[0]->Height() * src_[0]->Depth() *
                                    src_[0]->Channels();
  const double total_dst_elements = 1.0 * dst_[0]->Batch() * dst_[0]->Width() *
                                    dst_[0]->Height() * dst_[0]->Depth() *
                                    dst_[0]->Channels();
  const double reduction_size = total_src_elements / total_dst_elements;
  if (use_wg_reduction_) {
    const double size_0 =
        work_group_size_.x * work_group_size_.y * work_group_size_.z;
    const double size_1 = reduction_size / size_0;
  }

  return absl::OkStatus();
}

int3 GroupNormVar::GetGridSize() const {

  int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  int grid_y = dst_[0]->Height() * dst_[0]->Depth();
  int grid_z = dst_[0]->Slices();

  if (use_wg_reduction_) {
    grid_x *= work_group_size_.x;
    grid_y *= work_group_size_.y;
    grid_z *= work_group_size_.z;
  }

  return int3(grid_x, grid_y, grid_z);
}

void GroupNormVar::GetPossibleKernelWorkGroups(TuningType tuning_type,
                                         const GpuInfo& gpu_info,
                                         const KernelInfo& kernel_info,
                                         std::vector<int3>* work_groups) const {
  if (use_wg_reduction_) {
    work_groups->push_back(work_group_size_);
  } else {
    GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info, grid_size_,
                          work_groups);
  }
}


GroupNormVar::GroupNormVar(GroupNormVar&& operation)
    : GPUOperation(std::move(operation)),
      use_wg_reduction_(operation.use_wg_reduction_) {}

GroupNormVar& GroupNormVar::operator=(GroupNormVar&& operation) {
  if (this != &operation) {
    use_wg_reduction_ = operation.use_wg_reduction_;
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}


GroupNormVar CreateGroupNormVar(const GroupNormVarAttributes& attr,
                                 const BHWC& shape,
                                 const GpuInfo& gpu_info,
                                 const OperationDef& op_def) {
  auto gpu_op = GroupNormVar(shape, op_def, gpu_info);
  
  gpu_op.args_.AddInt("num_groups", attr.num_groups);

  return gpu_op;
}

}  // namespace gpu
}  // namespace tflite