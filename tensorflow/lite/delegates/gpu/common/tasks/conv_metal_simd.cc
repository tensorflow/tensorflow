/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/conv_metal_simd.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"

namespace tflite {
namespace gpu {
namespace {
std::string GenerateDstCoords(const int3& work_group_launch_order,
                              bool linear_spatial, bool need_depth) {
  std::string c;
  int3 launch_remap;
  launch_remap[work_group_launch_order.x] = 0;
  launch_remap[work_group_launch_order.y] = 1;
  launch_remap[work_group_launch_order.z] = 2;
  if (linear_spatial) {
    if (work_group_launch_order[0] == 0) {
      c += "  int linear_spatial = GLOBAL_ID_0;\n";
    } else {
      c += "  int linear_spatial = GROUP_ID_" +
           std::to_string(launch_remap[0]) + " * GROUP_SIZE_0 + LOCAL_ID_0;\n";
    }
    if (need_depth) {
      c += "  int DST_X = linear_spatial % args.dst_tensor.Width();\n";
      c += "  linear_spatial = linear_spatial / args.dst_tensor.Width();\n";
      c += "  int DST_Y = linear_spatial % args.dst_tensor.Height();\n";
      c += "  int DST_Z = linear_spatial / args.dst_tensor.Height();\n";
    } else {
      c += "  int DST_Y = linear_spatial / args.dst_tensor.Width();\n";
      c += "  int DST_X = linear_spatial % args.dst_tensor.Width();\n";
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
    std::string global_id_1;
    if (work_group_launch_order[1] == 1) {
      global_id_1 = "GLOBAL_ID_1";
    } else {
      global_id_1 = "GROUP_ID_" + std::to_string(launch_remap[1]) +
                    " * GROUP_SIZE_1 + LOCAL_ID_1";
    }
    if (need_depth) {
      c += "  int linear_id_1 = " + global_id_1 + ";\n";
      c += "  int DST_Z = linear_id_1 / dst_tensor.Height();\n";
      c += "  int DST_Y = linear_id_1 % dst_tensor.Height();\n";
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

  return c;
}

std::string GenerateConvolution(
    const OperationDef& definition,
    const ConvolutionMetalSimd::ConvParams& conv_params) {
  std::string c;
  c += "#define MMA simdgroup_multiply_accumulate\n";
  const int spatial_threads = conv_params.GetSpatialThreadsCount();
  c += "#define SPATIAL_THREADS " + std::to_string(spatial_threads) + "\n";
  c += "MAIN_FUNCTION($0) {\n";
  c += GenerateDstCoords(conv_params.work_group_launch_order,
                         conv_params.linear_spatial,
                         definition.src_tensors[0].HasAxis(Axis::DEPTH));
  if (conv_params.slices_per_thread != 1) {
    c += "  DST_S *= " + std::to_string(conv_params.slices_per_thread) + ";\n";
  }
  c += "  device FLT4* f_offseted = args.weights.GetPtr() + DST_S * 4 * "
       "args.src_tensor.Slices();\n";
  const bool cache_weight = spatial_threads > 32;
  const int src_x4_slices = conv_params.GetX4SlicesCount();
  const int src_x8_slices = src_x4_slices / 2;
  const int dst_x4_slices = conv_params.slices_per_thread;
  const int dst_x8_slices = dst_x4_slices / 2;
  const int weights_tiles8x8_per_spatial = src_x8_slices * dst_x8_slices;
  const int weights_flt4_per_spatial = weights_tiles8x8_per_spatial * 8 * 8 / 4;
  if (conv_params.linear_spatial) {
    c += "  int spatial_id = LOCAL_ID_0;\n";
    c += "  int slice_id = LOCAL_ID_1;\n";
  } else {
    c += "  int spatial_id = LOCAL_ID_1 * GROUP_SIZE_0 + LOCAL_ID_0;\n";
    c += "  int slice_id = LOCAL_ID_2;\n";
  }
  c += "  int tid = slice_id * SPATIAL_THREADS + spatial_id;\n";
  if (cache_weight) {
    c += "  threadgroup FLT4 tmp_w[" +
         std::to_string(weights_flt4_per_spatial *
                        conv_params.GetX4SlicesCount()) +
         "];\n";
    c += "  threadgroup FLT* tmp_w_x1 = (threadgroup FLT*)tmp_w;\n";
    c += "  tmp_w_x1 += " + std::to_string(weights_flt4_per_spatial * 4) +
         " * slice_id;\n";
    c += "  threadgroup FLT4* tmp_w_x4 = (threadgroup FLT4*)tmp_w_x1;\n\n";
  } else {
    c += "  device FLT* f_offseted_x1 = (device FLT*)f_offseted;\n\n";
  }
  c += "  threadgroup FLT4 tmp_src[SPATIAL_THREADS * " +
       std::to_string(src_x4_slices) + "];\n";
  c += "  threadgroup FLT* tmp_src_x1 = (threadgroup FLT*)tmp_src;\n\n";
  c += "  // sp - spatial dimensions, ch - channels dimension\n";
  c += "  // indexing relative to simdgroup\n";
  for (int sp = 0; sp < 32; sp += 8) {
    const std::string sp_start = std::to_string(sp);
    const std::string sp_end = std::to_string(sp + 8);
    const std::string dst_name = "dst_sp" + sp_start + "_" + sp_end;
    for (int slice = 0; slice < dst_x8_slices; slice += 1) {
      const std::string sl_start = std::to_string(slice * 8);
      const std::string sl_end = std::to_string(slice * 8 + 8);
      c += "  simdgroup_matrix<FLT, 8, 8> " + dst_name + "_ch" + sl_start +
           "_" + sl_end + "(0.0f);\n";
    }
  }
  if (spatial_threads > 32) {
    c += "  int spatial_group = spatial_id / 32;\n";
    c += "  tmp_src_x1 += 8 * 8 * 4 * spatial_group;\n";
  }
  c += R"(
  int c_x = min(DST_X, args.src_tensor.Width() - 1);
  int c_y = min(DST_Y, args.src_tensor.Height() - 1);
)";
  if (definition.src_tensors[0].IsLinear()) {
    c += "  args.src_tensor.GetAddress(src_address, c_x, c_y, slice_id);\n";
  }
  c += R"(
  int tid2 = 0;
  if (tid < SPATIAL_THREADS) {
    tid2 = tid * 2 + 0;
  } else if (tid < SPATIAL_THREADS * 2) {
    tid2 = (tid - SPATIAL_THREADS) * 2 + 1;
  })";
  for (int src_s = 1; src_s < src_x8_slices; ++src_s) {
    c += " else if (tid < SPATIAL_THREADS * " + std::to_string(src_s * 2 + 1) +
         ") {\n";
    c += "    tid2 = (tid - SPATIAL_THREADS * " + std::to_string(src_s * 2) +
         ") * 2 + 0 + SPATIAL_THREADS * " + std::to_string(src_s * 2) + ";\n";
    c += "  } else if (tid < SPATIAL_THREADS * " +
         std::to_string(src_s * 2 + 2) + ") {\n";
    c += "    tid2 = (tid - SPATIAL_THREADS * " +
         std::to_string(src_s * 2 + 1) + ") * 2 + 1 + SPATIAL_THREADS * " +
         std::to_string(src_s * 2) + ";\n";
    c += "  }";
  }
  c += "\n\n";
  c += "  for (int s = 0; s < args.src_tensor.Slices(); s += " +
       std::to_string(src_x4_slices) + ") {\n";
  for (int src_s = 0; src_s < src_x8_slices; ++src_s) {
    const std::string src_range =
        "i" + std::to_string(src_s * 8) + "_" + std::to_string(src_s * 8 + 8);
    for (int dst_s = 0; dst_s < dst_x8_slices; ++dst_s) {
      const std::string dst_range =
          "o" + std::to_string(dst_s * 8) + "_" + std::to_string(dst_s * 8 + 8);
      const std::string w_name = "w_" + dst_range + "_" + src_range;
      c += "    simdgroup_matrix<FLT, 8, 8> " + w_name + ";\n";
    }
  }
  c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  if (cache_weight) {
    const int groups = weights_flt4_per_spatial / spatial_threads;
    const int reminder = weights_flt4_per_spatial % spatial_threads;
    for (int i = 0; i < groups; ++i) {
      c += "    tmp_w_x4[spatial_id + " + std::to_string(spatial_threads * i) +
           "] = f_offseted[spatial_id + " +
           std::to_string(spatial_threads * i) + "];\n";
    }
    if (reminder != 0) {
      c += "    if (spatial_id < " + std::to_string(reminder) + ") {\n";
      c += "      tmp_w_x4[spatial_id + " +
           std::to_string(spatial_threads * groups) +
           "] = f_offseted[spatial_id + " +
           std::to_string(spatial_threads * groups) + "];\n";
      c += "    }\n";
    }
  } else {
    for (int src_s = 0; src_s < src_x8_slices; ++src_s) {
      const std::string src_range =
          "i" + std::to_string(src_s * 8) + "_" + std::to_string(src_s * 8 + 8);
      for (int dst_s = 0; dst_s < dst_x8_slices; ++dst_s) {
        const std::string dst_range = "o" + std::to_string(dst_s * 8) + "_" +
                                      std::to_string(dst_s * 8 + 8);
        const std::string w_name = "w_" + dst_range + "_" + src_range;
        c += "    simdgroup_load(" + w_name + ", f_offseted_x1 + " +
             std::to_string((src_s * dst_x8_slices + dst_s) * 64) + ", 8);\n";
      }
    }
  }
  if (definition.src_tensors[0].IsLinear()) {
    c += "    tmp_src[tid2] = args.src_tensor.Read(src_address);\n";
  } else {
    c += "    tmp_src[tid2] = args.src_tensor.Read(c_x, c_y, s + slice_id);\n";
  }
  if (cache_weight) {
    c += "    f_offseted += 16 * " +
         std::to_string(src_x8_slices * dst_x8_slices) + ";\n";
  } else {
    c += "    f_offseted_x1 += 64 * " +
         std::to_string(src_x8_slices * dst_x8_slices) + ";\n";
  }
  if (definition.src_tensors[0].IsLinear()) {
    c += "    src_address += args.src_tensor.SliceStride() * " +
         std::to_string(src_x4_slices) + ";\n";
  }
  c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  if (cache_weight) {
    for (int src_s = 0; src_s < src_x8_slices; ++src_s) {
      const std::string src_range =
          "i" + std::to_string(src_s * 8) + "_" + std::to_string(src_s * 8 + 8);
      for (int dst_s = 0; dst_s < dst_x8_slices; ++dst_s) {
        const std::string dst_range = "o" + std::to_string(dst_s * 8) + "_" +
                                      std::to_string(dst_s * 8 + 8);
        const std::string w_name = "w_" + dst_range + "_" + src_range;
        c += "    simdgroup_load(" + w_name + ", tmp_w_x1 + " +
             std::to_string((src_s * dst_x8_slices + dst_s) * 64) + ", 8);\n";
      }
    }
  }
  c += "    simdgroup_matrix<FLT, 8, 8> mat_src;\n";
  const int spatial_x8_count = spatial_threads / 8;
  for (int src_s = 0; src_s < src_x8_slices; ++src_s) {
    const std::string src_s_range =
        std::to_string(src_s * 8) + "_" + std::to_string(src_s * 8 + 8);
    for (int sp = 0; sp < 32; sp += 8) {
      const std::string sp_range =
          std::to_string(sp) + "_" + std::to_string(sp + 8);
      const int src_tile_offset = src_s * spatial_x8_count + (sp / 8);
      c += "    simdgroup_load(mat_src, tmp_src_x1 + " +
           std::to_string(src_tile_offset * 64) + ", 8);  // loading sp[" +
           sp_range + "] src_ch[" + src_s_range + "]\n";
      for (int dst_s = 0; dst_s < dst_x8_slices; ++dst_s) {
        const std::string dst_s_range =
            std::to_string(dst_s * 8) + "_" + std::to_string(dst_s * 8 + 8);
        const std::string dst_name = "dst_sp" + sp_range + "_ch" + dst_s_range;
        const std::string w_name = "w_o" + dst_s_range + "_i" + src_s_range;
        c += "    MMA(" + dst_name + ", mat_src, " + w_name + ", " + dst_name +
             ");\n";
      }
    }
  }
  c += "  }\n";
  for (int slice = 0; slice < dst_x8_slices * 2; slice += 1) {
    c += "  FLT4 r" + std::to_string(slice) + " = INIT_FLT4(0.0);\n";
  }
  c += "  // transferring from simdgroup memory to private registers.\n";
  c += "  const int kSpatialGroupsCount = " + std::to_string(src_x4_slices) +
       ";\n";
  c += "  for (int i = 0; i < kSpatialGroupsCount; ++i) {\n";
  c += "    int spatial_id = tid - i * SPATIAL_THREADS;\n";
  c += "    bool current_spatial_group = spatial_id >= 0 && spatial_id < "
       "SPATIAL_THREADS;\n";
  for (int dst_s = 0; dst_s < dst_x8_slices; ++dst_s) {
    const std::string dst_range =
        "ch" + std::to_string(dst_s * 8) + "_" + std::to_string(dst_s * 8 + 8);
    c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    c += "    if (current_spatial_group) {\n";
    c += "      simdgroup_store(dst_sp0_8_" + dst_range + ", tmp_src_x1, 8);\n";
    c += "      simdgroup_store(dst_sp8_16_" + dst_range +
         ", tmp_src_x1 + 64, 8);\n";
    c += "      simdgroup_store(dst_sp16_24_" + dst_range +
         ", tmp_src_x1 + 64 * 2, 8);\n";
    c += "      simdgroup_store(dst_sp24_32_" + dst_range +
         ", tmp_src_x1 + 64 * 3, 8);\n";
    c += "    }\n";
    c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    c += "    if (current_spatial_group) {\n";
    c += "      r" + std::to_string(dst_s * 2 + 0) +
         " += tmp_src[spatial_id * 2 + 0];\n";
    c += "      r" + std::to_string(dst_s * 2 + 1) +
         " += tmp_src[spatial_id * 2 + 1];\n";
    c += "    }\n";
  }
  c += "  }\n";
  c += "  if (DST_X >= args.dst_tensor.Width() || DST_Y >= "
       "args.dst_tensor.Height()) {\n";
  c += "    return;\n";
  c += "  }\n";
  for (int slice = 0; slice < dst_x8_slices * 2; slice += 1) {
    const std::string dst_s = "DST_S + " + std::to_string(slice);
    const std::string r_name = "r" + std::to_string(slice);
    c += "  if (" + dst_s + " < args.dst_tensor.Slices()) {\n";
    c += "    " + r_name + " += args.biases.Read(" + dst_s + ");\n";
    c += "    args.dst_tensor.Write(" + r_name + ", DST_X, DST_Y, " + dst_s +
         ");\n";
    c += "  }\n";
  }
  c += "}\n";
  return c;
}

void OIToVecOIOGroupIO(const std::vector<float>& src, int o_size, int i_size,
                       int vec_size, int o_group_size,
                       std::vector<float>* dst) {
  int o_slices = DivideRoundUp(o_size, vec_size);
  int i_slices = DivideRoundUp(i_size, vec_size);
  int o_groups = DivideRoundUp(o_slices, o_group_size);
  dst->resize(o_slices * vec_size * i_slices * vec_size);
  for (int os = 0; os < o_groups; ++os) {
    for (int is = 0; is < i_slices; ++is) {
      for (int o_group = 0; o_group < o_group_size; ++o_group) {
        for (int sub_o = 0; sub_o < vec_size; ++sub_o) {
          for (int sub_i = 0; sub_i < vec_size; ++sub_i) {
            float value = 0.0f;
            int i_ch = is * vec_size + sub_i;
            int o_ch = (os * o_group_size + o_group) * vec_size + sub_o;
            if (i_ch < i_size && o_ch < o_size) {
              value = src[o_ch * i_size + i_ch];
            }
            (*dst)[(((os * i_slices + is) * o_group_size + o_group) * vec_size +
                    sub_i) *
                       vec_size +
                   sub_o] = value;
          }
        }
      }
    }
  }
}

std::vector<uint8_t> ReorderWeightsForConv(
    const tflite::gpu::Tensor<OHWI, DataType::FLOAT32>& weights,
    const DataType& weights_type, int dst_x8_slices) {
  std::vector<float> weights_gpu;
  OIToVecOIOGroupIO(weights.data, weights.shape.o, weights.shape.i, 8,
                    dst_x8_slices, &weights_gpu);
  std::vector<uint8_t> result(weights_gpu.size() * SizeOf(weights_type));
  if (weights_type == DataType::FLOAT32) {
    float* gpu_data = reinterpret_cast<float*>(result.data());
    for (int i = 0; i < weights_gpu.size(); ++i) {
      gpu_data[i] = weights_gpu[i];
    }
  } else {
    half* gpu_data = reinterpret_cast<half*>(result.data());
    for (int i = 0; i < weights_gpu.size(); ++i) {
      gpu_data[i] = weights_gpu[i];
    }
  }
  return result;
}

std::vector<uint8_t> ReorderBiasesForConv(
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases,
    const DataType& biases_type, int output_size) {
  std::vector<uint8_t> result(output_size * SizeOf(biases_type));
  if (biases_type == DataType::FLOAT32) {
    float* gpu_data = reinterpret_cast<float*>(result.data());
    for (int i = 0; i < output_size; ++i) {
      gpu_data[i] = i < biases.shape.v ? biases.data[i] : 0.0f;
    }
  } else {
    half* gpu_data = reinterpret_cast<half*>(result.data());
    for (int i = 0; i < output_size; ++i) {
      gpu_data[i] = i < biases.shape.v ? biases.data[i] : 0.0f;
    }
  }
  return result;
}

std::vector<int2> Get2DWorkgroupsEqualTo32() {
  return {{8, 4}, {16, 2}, {4, 8}, {32, 1}, {2, 16}, {1, 32}};
}

int Get2dGroupsCount(const BHWC& dst_shape, const int2 group_size) {
  int x_groups = DivideRoundUp(dst_shape.w * dst_shape.b, group_size.x);
  int y_groups = DivideRoundUp(dst_shape.h, group_size.y);
  return x_groups * y_groups;
}

int2 GetOptimalGroupSize(const BHWC& dst_shape) {
  const auto base_work_groups = Get2DWorkgroupsEqualTo32();
  int min_2d_work_groups = Get2dGroupsCount(dst_shape, base_work_groups[0]);
  int min_index = 0;
  for (int i = 1; i < base_work_groups.size(); ++i) {
    int groups_count = Get2dGroupsCount(dst_shape, base_work_groups[i]);
    if (groups_count < min_2d_work_groups) {
      min_2d_work_groups = groups_count;
      min_index = i;
    }
  }
  return base_work_groups[min_index];
}

}  // namespace

int3 ConvolutionMetalSimd::GetGridSize() const {
  const int task_size_x = dst_[0]->Width() * dst_[0]->Batch();
  const int task_size_y = dst_[0]->Height();
  const int task_size_z = dst_[0]->Depth();
  const int task_size_s =
      DivideRoundUp(dst_[0]->Slices(), params_.slices_per_thread);
  if (params_.linear_spatial) {
    return int3(task_size_x * task_size_y * task_size_z, task_size_s, 1);
  } else {
    return int3(task_size_x, task_size_y * task_size_z, task_size_s);
  }
}

ConvolutionMetalSimd CreateConvolutionMetalSimd(
    const OperationDef& definition, const BHWC& dst_shape,
    const Convolution2DAttributes& attr, const GpuInfo& gpu_info) {
  ConvolutionMetalSimd desc(definition);
  const int2 optimal_2d_group_size = GetOptimalGroupSize(dst_shape);
  const int groups2d_count = Get2dGroupsCount(dst_shape, optimal_2d_group_size);
  const int groups1d_count =
      DivideRoundUp(dst_shape.w * dst_shape.b * dst_shape.h, 32);
  if (groups1d_count < groups2d_count) {
    desc.params_.work_group_size = int3(32, 4, 1);
    desc.params_.work_group_launch_order = int3(0, 1, 2);
    desc.params_.linear_spatial = true;
  } else {
    desc.params_.work_group_size =
        int3(optimal_2d_group_size.x, optimal_2d_group_size.y, 4);
    desc.params_.work_group_launch_order = int3(0, 1, 2);
    desc.params_.linear_spatial = false;
  }
  desc.params_.slices_per_thread = 4;
  desc.params_.x_kernel_is_1 = true;
  desc.params_.y_kernel_is_1 = true;
  desc.params_.z_kernel_is_1 = true;
  desc.code_ = GenerateConvolution(definition, desc.params_);

  auto src_desc = definition.src_tensors[0];
  if (definition.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  desc.AddSrcTensor("src_tensor", src_desc);
  auto dst_desc = definition.dst_tensors[0];
  if (definition.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  desc.AddDstTensor("dst_tensor", dst_desc);

  auto weights_type = DeduceDataTypeFromPrecision(definition.precision);

  MemoryType mem_type = MemoryType::GLOBAL;

  if (definition.src_tensors.size() == 2) {
    // dynamic weights
    BufferDescriptor weights_desc;
    weights_desc.element_type = definition.src_tensors[1].data_type;
    weights_desc.element_size = 4;
    weights_desc.memory_type = mem_type;
    desc.AddSrcBuffer("weights", weights_desc);
  } else {
    BufferDescriptor weights_desc;
    weights_desc.element_type = weights_type;
    weights_desc.element_size = 4;
    weights_desc.memory_type = mem_type;
    weights_desc.data = ReorderWeightsForConv(
        attr.weights, weights_type, desc.params_.slices_per_thread / 2);
    weights_desc.size = weights_desc.data.size();
    desc.args_.AddObject("weights", absl::make_unique<BufferDescriptor>(
                                        std::move(weights_desc)));
  }

  BufferDescriptor bias_desc;
  bias_desc.element_type = weights_type;
  bias_desc.element_size = 4;
  bias_desc.memory_type = mem_type;
  bias_desc.data = ReorderBiasesForConv(attr.bias, weights_type,
                                        AlignByN(attr.weights.shape.o, 4 * 4));
  bias_desc.size = bias_desc.data.size();
  desc.args_.AddObject(
      "biases", absl::make_unique<BufferDescriptor>(std::move(bias_desc)));

  desc.work_group_size_ = desc.params_.work_group_size;
  desc.work_group_launch_order_ = desc.params_.work_group_launch_order;
  if (desc.params_.linear_spatial) {
    desc.grid_dimension_ = 2;
  } else {
    desc.grid_dimension_ = 3;
  }

  return desc;
}

bool IsConvolutionMetalSimdSupported(const GpuInfo& gpu_info,
                                     const OperationDef& definition,
                                     const Convolution2DAttributes& attr) {
  if (!gpu_info.IsApple() || !gpu_info.apple_info.IsSIMDMatMulSupported()) {
    return false;
  }
  const bool genuine_1x1 =
      attr.weights.shape.w == 1 && attr.weights.shape.h == 1 &&
      attr.dilations.w == 1 && attr.dilations.h == 1 && attr.strides.w == 1 &&
      attr.strides.h == 1 && attr.padding.prepended.w == 0 &&
      attr.padding.prepended.h == 0 && attr.padding.appended.w == 0 &&
      attr.padding.appended.h == 0 && attr.groups == 1;
  const int src_slices = DivideRoundUp(attr.weights.shape.i, 4);
  const int dst_slices = DivideRoundUp(attr.weights.shape.o, 4);
  return genuine_1x1 && src_slices % 4 == 0 && dst_slices % 16 == 0;
}

bool IsGoodTaskSizeForAppleConvSimd(const BHWC& dst_shape,
                                    const GpuInfo& gpu_info) {
  const uint64_t task_size_spatial = dst_shape.b * dst_shape.h * dst_shape.w;
  const uint64_t wave_size = 32;
  const double useful_part = static_cast<double>(task_size_spatial) /
                             AlignByN(task_size_spatial, wave_size);
  if (useful_part < 0.625) {
    return false;
  }
  const double task_size_slices = DivideRoundUp(dst_shape.c, 16);
  const double task_size = task_size_spatial * task_size_slices;
  const double task_size_per_cu = task_size / gpu_info.GetComputeUnitsCount();
  const double waves_per_cu = task_size_per_cu / wave_size;
  return waves_per_cu >= 8.0;
}

}  // namespace gpu
}  // namespace tflite
