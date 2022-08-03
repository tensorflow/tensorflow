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

#include "tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_3x3.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

ConvolutionTransposed3x3::ConvolutionTransposed3x3(
    const OperationDef& definition, const GpuInfo& gpu_info, int2 padding)
    : GPUOperation(definition), padding_(padding) {
  work_group_size_ = int3(8, 4, 1);
  work_group_launch_order_ = int3(2, 0, 1);
  if (gpu_info.IsApple()) {
    if (gpu_info.apple_info.IsBionic()) {
      weights_upload_type_ = WeightsUploadType::GLOBAL_MEM;
    } else {
      weights_upload_type_ = WeightsUploadType::LOCAL_MEM_BY_THREADS;
    }
  } else if (gpu_info.IsPowerVR()) {
    weights_upload_type_ = WeightsUploadType::LOCAL_MEM_ASYNC;
  } else if (gpu_info.IsNvidia() || gpu_info.IsIntel()) {
    weights_upload_type_ = WeightsUploadType::LOCAL_MEM_BY_THREADS;
  } else if (gpu_info.IsAMD()) {
    weights_upload_type_ = WeightsUploadType::CONSTANT_MEM;
  } else {
    weights_upload_type_ = WeightsUploadType::GLOBAL_MEM;
  }
  if (gpu_info.IsApple()) {
    weights_layout_ = WeightsLayout::kOICustomSpatialO4I4;
  } else {
    weights_layout_ = WeightsLayout::kOICustomSpatialI4O4;
  }
  code_ = GenerateConvolutionTransposedCode(gpu_info, definition_,
                                            weights_upload_type_, padding_,
                                            work_group_launch_order_);
  if (definition_.precision == CalculationsPrecision::F16 &&
      gpu_info.IsPowerVR()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
}

std::string ConvolutionTransposed3x3::GenerateConvolutionTransposedCode(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    ConvolutionTransposed3x3::WeightsUploadType weights_upload_type,
    int2 padding, int3 work_group_launch_order) {
  auto src_desc = op_def.src_tensors[0];
  AddSrcTensor("src_tensor", src_desc);
  AddDstTensor("dst_tensor", op_def.src_tensors[0]);

  if (op_def.src_tensors.size() == 2) {
    // dynamic weights
    BufferDescriptor desc;
    desc.element_type = op_def.src_tensors[1].GetDataType();
    desc.element_size = 4;
    desc.memory_type =
        weights_upload_type ==
                ConvolutionTransposed3x3::WeightsUploadType::CONSTANT_MEM
            ? MemoryType::CONSTANT
            : MemoryType::GLOBAL;
    AddSrcBuffer("weights", desc);
  }

  args_.AddInt("filter_offset");
  args_.AddInt("padding_x");
  args_.AddInt("padding_y");

  const bool need_local_mem =
      weights_upload_type ==
          ConvolutionTransposed3x3::WeightsUploadType::LOCAL_MEM_BY_THREADS ||
      weights_upload_type ==
          ConvolutionTransposed3x3::WeightsUploadType::LOCAL_MEM_ASYNC;

  std::string c;
  if (GetWeightsDescription().IsI4O4()) {
    switch (op_def.precision) {
      case CalculationsPrecision::F32:
      case CalculationsPrecision::F16:
        c += "#define CONV(R, SRC, F) \\\n";
        c += "  R += SRC.x * weights_cache[F]; \\\n";
        c += "  R += SRC.y * weights_cache[F + 1]; \\\n";
        c += "  R += SRC.z * weights_cache[F + 2]; \\\n";
        c += "  R += SRC.w * weights_cache[F + 3];   \n";
        break;
      case CalculationsPrecision::F32_F16:
        c += "#define CONV(R, SRC, F) \\\n";
        c += "  R += TO_ACCUM_TYPE(SRC.x * weights_cache[F] + SRC.y * "
             "weights_cache[F + 1] + SRC.z * weights_cache[F + 2] + SRC.w * "
             "weights_cache[F + 3]);\n";
        break;
    }
  } else {
    // O4I4
    c += "#define CONV(R, SRC, F) \\\n";
    c += "  R.x += dot(SRC, weights_cache[F]); \\\n";
    c += "  R.y += dot(SRC, weights_cache[F + 1]); \\\n";
    c += "  R.z += dot(SRC, weights_cache[F + 2]); \\\n";
    c += "  R.w += dot(SRC, weights_cache[F + 3]);   \n";
  }

  const int wg_total_size =
      work_group_size_.x * work_group_size_.y * work_group_size_.z;
  const std::string barrier =
      wg_total_size == 32 && gpu_info.IsWaveSizeEqualTo32()
          ? "SIMD_LOCAL_MEM_BARRIER"
          : "LOCAL_MEM_BARRIER";
  const std::string weights_space =
      weights_upload_type ==
              ConvolutionTransposed3x3::WeightsUploadType::CONSTANT_MEM
          ? "__constant"
          : "__global";

  if (gpu_info.IsApiOpenCl()) {
    c += "__attribute__((reqd_work_group_size(8, 4, 1)))\n";
  }
  c += "MAIN_FUNCTION($0) {\n";
  int3 launch_remap;
  launch_remap[work_group_launch_order.x] = 0;
  launch_remap[work_group_launch_order.y] = 1;
  launch_remap[work_group_launch_order.z] = 2;
  auto GetGlobalID = [&](int id) {
    std::string result;
    const std::string sid = std::to_string(id);
    if (work_group_launch_order[id] == id) {
      return "GLOBAL_ID_" + sid;
    } else {
      return "GROUP_ID_" + std::to_string(launch_remap[id]) + " * GROUP_SIZE_" +
             sid + " + LOCAL_ID_" + sid;
    }
  };
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = " + GetGlobalID(0) + ";\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = " + GetGlobalID(0) + ";\n";
  }
  c += "  int DST_X = X * 2;\n";
  c += "  int SRC_X = X + args.padding_x;\n";
  c += "  int Y = " + GetGlobalID(1) + ";\n";
  c += "  int DST_Y = Y * 2;\n";
  c += "  int SRC_Y = Y + args.padding_y;\n";
  c += "  int Z = " + GetGlobalID(2) + ";\n";
  if (!need_local_mem) {
    c += "  if (DST_X >= args.dst_tensor.Width() || DST_Y >= "
         "args.dst_tensor.Height() || Z >= args.dst_tensor.Slices()) return;\n";
  }
  c += "  ACCUM_FLT4 r0 = INIT_ACCUM_FLT4(0.0f);\n";
  c += "  ACCUM_FLT4 r1 = INIT_ACCUM_FLT4(0.0f);\n";
  c += "  ACCUM_FLT4 r2 = INIT_ACCUM_FLT4(0.0f);\n";
  c += "  ACCUM_FLT4 r3 = INIT_ACCUM_FLT4(0.0f);\n";
  c += "  int f_offset = Z * args.filter_offset;\n";
  if (need_local_mem) {
    c += "  __local FLT4 weights_cache[36];\n";
  }
  if (weights_upload_type ==
      ConvolutionTransposed3x3::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    c += "  int local_id = LOCAL_ID_1 * 8 + LOCAL_ID_0;\n";
  }
  if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
    c += "  bool in_x0 = SRC_X >= 0 && SRC_X < args.src_tensor.Width();\n";
    c += "  bool in_x1 = SRC_X + 1 >= 0 && SRC_X + 1 < "
         "args.src_tensor.Width();\n";
  }
  if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
    c += "  bool in_y0 = SRC_Y >= 0 && SRC_Y < args.src_tensor.Height();\n";
    c += "  bool in_y1 = SRC_Y + 1 >= 0 && SRC_Y + 1 < "
         "args.src_tensor.Height();\n";
  }
  auto generate_check = [&](int x, int y) {
    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT};
    const std::vector<std::string> names{"in_x" + std::to_string(x),
                                         "in_y" + std::to_string(y)};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_desc.HasAxis(axis) &&
          !src_desc.SupportsZeroClamp(axis, gpu_info)) {
        if (!check.empty()) {
          check += " && ";
        }
        check += names[i];
      }
    }
    return check;
  };
  if (src_desc.IsLinear()) {
    if (src_desc.ReturnsZeroForNegOneRead(gpu_info)) {
      c += "  int addr_0 = args.src_tensor.GetAddress(SRC_X, SRC_Y, 0);\n";
      c += "  int addr_1 = args.src_tensor.GetAddress(SRC_X + 1, SRC_Y, 0);\n";
      c += "  int addr_2 = args.src_tensor.GetAddress(SRC_X, SRC_Y + 1, 0);\n";
      c += "  int addr_3 = args.src_tensor.GetAddress(SRC_X+1, SRC_Y+1, 0);\n";
      c += "  addr_0 = select(-1, addr_0, (in_x0 && in_y0));\n";
      c += "  addr_1 = select(-1, addr_1, (in_x1 && in_y0));\n";
      c += "  addr_2 = select(-1, addr_2, (in_x0 && in_y1));\n";
      c += "  addr_3 = select(-1, addr_3, (in_x1 && in_y1));\n";
      c += "  int dz_0 = select(0, args.src_tensor.SliceStride(), (in_x0 && "
           "in_y0));\n";
      c += "  int dz_1 = select(0, args.src_tensor.SliceStride(), (in_x1 && "
           "in_y0));\n";
      c += "  int dz_2 = select(0, args.src_tensor.SliceStride(), (in_x0 && "
           "in_y1));\n";
      c += "  int dz_3 = select(0, args.src_tensor.SliceStride(), (in_x1 && "
           "in_y1));\n";
    } else {
      c += "  int xc0 = clamp(SRC_X, 0, args.src_tensor.Width() - 1);\n";
      c += "  int xc1 = clamp(SRC_X + 1, 0, args.src_tensor.Width() - 1);\n";
      c += "  int yc0 = clamp(SRC_Y, 0, args.src_tensor.Height() - 1);\n";
      c += "  int yc1 = clamp(SRC_Y + 1, 0, args.src_tensor.Height() - 1);\n";
      c += "  int addr_0 = args.src_tensor.GetAddress(xc0, yc0, 0);\n";
      c += "  int addr_1 = args.src_tensor.GetAddress(xc1, yc0, 0);\n";
      c += "  int addr_2 = args.src_tensor.GetAddress(xc0, yc1, 0);\n";
      c += "  int addr_3 = args.src_tensor.GetAddress(xc1, yc1, 0);\n";
      c += "  int dz = args.src_tensor.SliceStride();\n";
    }
  }
  auto read_src = [&](int x, int y) {
    if (src_desc.IsLinear()) {
      const std::string id = std::to_string(y * 2 + x);
      const std::string addr = "addr_" + std::to_string(y * 2 + x);
      if (src_desc.ReturnsZeroForNegOneRead(gpu_info)) {
        return "args.src_tensor.Read(" + addr + "); " + addr + " += dz_" + id +
               ";\n";
      } else {
        return "args.src_tensor.Read(" + addr + ") * INIT_FLT(in_x" +
               std::to_string(x) + " && in_y" + std::to_string(y) + "); " +
               addr + " += dz;\n";
      }
    } else {
      std::string check = generate_check(x, y);
      if (!check.empty()) {
        check = " * INIT_FLT(" + check + ")";
      }
      return "args.src_tensor.Read(SRC_X + " + std::to_string(x) +
             ", SRC_Y + " + std::to_string(y) + ", s)" + check + ";\n";
    }
  };
  const int padding_x_rem = abs(padding.x) % 2;
  const int padding_y_rem = abs(padding.y) % 2;
  std::vector<std::pair<int, int>> permutation;
  if (padding_x_rem == 1 && padding_y_rem == 1) {
    permutation = {{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 2},
                   {3, 0}, {3, 1}, {3, 2}, {3, 3}};
  } else if (padding_x_rem == 0 && padding_y_rem == 1) {
    permutation = {{0, 0}, {0, 1}, {1, 1}, {2, 0}, {2, 1},
                   {2, 2}, {2, 3}, {3, 1}, {3, 3}};
  } else if (padding_x_rem == 1 && padding_y_rem == 0) {
    permutation = {{0, 0}, {0, 2}, {1, 0}, {1, 1}, {1, 2},
                   {1, 3}, {2, 2}, {3, 2}, {3, 3}};
  } else {  // padding_x_rem == 0 && padding_y_rem == 0
    permutation = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1},
                   {1, 3}, {2, 2}, {2, 3}, {3, 3}};
  }
  c += "  for (int s = 0; s < args.src_tensor.Slices(); ++s) {\n";
  if (need_local_mem) {
    c += "    " + barrier + ";\n";
  }
  if (weights_upload_type ==
      ConvolutionTransposed3x3::WeightsUploadType::LOCAL_MEM_ASYNC) {
    c += "    async_work_group_copy(weights_cache, "
         "args.weights.GetPtr(f_offset), 36, "
         "0);\n";
  } else if (weights_upload_type ==
             ConvolutionTransposed3x3::WeightsUploadType::
                 LOCAL_MEM_BY_THREADS) {
    c += "    weights_cache[local_id] = args.weights.Read(f_offset + "
         "local_id);\n";
    c += "    if (local_id < 4) {\n";
    c += "      weights_cache[local_id + 32] = args.weights.Read(f_offset + "
         "local_id + "
         "32);\n";
    c += "    };\n";
  } else {  // GLOBAL_MEM/CONSTANT_MEM
    c += "    " + weights_space +
         " FLT4* weights_cache = args.weights.GetPtr(f_offset);\n";
  }
  c += "    FLT4 src0 = " + read_src(0, 0);
  c += "    FLT4 src1 = " + read_src(1, 0);
  c += "    FLT4 src2 = " + read_src(0, 1);
  c += "    FLT4 src3 = " + read_src(1, 1);
  c += "    f_offset += 36;\n";
  if (need_local_mem) {
    c += "    " + barrier + ";\n";
  }
  for (int i = 0; i < 9; ++i) {
    const std::string r_name = "r" + std::to_string(permutation[i].first);
    const std::string s_name = "src" + std::to_string(permutation[i].second);
    const std::string w_name = std::to_string(i * 4);
    c += "    CONV(" + r_name + ", " + s_name + ", " + w_name + ");\n";
  }
  c += "  }\n";
  if (need_local_mem) {
    c += "  if (DST_X >= args.dst_tensor.Width() || DST_Y >= "
         "args.dst_tensor.Height() || Z >= args.dst_tensor.Slices()) return;\n";
  }
  c += "  FLT4 bias_val = args.biases.Read(Z);\n";
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 2; ++x) {
      const std::string s_x = std::to_string(x);
      const std::string s_y = std::to_string(y);
      const std::string id = std::to_string(y * 2 + x);
      const std::string x_c = "DST_X + " + s_x;
      const std::string y_c = "DST_Y + " + s_y;
      c += "  if (" + x_c + " < args.dst_tensor.Width() && " + y_c +
           " < args.dst_tensor.Height()) {\n";
      c += "    FLT4 res0 = TO_FLT4(r" + id + ") + bias_val;\n";
      c += "    args.dst_tensor.Write(res0, " + x_c + ", " + y_c + ", Z);\n";
      c += "  }\n";
    }
  }
  c += "}\n";
  return c;
}

absl::Status ConvolutionTransposed3x3::BindArguments(ArgumentsBinder* args) {
  RETURN_IF_ERROR(args->SetInt("filter_offset", 4 * 9 * src_[0]->Slices()));
  const int padding_x =
      padding_.x >= 1 ? (padding_.x - 1) / 2 : (padding_.x - 2) / 2;
  const int padding_y =
      padding_.y >= 1 ? (padding_.y - 1) / 2 : (padding_.y - 2) / 2;
  RETURN_IF_ERROR(args->SetInt("padding_x", padding_x));
  return args->SetInt("padding_y", padding_y);
}

void ConvolutionTransposed3x3::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info, std::vector<int3>* work_groups) const {
  if (weights_upload_type_ == WeightsUploadType::LOCAL_MEM_ASYNC ||
      weights_upload_type_ == WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    work_groups->push_back(work_group_size_);
    return;
  }
  GetPossibleWorkGroupsConv(tuning_type, gpu_info, kernel_info, grid_size_,
                            work_groups);
}

int3 ConvolutionTransposed3x3::GetGridSize() const {
  const int grid_x = DivideRoundUp(dst_[0]->Width(), 2) * dst_[0]->Batch();
  const int grid_y = DivideRoundUp(dst_[0]->Height(), 2);
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int> ConvolutionTransposed3x3::GetSpatialWeightsRemap() const {
  const int padding_x_rem = abs(padding_.x) % 2;
  const int padding_y_rem = abs(padding_.y) % 2;

  std::vector<int> remap;
  if (padding_x_rem == 1 && padding_y_rem == 1) {
    return std::vector<int>{4, 5, 3, 7, 1, 8, 6, 2, 0};
  } else if (padding_x_rem == 0 && padding_y_rem == 1) {
    return std::vector<int>{5, 3, 4, 8, 6, 2, 0, 7, 1};
  } else if (padding_x_rem == 1 && padding_y_rem == 0) {
    return std::vector<int>{7, 1, 8, 6, 2, 0, 4, 5, 3};
  } else {  // padding_x_rem == 0 && padding_y_rem == 0
    return std::vector<int>{8, 6, 2, 0, 7, 1, 5, 3, 4};
  }
}

void ConvolutionTransposed3x3::UploadWeights(
    const tflite::gpu::Tensor<OHWI, DataType::FLOAT32>& weights) {
  const auto weights_desc = GetWeightsDescription();
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  BufferDescriptor desc;
  desc.element_type = weights_desc.type;
  desc.element_size = 4;
  desc.memory_type =
      weights_upload_type_ ==
              ConvolutionTransposed3x3::WeightsUploadType::CONSTANT_MEM
          ? MemoryType::CONSTANT
          : MemoryType::GLOBAL;
  desc.size = flt_count * SizeOf(desc.element_type);
  desc.data.resize(desc.size);

  RearrangeWeights(weights, weights_desc, absl::MakeSpan(desc.data));

  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(desc)));
}

bool IsConvolutionTransposed3x3Supported(
    const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr) {
  return attr.weights.shape.w == 3 && attr.weights.shape.h == 3 &&
         attr.stride.w == 2 && attr.stride.h == 2;
}

ConvolutionTransposed3x3 CreateConvolutionTransposed3x3(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr) {
  const int2 padding = int2(attr.padding.prepended.w, attr.padding.prepended.h);
  ConvolutionTransposed3x3 result(definition, gpu_info, padding);
  result.UploadWeights(attr.weights);

  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(attr.bias);
  result.args_.AddObject(
      "biases", std::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return result;
}

ConvolutionTransposed3x3 CreateConvolutionTransposed3x3DynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr) {
  OperationDef new_def = definition;
  new_def.src_tensors = {
      definition.src_tensors[0]};  // leaving only src_tensor def, weights defs
                                   // will be added later
  const DataType weights_type = definition.GetDataType();
  // add 1 src_tensor(buffer) for weights
  new_def.src_tensors.push_back(
      {weights_type, TensorStorageType::BUFFER, Layout::HWC});

  const int2 padding = int2(attr.padding.prepended.w, attr.padding.prepended.h);
  ConvolutionTransposed3x3 result(new_def, gpu_info, padding);

  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::TEXTURE_2D;
  desc.element_type = new_def.GetDataType();
  desc.UploadLinearData(attr.bias);
  result.args_.AddObject(
      "biases", std::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return result;
}

}  // namespace gpu
}  // namespace tflite
