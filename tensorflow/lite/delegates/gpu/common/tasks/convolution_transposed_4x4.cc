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

#include "tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_4x4.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
ConvolutionTransposed4x4::WeightsUploadType GetBestWeightsUploadType(
    const GpuInfo& gpu_info) {
  ConvolutionTransposed4x4::WeightsUploadType weights_upload_type =
      ConvolutionTransposed4x4::WeightsUploadType::GLOBAL_MEM;
  if (gpu_info.IsPowerVR()) {
    weights_upload_type =
        ConvolutionTransposed4x4::WeightsUploadType::LOCAL_MEM_ASYNC;
  } else if (gpu_info.IsNvidia() || gpu_info.IsIntel()) {
    weights_upload_type =
        ConvolutionTransposed4x4::WeightsUploadType::LOCAL_MEM_BY_THREADS;
  } else if (gpu_info.IsAMD()) {
    weights_upload_type =
        ConvolutionTransposed4x4::WeightsUploadType::CONSTANT_MEM;
  } else {
    weights_upload_type =
        ConvolutionTransposed4x4::WeightsUploadType::GLOBAL_MEM;
  }
  return weights_upload_type;
}
}  // namespace

ConvolutionTransposed4x4::ConvolutionTransposed4x4(
    const OperationDef& definition, const GpuInfo& gpu_info)
    : GPUOperation(definition) {
  work_group_size_ = int3(8, 4, 1);
  if (gpu_info.IsApple()) {
    work_group_launch_order_ = int3(2, 0, 1);
  }

  if (gpu_info.IsApple()) {
    weights_layout_ = WeightsLayout::kOICustomSpatialO4I4;
  } else {
    weights_layout_ = WeightsLayout::kOICustomSpatialI4O4;
  }

  code_ = GenerateConvolutionTransposedCode(gpu_info, definition_,
                                            GetBestWeightsUploadType(gpu_info));
  if (definition_.precision == CalculationsPrecision::F16 &&
      gpu_info.IsPowerVR()) {
    compiler_options_.push_back(CompilerOptions::kClPowervrFp16);
  }
}

std::string ConvolutionTransposed4x4::GenerateConvolutionTransposedCode(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    WeightsUploadType weights_upload_type) {
  auto src_desc = op_def.src_tensors[0];
  src_desc.SetAddressMode(AddressMode::kZero);
  if (op_def.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  AddSrcTensor("src_tensor", src_desc);

  auto dst_desc = op_def.dst_tensors[0];
  if (op_def.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  AddDstTensor("dst_tensor", dst_desc);

  if (op_def.src_tensors.size() == 2) {
    // dynamic weights
    BufferDescriptor desc;
    desc.element_type = op_def.src_tensors[1].data_type;
    desc.element_size = 4;
    desc.memory_type =
        weights_upload_type ==
                ConvolutionTransposed4x4::WeightsUploadType::CONSTANT_MEM
            ? MemoryType::CONSTANT
            : MemoryType::GLOBAL;
    AddSrcBuffer("weights", desc);
  }

  args_.AddInt("filter_offset");

  const bool need_local_mem =
      weights_upload_type ==
          ConvolutionTransposed4x4::WeightsUploadType::LOCAL_MEM_BY_THREADS ||
      weights_upload_type ==
          ConvolutionTransposed4x4::WeightsUploadType::LOCAL_MEM_ASYNC;

  const int wg_total_size =
      work_group_size_.x * work_group_size_.y * work_group_size_.z;
  const std::string barrier =
      wg_total_size == 32 && gpu_info.IsWaveSizeEqualTo32()
          ? "SIMD_LOCAL_MEM_BARRIER"
          : "LOCAL_MEM_BARRIER";

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

  const std::string weights_space =
      weights_upload_type ==
              ConvolutionTransposed4x4::WeightsUploadType::CONSTANT_MEM
          ? "__constant"
          : "__global";

  const std::string pixel_stride =
      op_def.IsBatchSupported() ? "args.dst_tensor.Batch()" : "1";
  if (gpu_info.IsApiOpenCl()) {
    c += "__attribute__((reqd_work_group_size(8, 4, 1)))\n";
  }
  c += "MAIN_FUNCTION($0) {\n";
  std::string grid_coords[3];
  int3 launch_remap;
  launch_remap[work_group_launch_order_.x] = 0;
  launch_remap[work_group_launch_order_.y] = 1;
  launch_remap[work_group_launch_order_.z] = 2;
  if (work_group_launch_order_[0] == 0) {
    grid_coords[0] = "GLOBAL_ID_0";
  } else {
    grid_coords[0] = "(GROUP_ID_" + std::to_string(launch_remap[0]) +
                     " * GROUP_SIZE_0 + LOCAL_ID_0);\n";
  }
  if (work_group_launch_order_[1] == 1) {
    grid_coords[1] = "GLOBAL_ID_1";
  } else {
    grid_coords[1] = "(GROUP_ID_" + std::to_string(launch_remap[1]) +
                     " * GROUP_SIZE_1 + LOCAL_ID_1);\n";
  }
  if (work_group_launch_order_[2] == 2) {
    grid_coords[2] = "GLOBAL_ID_2";
  } else {
    grid_coords[2] = "(GROUP_ID_" + std::to_string(launch_remap[2]) +
                     " * GROUP_SIZE_2 + LOCAL_ID_2);\n";
  }
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = " + grid_coords[0] + ";\n";
    c += "  int X0 = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
  }
  c += "  int X = " + grid_coords[0] + ";\n";
  c += "  int Y = " + grid_coords[1] + ";\n";
  c += "  int Z = " + grid_coords[2] + ";\n";
  if (!need_local_mem) {
    if (op_def.IsBatchSupported()) {
      c += "  if (X0 * 2 * args.dst_tensor.Batch() > args.dst_tensor.Width() "
           "|| Y * 2 > args.dst_tensor.Height() || Z "
           ">= args.dst_tensor.Slices()) return;\n";
    } else {
      c += "  if (X * 2 > args.dst_tensor.Width() || Y * 2 > "
           "args.dst_tensor.Height() || Z >= args.dst_tensor.Slices()) "
           "return;\n";
    }
  }
  c += "  ACCUM_FLT4 r0 = INIT_ACCUM_FLT4(0.0f);\n";
  c += "  ACCUM_FLT4 r1 = INIT_ACCUM_FLT4(0.0f);\n";
  c += "  ACCUM_FLT4 r2 = INIT_ACCUM_FLT4(0.0f);\n";
  c += "  ACCUM_FLT4 r3 = INIT_ACCUM_FLT4(0.0f);\n";
  c += "  int f_offset = Z * args.filter_offset;\n";
  if (need_local_mem) {
    c += "  __local FLT4 weights_cache[64];\n";
  }
  if (weights_upload_type ==
      ConvolutionTransposed4x4::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    c += "  int local_id = LOCAL_ID_1 * 8 + LOCAL_ID_0;\n";
  }
  const std::string prev_x = "X - " + pixel_stride;
  if (!src_desc.SupportsZeroClamp(Axis::WIDTH)) {
    c += "  bool in_x0 = " + prev_x + " >= 0 && " + prev_x +
         " < args.src_tensor.Width();\n";
    c += "  bool in_x1 = X >= 0 && X < args.src_tensor.Width();\n";
  }
  if (!src_desc.SupportsZeroClamp(Axis::HEIGHT)) {
    c += "  bool in_y0 = Y - 1 >= 0 && Y - 1 < args.src_tensor.Height();\n";
    c += "  bool in_y1 = Y >= 0 && Y < args.src_tensor.Height();\n";
  }
  auto generate_check = [&](int x, int y) {
    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT};
    const std::vector<std::string> names{"in_x" + std::to_string(x),
                                         "in_y" + std::to_string(y)};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_desc.HasAxis(axis) && !src_desc.SupportsZeroClamp(axis)) {
        if (!check.empty()) {
          check += " && ";
        }
        check += names[i];
      }
    }
    return check;
  };
  if (src_desc.IsLinear()) {
    if (src_desc.ReturnsZeroForNegOneRead()) {
      c += "  args.src_tensor.GetAddress(addr_0, " + prev_x + ", Y - 1, 0);\n";
      c += "  args.src_tensor.GetAddress(addr_1, X, Y - 1, 0);\n";
      c += "  args.src_tensor.GetAddress(addr_2, " + prev_x + ", Y, 0);\n";
      c += "  args.src_tensor.GetAddress(addr_3, X, Y, 0);\n";
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
      c += "  int xc0 = clamp(" + prev_x +
           ", 0, args.src_tensor.Width() - 1);\n";
      c += "  int xc1 = clamp(X, 0, args.src_tensor.Width() - 1);\n";
      c += "  int yc0 = clamp(Y - 1, 0, args.src_tensor.Height() - 1);\n";
      c += "  int yc1 = clamp(Y, 0, args.src_tensor.Height() - 1);\n";
      c += "  args.src_tensor.GetAddress(addr_0, xc0, yc0, 0);\n";
      c += "  args.src_tensor.GetAddress(addr_1, xc1, yc0, 0);\n";
      c += "  args.src_tensor.GetAddress(addr_2, xc0, yc1, 0);\n";
      c += "  args.src_tensor.GetAddress(addr_3, xc1, yc1, 0);\n";
      c += "  int dz = args.src_tensor.SliceStride();\n";
    }
  }
  auto read_src = [&](int x, int y) {
    if (src_desc.IsLinear()) {
      const std::string id = std::to_string(y * 2 + x);
      const std::string addr = "addr_" + std::to_string(y * 2 + x);
      if (src_desc.ReturnsZeroForNegOneRead()) {
        return "args.src_tensor.Read(" + addr + "); " + addr + " += dz_" + id +
               ";";
      } else {
        return "args.src_tensor.Read(" + addr + ") * INIT_FLT(in_x" +
               std::to_string(x) + " && in_y" + std::to_string(y) + "); " +
               addr + " += dz;";
      }
    } else {
      std::string check = generate_check(x, y);
      if (!check.empty()) {
        check = " * INIT_FLT(" + check + ")";
      }
      return "args.src_tensor.Read(X + " + std::to_string(x - 1) + " * " +
             pixel_stride + ", Y + " + std::to_string(y - 1) + ", s)" + check +
             ";";
    }
  };
  c += "  for (int s = 0; s < args.src_tensor.Slices(); ++s) {\n";
  if (need_local_mem) {
    c += "    " + barrier + ";\n";
  }
  if (weights_upload_type ==
      ConvolutionTransposed4x4::WeightsUploadType::LOCAL_MEM_ASYNC) {
    c += "    async_work_group_copy(weights_cache, "
         "args.weights.GetPtr(f_offset), 64, "
         "0);\n";
  } else if (weights_upload_type ==
             ConvolutionTransposed4x4::WeightsUploadType::
                 LOCAL_MEM_BY_THREADS) {
    c += "    weights_cache[local_id] = args.weights.Read(f_offset + "
         "local_id);\n";
    c += "    weights_cache[local_id + 32] = args.weights.Read(f_offset + "
         "local_id + "
         "32);\n";
  } else {  // GLOBAL_MEM
    c += "    " + weights_space +
         " FLT4* weights_cache = args.weights.GetPtr(f_offset);\n";
  }
  c += "    FLT4 src0 = " + read_src(0, 0) + ";\n";
  c += "    FLT4 src1 = " + read_src(1, 0) + ";\n";
  c += "    FLT4 src2 = " + read_src(0, 1) + ";\n";
  c += "    FLT4 src3 = " + read_src(1, 1) + ";\n";
  c += "    f_offset += 64;\n";
  if (need_local_mem) {
    c += "    " + barrier + ";\n";
  }
  c += "    CONV(r0, src0, 0);\n";
  c += "    CONV(r1, src0, 4);\n";
  c += "    CONV(r2, src0, 8);\n";
  c += "    CONV(r3, src0, 12);\n";
  c += "    CONV(r0, src1, 16);\n";
  c += "    CONV(r1, src1, 20);\n";
  c += "    CONV(r2, src1, 24);\n";
  c += "    CONV(r3, src1, 28);\n";
  c += "    CONV(r0, src2, 32);\n";
  c += "    CONV(r1, src2, 36);\n";
  c += "    CONV(r2, src2, 40);\n";
  c += "    CONV(r3, src2, 44);\n";
  c += "    CONV(r0, src3, 48);\n";
  c += "    CONV(r1, src3, 52);\n";
  c += "    CONV(r2, src3, 56);\n";
  c += "    CONV(r3, src3, 60);\n";
  c += "  }\n";
  c += "\n";
  if (need_local_mem) {
    if (op_def.IsBatchSupported()) {
      c += "  if (X0 * 2 * args.dst_tensor.Batch() > args.dst_tensor.Width() "
           "|| Y * 2 > args.dst_tensor.Height() || Z "
           ">= args.dst_tensor.Slices()) return;\n";
    } else {
      c += "  if (X * 2 > args.dst_tensor.Width() || Y * 2 > "
           "args.dst_tensor.Height() || Z >= args.dst_tensor.Slices()) "
           "return;\n";
    }
  }
  if (op_def.IsBatchSupported()) {
    c += "  X = X0 * 2 * args.dst_tensor.Batch() + B - "
         "args.dst_tensor.Batch();\n";
  } else {
    c += "  X = X * 2 - 1;\n";
  }
  c += "  Y = Y * 2 - 1;\n";
  c += "\n";
  c += "  FLT4 bias_val = args.biases.Read(Z);\n";
  c += "  if (X >= 0 && Y >= 0) {\n";
  c += "    FLT4 result = TO_FLT4(r0) + bias_val;\n";
  c += "    args.dst_tensor.Write(result, X, Y, Z);\n";
  c += "  }\n";
  c +=
      "  if (X + " + pixel_stride + " < args.dst_tensor.Width() && Y >= 0) {\n";
  c += "    FLT4 result = TO_FLT4(r1) + bias_val;\n";
  c += "    args.dst_tensor.Write(result, X + " + pixel_stride + ", Y, Z);\n";
  c += "  }\n";
  c += "  if (X >= 0 && Y + 1 < args.dst_tensor.Height()) {\n";
  c += "    FLT4 result = TO_FLT4(r2) + bias_val;\n";
  c += "    args.dst_tensor.Write(result, X, Y + 1, Z);\n";
  c += "  }\n";
  c += "  if (X + " + pixel_stride +
       " < args.dst_tensor.Width() && Y + 1 < args.dst_tensor.Height()) {\n";
  c += "    FLT4 result = TO_FLT4(r3) + bias_val;\n";
  c += "    args.dst_tensor.Write(result, X + " + pixel_stride + ", Y+1, Z);\n";
  c += "  }\n";
  c += "}\n";
  return c;
}

absl::Status ConvolutionTransposed4x4::BindArguments(ArgumentsBinder* args) {
  return args->SetInt("filter_offset", 4 * 16 * src_[0]->Slices());
}

int3 ConvolutionTransposed4x4::GetGridSize() const {
  const int grid_x = DivideRoundUp(dst_[0]->Width() + 2, 2) * dst_[0]->Batch();
  const int grid_y = DivideRoundUp(dst_[0]->Height() + 2, 2);
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int> ConvolutionTransposed4x4::GetSpatialWeightsRemap() const {
  return std::vector<int>{10, 11, 14, 15, 8, 9, 12, 13, 2, 3, 6, 7, 0, 1, 4, 5};
}

void ConvolutionTransposed4x4::UploadWeights(
    const tflite::gpu::Tensor<OHWI, DataType::FLOAT32>& weights,
    WeightsUploadType weights_upload_type) {
  const int flt_count =
      GetTotalElementsCountForLayout(GetWeightsDescription(), weights.shape);

  DataType weights_type = definition_.precision == CalculationsPrecision::F32
                              ? DataType::FLOAT32
                              : DataType::FLOAT16;

  BufferDescriptor desc;
  desc.element_type = weights_type;
  desc.element_size = 4;
  desc.memory_type =
      weights_upload_type ==
              ConvolutionTransposed4x4::WeightsUploadType::CONSTANT_MEM
          ? MemoryType::CONSTANT
          : MemoryType::GLOBAL;
  desc.size = flt_count * SizeOf(desc.element_type);
  desc.data.resize(desc.size);

  RearrangeWeights(weights, GetWeightsDescription(), weights_type,
                   absl::MakeSpan(desc.data));
  args_.AddObject("weights",
                  absl::make_unique<BufferDescriptor>(std::move(desc)));
}

bool IsConvolutionTransposed4x4Supported(
    const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr) {
  return attr.weights.shape.w == 4 && attr.weights.shape.h == 4 &&
         attr.stride.w == 2 && attr.stride.h == 2 &&
         attr.padding.prepended.w == 1 && attr.padding.prepended.h == 1;
}

ConvolutionTransposed4x4 CreateConvolutionTransposed4x4(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr) {
  ConvolutionTransposed4x4 result(definition, gpu_info);
  result.UploadWeights(attr.weights, GetBestWeightsUploadType(gpu_info));

  TensorLinearDescriptor desc;
  desc.storage_type = gpu_info.IsApple() || !gpu_info.SupportsImages()
                          ? LinearStorageType::BUFFER
                          : LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();
  desc.UploadLinearData(attr.bias);
  result.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return result;
}

ConvolutionTransposed4x4 CreateConvolutionTransposed4x4DynamicWeights(
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

  ConvolutionTransposed4x4 result(new_def, gpu_info);

  TensorLinearDescriptor desc;
  desc.storage_type = gpu_info.IsApple() || !gpu_info.SupportsImages()
                          ? LinearStorageType::BUFFER
                          : LinearStorageType::TEXTURE_2D;
  desc.element_type = new_def.GetDataType();
  desc.UploadLinearData(attr.bias);
  result.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return result;
}

}  // namespace gpu
}  // namespace tflite
