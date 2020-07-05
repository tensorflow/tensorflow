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

#include "tensorflow/lite/delegates/gpu/cl/kernels/convolution_transposed_3x3.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GenerateConvolutionTransposedCode(
    const OperationDef& op_def,
    ConvolutionTransposed3x3::WeightsUploadType weights_upload_type,
    int2 padding, int3 work_group_launch_order, Arguments* args) {
  auto src_desc = absl::make_unique<TensorDescriptor>(op_def.src_tensors[0]);
  src_desc->SetTextureAddressMode(TextureAddressMode::ZERO);
  if (op_def.IsBatchSupported()) {
    src_desc->SetStateVar("BatchedWidth", "true");
  }
  args->AddObjectRef("src_tensor", AccessType::READ, std::move(src_desc));
  auto dst_desc = absl::make_unique<TensorDescriptor>(op_def.dst_tensors[0]);
  if (op_def.IsBatchSupported()) {
    dst_desc->SetStateVar("BatchedWidth", "true");
  }
  args->AddObjectRef("dst_tensor", AccessType::WRITE, std::move(dst_desc));
  args->AddInt("filter_offset");
  args->AddInt("padding_x");
  args->AddInt("padding_y");

  const auto src_tensor_type = op_def.src_tensors[0].storage_type;
  const bool manual_clamp = src_tensor_type == TensorStorageType::BUFFER ||
                            src_tensor_type == TensorStorageType::IMAGE_BUFFER;

  const bool need_local_mem =
      weights_upload_type ==
          ConvolutionTransposed3x3::WeightsUploadType::LOCAL_MEM_BY_THREADS ||
      weights_upload_type ==
          ConvolutionTransposed3x3::WeightsUploadType::LOCAL_MEM_ASYNC;

  std::string c = GetCommonDefines(op_def.precision);
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
      c += "  R += convert_float4(SRC.x * weights_cache[F] + SRC.y * "
           "weights_cache[F + 1] + SRC.z * weights_cache[F + 2] + SRC.w * "
           "weights_cache[F + 3]);\n";
      break;
  }

  const std::string weights_space =
      weights_upload_type ==
              ConvolutionTransposed3x3::WeightsUploadType::CONSTANT_MEM
          ? "__constant"
          : "__global";

  const std::string pixel_stride =
      op_def.IsBatchSupported() ? "args.dst_tensor.Batch()" : "1";
  c += "__attribute__((reqd_work_group_size(8, 4, 1)))\n";
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  int3 launch_remap;
  launch_remap[work_group_launch_order.x] = 0;
  launch_remap[work_group_launch_order.y] = 1;
  launch_remap[work_group_launch_order.z] = 2;
  auto GetGlobalID = [&](int id) {
    std::string result;
    const std::string sid = std::to_string(id);
    if (work_group_launch_order[id] == id) {
      return "get_global_id(" + sid + ")";
    } else {
      return "get_group_id(" + std::to_string(launch_remap[id]) +
             ") * get_local_size(" + sid + ") + get_local_id(" + sid + ")";
    }
  };
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = " + GetGlobalID(0) + ";\n";
    c += "  int X0 = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  int DST_X = X0 * 2 * args.dst_tensor.Batch() + B;\n";
    c += "  int SRC_X = linear_id + args.padding_x;\n";
  } else {
    c += "  int X = " + GetGlobalID(0) + ";\n";
    c += "  int DST_X = X * 2;\n";
    c += "  int SRC_X = X + args.padding_x;\n";
  }
  c += "  int Y = " + GetGlobalID(1) + ";\n";
  c += "  int DST_Y = Y * 2;\n";
  c += "  int SRC_Y = Y + args.padding_y;\n";
  c += "  int Z = " + GetGlobalID(2) + ";\n";
  if (!need_local_mem) {
    c += "  if (DST_X >= args.dst_tensor.Width() || DST_Y >= "
         "args.dst_tensor.Height() || Z >= args.dst_tensor.Slices()) return;\n";
  }
  c += "  ACCUM_FLT4 r0 = (ACCUM_FLT4)(0.0f);\n";
  c += "  ACCUM_FLT4 r1 = (ACCUM_FLT4)(0.0f);\n";
  c += "  ACCUM_FLT4 r2 = (ACCUM_FLT4)(0.0f);\n";
  c += "  ACCUM_FLT4 r3 = (ACCUM_FLT4)(0.0f);\n";
  c += "  int f_offset = Z * args.filter_offset;\n";
  if (need_local_mem) {
    c += "  __local FLT4 weights_cache[36];\n";
  }
  if (weights_upload_type ==
      ConvolutionTransposed3x3::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    c += "  int local_id = (int)(get_local_id(1) * 8 + get_local_id(0));\n";
  }
  if (manual_clamp) {
    const std::string next_x = "SRC_X + " + pixel_stride;
    c += "  bool in_x0 = SRC_X >= 0 && SRC_X < args.src_tensor.Width();\n";
    c += "  bool in_x1 = " + next_x + " >= 0 && " + next_x +
         " < args.src_tensor.Width();\n";
    c += "  bool in_y0 = SRC_Y >= 0 && SRC_Y < args.src_tensor.Height();\n";
    c += "  bool in_y1 = SRC_Y + 1 >= 0 && SRC_Y + 1 < "
         "args.src_tensor.Height();\n";
    if (src_tensor_type == TensorStorageType::BUFFER) {
      c += "  int xc0 = clamp(SRC_X, 0, args.src_tensor.Width() - 1);\n";
      c += "  int xc1 = clamp(" + next_x +
           ", 0, args.src_tensor.Width() - 1);\n";
      c += "  int yc0 = clamp(SRC_Y, 0, args.src_tensor.Height() - 1);\n";
      c += "  int yc1 = clamp(SRC_Y + 1, 0, args.src_tensor.Height() - 1);\n";
      c += "  args.src_tensor.GetAddress(addr_0, xc0, yc0, 0);\n";
      c += "  args.src_tensor.GetAddress(addr_1, xc1, yc0, 0);\n";
      c += "  args.src_tensor.GetAddress(addr_2, xc0, yc1, 0);\n";
      c += "  args.src_tensor.GetAddress(addr_3, xc1, yc1, 0);\n";
      c += "  int dz = args.src_tensor.SliceStride();\n";
    } else {  // TensorStorageType::IMAGE_BUFFER
      c += "  args.src_tensor.GetAddress(addr_0, SRC_X, SRC_Y, 0);\n";
      c += "  args.src_tensor.GetAddress(addr_1," + next_x + ", SRC_Y, 0);\n";
      c += "  args.src_tensor.GetAddress(addr_2, SRC_X, SRC_Y + 1, 0);\n";
      c += "  args.src_tensor.GetAddress(addr_3," + next_x + ", SRC_Y+1, 0);\n";
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
    }
  }
  auto read_src = [&](int x, int y) {
    if (manual_clamp) {
      const std::string id = std::to_string(y * 2 + x);
      const std::string addr = "addr_" + std::to_string(y * 2 + x);
      if (src_tensor_type == TensorStorageType::IMAGE_BUFFER) {
        return "args.src_tensor.Read(" + addr + "); " + addr + " += dz_" + id +
               ";\n";
      } else {
        return "args.src_tensor.Read(" + addr + ") * (FLT)(in_x" +
               std::to_string(x) + " && in_y" + std::to_string(y) + "); " +
               addr + " += dz;\n";
      }
    } else {
      return "args.src_tensor.Read(SRC_X + " + std::to_string(x) + "*" +
             pixel_stride + ", SRC_Y + " + std::to_string(y) + ", s);\n";
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
    c += "    barrier(CLK_LOCAL_MEM_FENCE);\n";
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
    c += "    barrier(CLK_LOCAL_MEM_FENCE);\n";
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
      const std::string x_c = "DST_X + " + s_x + " * " + pixel_stride;
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

}  // namespace

ConvolutionTransposed3x3::ConvolutionTransposed3x3(
    const OperationDef& definition, const CLDevice& device, int2 padding)
    : GPUOperation(definition),
      padding_(padding),
      work_group_launch_order_(2, 0, 1) {
  if (device.IsPowerVR()) {
    weights_upload_type_ = WeightsUploadType::LOCAL_MEM_ASYNC;
  } else if (device.IsNvidia() || device.IsIntel()) {
    weights_upload_type_ = WeightsUploadType::LOCAL_MEM_BY_THREADS;
  } else if (device.IsAMD()) {
    weights_upload_type_ = WeightsUploadType::CONSTANT_MEM;
  } else {
    weights_upload_type_ = WeightsUploadType::GLOBAL_MEM;
  }
}

ConvolutionTransposed3x3::ConvolutionTransposed3x3(
    ConvolutionTransposed3x3&& operation)
    : GPUOperation(std::move(operation)),
      padding_(operation.padding_),
      work_group_launch_order_(operation.work_group_launch_order_),
      weights_upload_type_(operation.weights_upload_type_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

ConvolutionTransposed3x3& ConvolutionTransposed3x3::operator=(
    ConvolutionTransposed3x3&& operation) {
  if (this != &operation) {
    std::swap(padding_, operation.padding_);
    std::swap(work_group_launch_order_, operation.work_group_launch_order_);
    std::swap(weights_upload_type_, operation.weights_upload_type_);
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status ConvolutionTransposed3x3::Compile(
    const CreationContext& creation_context) {
  std::string code = GenerateConvolutionTransposedCode(
      definition_, weights_upload_type_, padding_, work_group_launch_order_,
      &args_);
  std::string element_wise_code;
  RETURN_IF_ERROR(
      MergeOperations(linked_operations_, &args_, &element_wise_code));
  RETURN_IF_ERROR(args_.TransformToCLCode(creation_context.device->GetInfo(),
                                          {{"dst_tensor", element_wise_code}},
                                          &code));

  std::vector<CompilerOptions> options;
  if (definition_.precision == CalculationsPrecision::F16 &&
      creation_context.device->IsPowerVR()) {
    options.push_back(CompilerOptions::POWERVR_FP16);
  }
  RETURN_IF_ERROR(creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", options, *creation_context.context,
      *creation_context.device, &kernel_));
  return absl::OkStatus();
}

absl::Status ConvolutionTransposed3x3::BindArguments() {
  RETURN_IF_ERROR(args_.SetObjectRef("src_tensor", src_[0]));
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  RETURN_IF_ERROR(args_.SetInt("filter_offset", 4 * 9 * src_[0]->Slices()));
  const int padding_x =
      padding_.x >= 1 ? (padding_.x - 1) / 2 : (padding_.x - 2) / 2;
  const int padding_y =
      padding_.y >= 1 ? (padding_.y - 1) / 2 : (padding_.y - 2) / 2;
  RETURN_IF_ERROR(args_.SetInt("padding_x", padding_x * src_[0]->Batch()));
  RETURN_IF_ERROR(args_.SetInt("padding_y", padding_y));
  RETURN_IF_ERROR(SetArguments(linked_operations_, &args_));
  return args_.Bind(kernel_.kernel());
}

int3 ConvolutionTransposed3x3::GetGridSize() const {
  const int grid_x = DivideRoundUp(dst_[0]->Width(), 2) * dst_[0]->Batch();
  const int grid_y = DivideRoundUp(dst_[0]->Height(), 2);
  const int grid_z = dst_[0]->Slices();
  int3 wg;
  wg.x = DivideRoundUp(grid_x, work_group_size_.x);
  wg.y = DivideRoundUp(grid_y, work_group_size_.y);
  wg.z = DivideRoundUp(grid_z, work_group_size_.z);
  return int3(wg[work_group_launch_order_[0]] * work_group_size_.x,
              wg[work_group_launch_order_[1]] * work_group_size_.y,
              wg[work_group_launch_order_[2]] * work_group_size_.z);
  return int3(grid_x, grid_y, grid_z);
}

absl::Status ConvolutionTransposed3x3::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

bool IsConvolutionTransposed3x3Supported(
    const CLDevice& device, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr) {
  return attr.weights.shape.w == 3 && attr.weights.shape.h == 3 &&
         attr.stride.w == 2 && attr.stride.h == 2;
}

absl::Status CreateConvolutionTransposed3x3(
    const CreationContext& creation_context, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr,
    ConvolutionTransposed3x3* result) {
  if (!IsConvolutionTransposed3x3Supported(*creation_context.device, definition,
                                           attr)) {
    return absl::InvalidArgumentError(
        "ConvolutionTransposed3x3 doesn't support this attributes");
  }
  const int2 padding = int2(attr.padding.prepended.w, attr.padding.prepended.h);
  *result =
      ConvolutionTransposed3x3(definition, *creation_context.device, padding);
  RETURN_IF_ERROR(
      result->UploadWeights(attr.weights, creation_context.context));

  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::TEXTURE_2D;
  desc.element_type = definition.GetDataType();

  LinearStorage lt;
  RETURN_IF_ERROR(
      CreateLinearStorage(desc, attr.bias, creation_context.context, &lt));
  result->args_.AddObject("biases", AccessType::READ,
                          absl::make_unique<LinearStorage>(std::move(lt)),
                          absl::make_unique<TensorLinearDescriptor>(desc));
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
