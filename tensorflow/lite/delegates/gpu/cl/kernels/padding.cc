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

#include "tensorflow/lite/delegates/gpu/cl/kernels/padding.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetPaddingCode(
    const OperationDef& op_def,
    const std::vector<ElementwiseOperation*>& linked_operations,
    const PadAttributes& attr) {
  TensorCodeGenerator src_tensor(
      "src_data",
      WHSBPoint{"src_size.x", "src_size.y", "src_size.z", "src_size.w"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data",
      WHSBPoint{"dst_size.x", "dst_size.y", "dst_size.z", "dst_size.w"},
      op_def.dst_tensors[0]);

  const std::string dst_batch = op_def.IsBatchSupported() ? "B" : "";
  std::string c = GetCommonDefines(op_def.precision);
  const std::string channels[] = {".x", ".y", ".z", ".w"};

  if (attr.type == PaddingContentType::REFLECT) {
    c += "int reflect(int x, int size) {\n";
    c += "  int t = abs(x) - size + 1;\n";
    c += "  return size - 1 - abs(t);\n";
    c += "}\n\n";
  }

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,      \n";
  c += "    int src_channels,   \n";
  c += "    int4 dst_size,      \n";
  c += "    int4 prepended      \n";
  c += ") {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int X = linear_id / dst_size.w;\n";
    c += "  int B = linear_id % dst_size.w;\n";
  } else {
    c += "  int X = get_global_id(0);\n";
  }
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;\n";
  c += "  FLT4 result = (FLT4)(0.0);\n";
  c += "  int s_x = X - prepended.x;\n";
  c += "  int s_y = Y - prepended.y;\n";
  if (op_def.IsBatchSupported()) {
    c += "  int s_b = B - prepended.w;\n";
  }
  const std::string src_batch = op_def.IsBatchSupported() ? "s_b" : "";
  if (attr.type == PaddingContentType::REFLECT) {
    c += "  s_x = reflect(s_x, src_size.x);\n";
    c += "  s_y = reflect(s_y, src_size.y);\n";
    if (op_def.IsBatchSupported()) {
      c += "  int s_b = reflect(s_b, src_size.w);\n";
    }
    if (attr.prepended.c == 0 && attr.appended.c == 0) {
      // optimized case
      c += "  result = " + src_tensor.ReadWHSB("s_x", "s_y", "Z", src_batch) +
           ";\n";
    } else {
      c += "  int start_channel = Z * 4;\n";
      for (int i = 0; i < 4; ++i) {
        const auto& s = channels[i];
        c += "  {\n";
        c += "    int channel = start_channel + " + std::to_string(i) + ";\n";
        c += "    int s_z = channel - prepended.z;\n";
        // We need additional clamp for z, so that we use alignment for channels
        // and can proceed extra channels that can lead to reading out of
        // resource.
        c += "    s_z = clamp(reflect(s_z, src_channels), 0, src_channels - "
             "1);\n";
        c += "    FLT4 t = " +
             src_tensor.ReadWHSB("s_x", "s_y", "s_z / 4", src_batch) + ";\n";
        c += "    FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
        c += "    result" + s + " = t_ar[s_z % 4];\n";
        c += "  }\n";
      }
    }
  } else {
    c += "  bool inside_x = s_x >= 0 && s_x < src_size.x;\n";
    c += "  bool inside_y = s_y >= 0 && s_y < src_size.y;\n";
    if (op_def.IsBatchSupported()) {
      c += "  inside_y &= (s_b >= 0 && s_b < src_size.w);\n";
    }
    c += "  if (inside_x && inside_y) {\n";
    if (attr.prepended.c == 0 && attr.appended.c == 0) {
      // optimized case
      c += "    result = " + src_tensor.ReadWHSB("s_x", "s_y", "Z", src_batch) +
           ";\n";
    } else if (attr.prepended.c % 4 == 0) {
      c += "    int s_z = Z - prepended.z / 4;\n";
      c += "    if (s_z >= 0 && s_z < src_size.z) {\n";
      c += "      result = " +
           src_tensor.ReadWHSB("s_x", "s_y", "s_z", src_batch) + ";\n";
      c += "    }\n";
    } else {
      c += "    int start_channel = Z * 4;\n";
      for (int i = 0; i < 4; ++i) {
        const auto& s = channels[i];
        c += "    {\n";
        c += "    int channel = start_channel + " + std::to_string(i) + ";\n";
        c += "    int s_z = channel - prepended.z;\n";
        c += "    if (s_z >= 0 && s_z < src_channels) {\n";
        c += "      FLT4 t = " +
             src_tensor.ReadWHSB("s_x", "s_y", "s_z / 4", src_batch) + ";\n";
        c += "      FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
        c += "      result" + s + " = t_ar[s_z % 4];\n";
        c += "    }\n";
        c += "    }\n";
      }
    }
    c += "  }\n";
  }
  std::string x_3dcoord =
      op_def.IsBatchSupported() ? "X * dst_size.w + B" : "X";
  c += PostProcess(linked_operations, {"result", x_3dcoord, "Y", "Z"});
  c += "  " + dst_tensor.WriteWHSB("result", "X", "Y", "Z", dst_batch);
  c += "}\n";

  return c;
}
}  // namespace

Padding::Padding(const OperationDef& definition, const PadAttributes& attr)
    : GPUOperation(definition), attributes_(attr) {}

Padding::Padding(Padding&& kernel)
    : GPUOperation(std::move(kernel)),
      attributes_(kernel.attributes_),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

Padding& Padding::operator=(Padding&& kernel) {
  if (this != &kernel) {
    std::swap(attributes_, kernel.attributes_);
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

absl::Status Padding::Compile(const CreationContext& creation_context) {
  const auto code =
      GetPaddingCode(definition_, linked_operations_, attributes_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status Padding::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWHSB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->Channels()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWHSB()));
  const auto& prep = attributes_.prepended;
  RETURN_IF_ERROR(kernel_.SetBytesAuto(int4(prep.w, prep.h, prep.c, prep.b)));
  return absl::OkStatus();
}

int3 Padding::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status Padding::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status Padding::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Padding CreatePadding(const OperationDef& definition,
                      const PadAttributes& attr) {
  return Padding(definition, attr);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
