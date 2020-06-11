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

#include "tensorflow/lite/delegates/gpu/cl/kernels/transpose.h"

#include <string>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetTransposeCode(
    const OperationDef& op_def, const TransposeAttributes& attr,
    const std::vector<ElementwiseOperation*>& linked_operations,
    Arguments* args) {
  args->AddObjectRef(
      "src_tensor", AccessType::READ,
      absl::make_unique<TensorDescriptor>(op_def.src_tensors[0]));
  args->AddObjectRef(
      "dst_tensor", AccessType::WRITE,
      absl::make_unique<TensorDescriptor>(op_def.dst_tensors[0]));

  const std::string batch_id = op_def.IsBatchSupported() ? "B" : "";
  std::string c = GetCommonDefines(op_def.precision);
  std::string linked_args = GetArgsDeclaration(linked_operations);
  if (linked_args[0] == ',') {
    linked_args[0] = ' ';
  }
  c += "__kernel void main_function(\n";
  c += linked_args;
  c += "$0) {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = get_global_id(0);\n";
  }
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT temps[4];\n";
  c += "  temps[0] = (FLT)(0.0f);\n";
  c += "  temps[1] = (FLT)(0.0f);\n";
  c += "  temps[2] = (FLT)(0.0f);\n";
  c += "  temps[3] = (FLT)(0.0f);\n";
  int remap[4];
  remap[attr.perm.b] = 0;
  remap[attr.perm.h] = 1;
  remap[attr.perm.w] = 2;
  remap[attr.perm.c] = 3;
  if (attr.perm.c == 3) {  // optimized reading when no channels permutation
    const std::string bhw[] = {"B", "Y", "X"};
    if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
      c += "  args.src_tensor.SetBatchRef(" + bhw[remap[0]] + ");\n";
    }
    c += "  int s_y = " + bhw[remap[1]] + ";\n";
    c += "  int s_x = " + bhw[remap[2]] + ";\n";
    c += "  FLT4 t = args.src_tensor.Read(s_x, s_y, Z);\n";
    c += "  temps[0] = t.x;\n";
    c += "  temps[1] = t.y;\n";
    c += "  temps[2] = t.z;\n";
    c += "  temps[3] = t.w;\n";
  } else {
    c += "  for (int i = 0; i < 4; ++i) {\n";
    c += "    int dst_channel = Z * 4 + i;\n";
    c += "    if (dst_channel < args.dst_tensor.Channels()) {\n";
    const std::string bhwc[] = {"B", "Y", "X", "dst_channel"};
    if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
      c += "      args.src_tensor.SetBatchRef(" + bhwc[remap[0]] + ");\n";
    }
    c += "      int s_y = " + bhwc[remap[1]] + ";\n";
    c += "      int s_x = " + bhwc[remap[2]] + ";\n";
    c += "      int s_c = " + bhwc[remap[3]] + ";\n";
    c += "      int s_z = s_c / 4;\n";
    c += "      int src_sub_ch = s_c % 4;\n";
    c += "      FLT4 t = args.src_tensor.Read(s_x, s_y, s_z);\n";
    c += "      FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
    c += "      temps[i] = t_ar[src_sub_ch];\n";
    c += "    }\n";
    c += "  }\n";
  }
  c += "  FLT4 result = (FLT4)(temps[0], temps[1], temps[2], temps[3]);\n";
  std::string x_3dcoord =
      op_def.IsBatchSupported() ? "X * args.dst_tensor.Batch() + B" : "X";
  const LinkingContext context{"result", x_3dcoord, "Y", "Z"};
  c += PostProcess(linked_operations, context);
  c += "  args.dst_tensor.Write(result, X, Y, Z);\n";
  c += "}\n";
  return c;
}
}  // namespace

Transpose::Transpose(Transpose&& operation)
    : GPUOperation(std::move(operation)),
      attr_(operation.attr_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

Transpose& Transpose::operator=(Transpose&& operation) {
  if (this != &operation) {
    attr_ = operation.attr_;
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status Transpose::Compile(const CreationContext& creation_context) {
  std::string code =
      GetTransposeCode(definition_, attr_, linked_operations_, &args_);
  RETURN_IF_ERROR(args_.TransformToCLCode({}, &code));
  code = absl::Substitute(code, args_.GetListOfArgs());
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status Transpose::BindArguments() {
  RETURN_IF_ERROR(args_.SetObjectRef("src_tensor", src_[0]));
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(args_.Bind(kernel_.kernel(), kernel_.GetBindingCounter()));
  return absl::OkStatus();
}

int3 Transpose::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status Transpose::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status Transpose::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Transpose CreateTranspose(const OperationDef& definition,
                          const TransposeAttributes& attr) {
  return Transpose(definition, attr);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
