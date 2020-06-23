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

#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_weights_converter.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetConverterToConvWeightsCode(
    const OperationDef& op_def, const ConvWeightsDescription& conv_weights_desc,
    Arguments* args) {
  args->AddObjectRef(
      "src_tensor", AccessType::READ,
      absl::make_unique<TensorDescriptor>(op_def.src_tensors[0]));
  args->AddObjectRef(
      "dst_tensor", AccessType::WRITE,
      absl::make_unique<TensorDescriptor>(op_def.dst_tensors[0]));
  args->AddFloat("mask_x");
  args->AddFloat("mask_y");
  args->AddFloat("mask_z");
  args->AddFloat("mask_w");

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  int GROUP_SIZE = " +
       std::to_string(conv_weights_desc.output_group_size) + ";\n";
  c += "  int O = get_global_id(0) * 4;\n";
  c += "  int I = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  int W = Z % args.src_tensor.Width();\n";
  c += "  int H = Z / args.src_tensor.Width();\n";
  c += "  if (O >= args.src_tensor.Batch() || I >= args.src_tensor.Slices() || "
       "H >= args.src_tensor.Height()) return;\n";
  c += "  FLT4 v0 = args.src_tensor.Read(W, H, I, O + 0);\n";
  c += "  FLT4 v1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
  c += "  FLT4 v2 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
  c += "  FLT4 v3 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
  c += "  if (O + 1 < args.src_tensor.Batch()) {\n";
  c += "    v1 = args.src_tensor.Read(W, H, I, O + 1);\n";
  c += "  }\n";
  c += "  if (O + 2 < args.src_tensor.Batch()) {\n";
  c += "    v2 = args.src_tensor.Read(W, H, I, O + 2);\n";
  c += "  }\n";
  c += "  if (O + 3 < args.src_tensor.Batch()) {\n";
  c += "    v3 = args.src_tensor.Read(W, H, I, O + 3);\n";
  c += "  }\n";
  c += "  if (I == args.src_tensor.Slices() - 1) {\n";
  c += "    FLT4 mask = (FLT4)(args.mask_x, args.mask_y, args.mask_z, "
       "args.mask_w);\n";
  c += "    v0 *= mask;\n";
  c += "    v1 *= mask;\n";
  c += "    v2 *= mask;\n";
  c += "    v3 *= mask;\n";
  c += "  }\n";
  c += "  FLT4 r0 = (FLT4)(v0.x, v1.x, v2.x, v3.x);\n";
  c += "  FLT4 r1 = (FLT4)(v0.y, v1.y, v2.y, v3.y);\n";
  c += "  FLT4 r2 = (FLT4)(v0.z, v1.z, v2.z, v3.z);\n";
  c += "  FLT4 r3 = (FLT4)(v0.w, v1.w, v2.w, v3.w);\n";
  c += "  int d_index = O / (GROUP_SIZE * 4);\n";
  c += "  int k_index = (O % (GROUP_SIZE * 4)) / 4;\n";
  c += "  int dst_offset = (((d_index * args.src_tensor.Height() + H) * "
       "args.src_tensor.Width() + W) * "
       "args.src_tensor.Slices() + I) * GROUP_SIZE + "
       "k_index;\n";
  c += "  int address0 = dst_offset * 4 + 0;\n";
  c += "  int address1 = dst_offset * 4 + 1;\n";
  c += "  int address2 = dst_offset * 4 + 2;\n";
  c += "  int address3 = dst_offset * 4 + 3;\n";
  c += "  args.dst_tensor.WriteLinear(r0, dst_offset * 4 + 0)\n;";
  c += "  args.dst_tensor.WriteLinear(r1, dst_offset * 4 + 1)\n;";
  c += "  args.dst_tensor.WriteLinear(r2, dst_offset * 4 + 2)\n;";
  c += "  args.dst_tensor.WriteLinear(r3, dst_offset * 4 + 3)\n;";
  c += "}\n";
  return c;
}
}  // namespace

ConverterToConvWeights::ConverterToConvWeights(
    ConverterToConvWeights&& operation)
    : GPUOperation(std::move(operation)),
      conv_weights_desc_(operation.conv_weights_desc_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

ConverterToConvWeights& ConverterToConvWeights::operator=(
    ConverterToConvWeights&& operation) {
  if (this != &operation) {
    conv_weights_desc_ = operation.conv_weights_desc_;
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status ConverterToConvWeights::Compile(
    const CreationContext& creation_context) {
  std::string code =
      GetConverterToConvWeightsCode(definition_, conv_weights_desc_, &args_);
  RETURN_IF_ERROR(
      args_.TransformToCLCode(creation_context.device->GetInfo(), {}, &code));
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status ConverterToConvWeights::BindArguments() {
  RETURN_IF_ERROR(args_.SetObjectRef("src_tensor", src_[0]));
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  float4 mask = GetMaskForLastPlane(src_[0]->Channels());
  RETURN_IF_ERROR(args_.SetFloat("mask_x", mask.x));
  RETURN_IF_ERROR(args_.SetFloat("mask_y", mask.y));
  RETURN_IF_ERROR(args_.SetFloat("mask_z", mask.z));
  RETURN_IF_ERROR(args_.SetFloat("mask_w", mask.w));
  RETURN_IF_ERROR(SetArguments(linked_operations_, &args_));
  return args_.Bind(kernel_.kernel());
}

int3 ConverterToConvWeights::GetGridSize() const {
  const int grid_x = DivideRoundUp(
      AlignByN(src_[0]->Batch(), 4 * conv_weights_desc_.output_group_size), 4);
  const int grid_y = src_[0]->Slices();
  const int grid_z = src_[0]->Width() * src_[0]->Height();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status ConverterToConvWeights::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status ConverterToConvWeights::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

ConverterToConvWeights CreateConverterToConvWeights(
    const OperationDef& definition,
    const ConvWeightsDescription& conv_weights_desc) {
  return ConverterToConvWeights(definition, conv_weights_desc);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
