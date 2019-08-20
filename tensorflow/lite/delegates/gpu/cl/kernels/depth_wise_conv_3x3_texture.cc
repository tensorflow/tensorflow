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

#include "tensorflow/lite/delegates/gpu/cl/kernels/depth_wise_conv_3x3_texture.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GenerateDepthWiseConvCode(
    const TensorDescriptor& src_descriptor,
    const TensorDescriptor& dst_descriptor, CalculationsPrecision precision,
    const std::vector<ElementwiseOperation*>& linked_operations,
    const CLDevice& device) {
  std::string c = GetCommonDefines(precision);
  TensorCodeGenerator src_tensor("src_data", "dst_size", src_descriptor);
  TensorCodeGenerator dst_tensor("dst_data", "dst_size", dst_descriptor);

  const auto mode = device.IsAdreno3xx() ? TextureAddressMode::DONT_CARE
                                         : TextureAddressMode::ZERO;

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += "    __read_only image2d_t filters\n";
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 dst_size\n";
  c += ") {\n";
  c += "  int X = get_global_id(0) * 2;\n";
  c += "  int Y = get_global_id(1) * 2;\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.w) return;\n";
  c += "   ACCUM_FLT4 r0 = (ACCUM_FLT4)(0.0f);\n";
  c += "   ACCUM_FLT4 r1 = (ACCUM_FLT4)(0.0f);\n";
  c += "   ACCUM_FLT4 r2 = (ACCUM_FLT4)(0.0f);\n";
  c += "   ACCUM_FLT4 r3 = (ACCUM_FLT4)(0.0f);\n";
  c += "   FLT4 f0 = READ_IMAGE(filters, smp_none, (int2)(0, Z));\n";
  c += "   FLT4 f1 = READ_IMAGE(filters, smp_none, (int2)(1, Z));\n";
  c += "   FLT4 f2 = READ_IMAGE(filters, smp_none, (int2)(2, Z));\n";
  c += "   FLT4 f3 = READ_IMAGE(filters, smp_none, (int2)(3, Z));\n";
  c += "   FLT4 f4 = READ_IMAGE(filters, smp_none, (int2)(4, Z));\n";
  c += "   FLT4 f5 = READ_IMAGE(filters, smp_none, (int2)(5, Z));\n";
  c += "   FLT4 f6 = READ_IMAGE(filters, smp_none, (int2)(6, Z));\n";
  c += "   FLT4 f7 = READ_IMAGE(filters, smp_none, (int2)(7, Z));\n";
  c += "   FLT4 f8 = READ_IMAGE(filters, smp_none, (int2)(8, Z));\n";
  c += " \n";
  c += "   FLT4 s0;\n";
  c += "   FLT4 s1;\n";
  c += "   FLT4 s2;\n";
  c += "   FLT4 s3;\n";
  c += " \n";
  if (device.IsAdreno3xx()) {
    c += "   FLT4 in_x;\n";
    c += "   FLT4 in_y;\n";
    c += "   in_x.x = (FLT)(X - 1 >= 0 && X - 1 < dst_size.x);\n";
    c += "   in_x.y = (FLT)(X >= 0 && X < dst_size.x);\n";
    c += "   in_x.z = (FLT)(X + 1 >= 0 && X + 1 < dst_size.x);\n";
    c += "   in_x.w = (FLT)(X + 2 >= 0 && X + 2 < dst_size.x);\n";
    c += "   in_y.x = (FLT)(Y - 1 >= 0 && Y - 1 < dst_size.y);\n";
    c += "   in_y.y = (FLT)(Y >= 0 && Y < dst_size.y);\n";
    c += "   in_y.z = (FLT)(Y + 1 >= 0 && Y + 1 < dst_size.y);\n";
    c += "   in_y.w = (FLT)(Y + 2 >= 0 && Y + 2 < dst_size.y);\n";
  }
  if (device.IsAdreno3xx()) {
    c += " if (Z > -4) {\n";
    c += "   s0 = " + src_tensor.Read3D("X - 1", "Y - 1", "Z", mode) +
         " * in_x.x * in_y.x;\n";
    c += "   s1 = " + src_tensor.Read3D("X", "Y - 1", "Z", mode) +
         " * in_x.y * in_y.x;\n";
    c += "   s2 = " + src_tensor.Read3D("X + 1", "Y - 1", "Z", mode) +
         " * in_x.z * in_y.x;\n";
    c += "   s3 = " + src_tensor.Read3D("X + 2", "Y - 1", "Z", mode) +
         " * in_x.w * in_y.x;\n";
  } else {
    c += " {\n";
    c += "   s0 = " + src_tensor.Read3D("X - 1", "Y - 1", "Z", mode) + ";\n";
    c += "   s1 = " + src_tensor.Read3D("X", "Y - 1", "Z", mode) + ";\n";
    c += "   s2 = " + src_tensor.Read3D("X + 1", "Y - 1", "Z", mode) + ";\n";
    c += "   s3 = " + src_tensor.Read3D("X + 2", "Y - 1", "Z", mode) + ";\n";
  }
  c += "   r0 += TO_ACCUM_TYPE(f0 * s0);\n";
  c += "   r0 += TO_ACCUM_TYPE(f1 * s1);\n";
  c += "   r1 += TO_ACCUM_TYPE(f0 * s1);\n";
  c += "   r0 += TO_ACCUM_TYPE(f2 * s2);\n";
  c += "   r1 += TO_ACCUM_TYPE(f1 * s2);\n";
  c += "   r1 += TO_ACCUM_TYPE(f2 * s3);\n";
  c += " }\n";
  if (device.IsAdreno3xx()) {
    c += " if (Z > -3) {\n";
    c += "   s0 = " + src_tensor.Read3D("X - 1", "Y", "Z", mode) +
         " * in_x.x * in_y.y;\n";
    c += "   s1 = " + src_tensor.Read3D("X", "Y", "Z", mode) +
         " * in_x.y * in_y.y;\n";
    c += "   s2 = " + src_tensor.Read3D("X + 1", "Y", "Z", mode) +
         " * in_x.z * in_y.y;\n";
    c += "   s3 = " + src_tensor.Read3D("X + 2", "Y", "Z", mode) +
         " * in_x.w * in_y.y;\n";
  } else {
    c += " {\n";
    c += "   s0 = " + src_tensor.Read3D("X - 1", "Y", "Z", mode) + ";\n";
    c += "   s1 = " + src_tensor.Read3D("X", "Y", "Z", mode) + ";\n";
    c += "   s2 = " + src_tensor.Read3D("X + 1", "Y", "Z", mode) + ";\n";
    c += "   s3 = " + src_tensor.Read3D("X + 2", "Y", "Z", mode) + ";\n";
  }
  c += "   r0 += TO_ACCUM_TYPE(f3 * s0);\n";
  c += "   r2 += TO_ACCUM_TYPE(f0 * s0);\n";
  c += "   r0 += TO_ACCUM_TYPE(f4 * s1);\n";
  c += "   r1 += TO_ACCUM_TYPE(f3 * s1);\n";
  c += "   r2 += TO_ACCUM_TYPE(f1 * s1);\n";
  c += "   r3 += TO_ACCUM_TYPE(f0 * s1);\n";
  c += "   r0 += TO_ACCUM_TYPE(f5 * s2);\n";
  c += "   r1 += TO_ACCUM_TYPE(f4 * s2);\n";
  c += "   r2 += TO_ACCUM_TYPE(f2 * s2);\n";
  c += "   r3 += TO_ACCUM_TYPE(f1 * s2);\n";
  c += "   r1 += TO_ACCUM_TYPE(f5 * s3);\n";
  c += "   r3 += TO_ACCUM_TYPE(f2 * s3);\n";
  c += " }\n";
  if (device.IsAdreno3xx()) {
    c += " if (Z > -2) {\n";
    c += "   s0 = " + src_tensor.Read3D("X - 1", "Y + 1", "Z", mode) +
         " * in_x.x * in_y.z;\n";
    c += "   s1 = " + src_tensor.Read3D("X", "Y + 1", "Z", mode) +
         " * in_x.y * in_y.z;\n";
    c += "   s2 = " + src_tensor.Read3D("X + 1", "Y + 1", "Z", mode) +
         " * in_x.z * in_y.z;\n";
    c += "   s3 = " + src_tensor.Read3D("X + 2", "Y + 1", "Z", mode) +
         " * in_x.w * in_y.z;\n";
  } else {
    c += " {\n";
    c += "   s0 = " + src_tensor.Read3D("X - 1", "Y + 1", "Z", mode) + ";\n";
    c += "   s1 = " + src_tensor.Read3D("X", "Y + 1", "Z", mode) + ";\n";
    c += "   s2 = " + src_tensor.Read3D("X + 1", "Y + 1", "Z", mode) + ";\n";
    c += "   s3 = " + src_tensor.Read3D("X + 2", "Y + 1", "Z", mode) + ";\n";
  }
  c += "   r0 += TO_ACCUM_TYPE(f6 * s0);\n";
  c += "   r2 += TO_ACCUM_TYPE(f3 * s0);\n";
  c += "   r0 += TO_ACCUM_TYPE(f7 * s1);\n";
  c += "   r1 += TO_ACCUM_TYPE(f6 * s1);\n";
  c += "   r2 += TO_ACCUM_TYPE(f4 * s1);\n";
  c += "   r3 += TO_ACCUM_TYPE(f3 * s1);\n";
  c += "   r0 += TO_ACCUM_TYPE(f8 * s2);\n";
  c += "   r1 += TO_ACCUM_TYPE(f7 * s2);\n";
  c += "   r2 += TO_ACCUM_TYPE(f5 * s2);\n";
  c += "   r3 += TO_ACCUM_TYPE(f4 * s2);\n";
  c += "   r1 += TO_ACCUM_TYPE(f8 * s3);\n";
  c += "   r3 += TO_ACCUM_TYPE(f5 * s3);\n";
  c += " }\n";
  if (device.IsAdreno3xx()) {
    c += " if (Z > -1) {\n";
    c += "   s0 = " + src_tensor.Read3D("X - 1", "Y + 2", "Z", mode) +
         " * in_x.x * in_y.w;\n";
    c += "   s1 = " + src_tensor.Read3D("X", "Y + 2", "Z", mode) +
         " * in_x.y * in_y.w;\n";
    c += "   s2 = " + src_tensor.Read3D("X + 1", "Y + 2", "Z", mode) +
         " * in_x.z * in_y.w;\n";
    c += "   s3 = " + src_tensor.Read3D("X + 2", "Y + 2", "Z", mode) +
         " * in_x.w * in_y.w;\n";
  } else {
    c += " {\n";
    c += "   s0 = " + src_tensor.Read3D("X - 1", "Y + 2", "Z", mode) + ";\n";
    c += "   s1 = " + src_tensor.Read3D("X", "Y + 2", "Z", mode) + ";\n";
    c += "   s2 = " + src_tensor.Read3D("X + 1", "Y + 2", "Z", mode) + ";\n";
    c += "   s3 = " + src_tensor.Read3D("X + 2", "Y + 2", "Z", mode) + ";\n";
  }
  c += "   r2 += TO_ACCUM_TYPE(f6 * s0);\n";
  c += "   r2 += TO_ACCUM_TYPE(f7 * s1);\n";
  c += "   r3 += TO_ACCUM_TYPE(f6 * s1);\n";
  c += "   r2 += TO_ACCUM_TYPE(f8 * s2);\n";
  c += "   r3 += TO_ACCUM_TYPE(f7 * s2);\n";
  c += "   r3 += TO_ACCUM_TYPE(f8 * s3);\n";
  c += " }\n";
  c += "   FLT4 bias = READ_IMAGE(filters, smp_none, (int2)(9, Z));\n";
  c += "   r0 += TO_ACCUM_TYPE(bias);\n";
  c += "   r1 += TO_ACCUM_TYPE(bias);\n";
  c += "   r2 += TO_ACCUM_TYPE(bias);\n";
  c += "   r3 += TO_ACCUM_TYPE(bias);\n";
  c += "   if(X + 0 < dst_size.x && Y + 0 < dst_size.y) {\n";
  c += "     FLT4 result = TO_FLT4(r0);\n";
  c += "  " + dst_tensor.GetAddress("address", "X + 0", "Y + 0", "Z") + "\n";
  c += PostProcess(linked_operations, "result", "Z", "address");
  c += "  " + dst_tensor.Write3D("result", "address") + "\n";
  c += "   }\n";
  c += "   if(X + 1 < dst_size.x && Y + 0 < dst_size.y) {\n";
  c += "     FLT4 result = TO_FLT4(r1);\n";
  c += "  " + dst_tensor.GetAddress("address", "X + 1", "Y + 0", "Z") + "\n";
  c += PostProcess(linked_operations, "result", "Z", "address");
  c += "  " + dst_tensor.Write3D("result", "address") + "\n";
  c += "   }\n";
  c += "   if(X + 0 < dst_size.x && Y + 1 < dst_size.y) {\n";
  c += "     FLT4 result = TO_FLT4(r2);\n";
  c += "  " + dst_tensor.GetAddress("address", "X + 0", "Y + 1", "Z") + "\n";
  c += PostProcess(linked_operations, "result", "Z", "address");
  c += "  " + dst_tensor.Write3D("result", "address") + "\n";
  c += "   }\n";
  c += "   if(X + 1 < dst_size.x && Y + 1 < dst_size.y) {\n";
  c += "     FLT4 result = TO_FLT4(r3);\n";
  c += "  " + dst_tensor.GetAddress("address", "X + 1", "Y + 1", "Z") + "\n";
  c += PostProcess(linked_operations, "result", "Z", "address");
  c += "  " + dst_tensor.Write3D("result", "address") + "\n";
  c += "   }\n";
  c += " }\n";

  return c;
}

}  // namespace

DepthWiseConv3x3Texture::DepthWiseConv3x3Texture(const OperationDef& definition)
    : GPUOperation(definition) {}

DepthWiseConv3x3Texture::DepthWiseConv3x3Texture(
    DepthWiseConv3x3Texture&& kernel)
    : GPUOperation(std::move(kernel)),
      weights_(std::move(kernel.weights_)),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

DepthWiseConv3x3Texture& DepthWiseConv3x3Texture::operator=(
    DepthWiseConv3x3Texture&& kernel) {
  if (this != &kernel) {
    weights_ = std::move(kernel.weights_);
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

Status DepthWiseConv3x3Texture::Compile(
    const CreationContext& creation_context) {
  std::string code = GenerateDepthWiseConvCode(
      definition_.src_tensors[0], definition_.dst_tensors[0],
      definition_.precision, linked_operations_, *creation_context.device);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status DepthWiseConv3x3Texture::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));

  return OkStatus();
}

int3 DepthWiseConv3x3Texture::GetGridSize() const {
  const int grid_x = IntegralDivideRoundUp(dst_[0]->Width(), 2);
  const int grid_y = IntegralDivideRoundUp(dst_[0]->Height(), 2);
  const int grid_z = dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

Status DepthWiseConv3x3Texture::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status DepthWiseConv3x3Texture::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

bool IsDepthWiseConv3x3TextureSupported(
    const DepthwiseConvolution2DAttributes& attr) {
  return attr.weights.shape.o == 1 && attr.dilations.w == 1 &&
         attr.dilations.h == 1 && attr.weights.shape.w == 3 &&
         attr.weights.shape.h == 3 && attr.strides.w == 1 &&
         attr.strides.h == 1 && attr.padding.prepended.w == 1 &&
         attr.padding.prepended.h == 1 && attr.padding.appended.w == 1 &&
         attr.padding.appended.h == 1;
}

Status CreateDepthWiseConv3x3Texture(
    const CreationContext& creation_context, const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr,
    DepthWiseConv3x3Texture* result) {
  if (!IsDepthWiseConv3x3TextureSupported(attr)) {
    return InvalidArgumentError(
        "DepthWiseConv3x3Texture doesn't support this attributes");
  }
  *result = DepthWiseConv3x3Texture(definition);
  RETURN_IF_ERROR(result->UploadWeightsAndBiases(attr.weights, attr.bias,
                                                 creation_context.context));
  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
