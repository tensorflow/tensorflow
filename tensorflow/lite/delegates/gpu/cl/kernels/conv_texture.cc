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

#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_texture.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GenerateConvCode(
    const OperationDef& op_def, bool is1x1, bool adreno4xx_optimization,
    const CLDevice& device,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  std::string c = GetCommonDefines(op_def.precision);
  TensorCodeGenerator src_tensor("src_data", "src_size", op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor("dst_data", "dst_size", op_def.dst_tensors[0]);

  const bool is_image_buffer =
      op_def.src_tensors[0].storage_type == TensorStorageType::IMAGE_BUFFER;

  switch (op_def.precision) {
    case CalculationsPrecision::F32:
    case CalculationsPrecision::F16:
      c += "#define CONV1(R, S)    \\\n";
      c += "R += S.x * f0; \\\n";
      c += "R += S.y * f1; \\\n";
      c += "R += S.z * f2; \\\n";
      c += "R += S.w * f3;   \n";
      c += "#define CONV2(R, S)    \\\n";
      c += "R += S.x * f4; \\\n";
      c += "R += S.y * f5; \\\n";
      c += "R += S.z * f6; \\\n";
      c += "R += S.w * f7;   \n";
      break;
    case CalculationsPrecision::F32_F16:
      c += "#define CONV1(R, S) \\\n";
      c += "R += convert_float4(S.x * f0 + S.y * f1 + S.z * f2 + S.w * f3);\n";
      c += "#define CONV2(R, S) \\\n";
      c += "R += convert_float4(S.x * f4 + S.y * f5 + S.z * f6 + S.w * f7);\n";
      break;
  }

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += "    __read_only image2d_t filters0,   \n";
  c += "    __read_only image2d_t filters1,   \n";
  c += "    __read_only image2d_t filters2,   \n";
  c += "    __read_only image2d_t filters3,   \n";
  c += "    __read_only image2d_t biases";
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,                   \n";
  c += "    int4 dst_size,                   \n";
  if (!is1x1) {
    c += "    int2 kernel_size,              \n";
    c += "    int2 dilation,                 \n";
  }
  c += "    int2 stride,                     \n";
  c += "    int2 padding                     \n";
  c += ") {\n";
  c += "  int X = get_global_id(0) * 2;\n";
  c += "  int Y = get_global_id(1) * 2;\n";
  c += "  int Z = get_global_id(2) * 2;\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.w) return;\n";
  c += "  int xc0 = X * stride.x + padding.x;\n";
  c += "  int xc1 = (X + 1) * stride.x + padding.x;\n";
  c += "  int yc0 = Y * stride.y + padding.y;\n";
  c += "  int yc1 = (Y + 1) * stride.y + padding.y;\n";
  for (int i = 0; i < 8; ++i) {
    c += "  ACCUM_FLT4 r" + std::to_string(i) +
         " = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
  }
  std::string f_y = is1x1 ? "s" : "filter_offset";
  std::string s_x0 = is1x1 ? "xc0" : "c0.x";
  std::string s_x1 = is1x1 ? "xc1" : "c1.x";
  std::string s_y0 = is1x1 ? "yc0" : "c0.y";
  std::string s_y1 = is1x1 ? "yc1" : "c1.y";
  if (!is1x1) {
    c += "  int2 c0;\n";
    c += "  int2 c1;\n";
    c += "  int filter_offset = 0;\n";
    c += "  for (int y = 0; y < kernel_size.y; ++y) {\n";
    c += "  c0.y = y * dilation.y + yc0;\n";
    c += "  c1.y = y * dilation.y + yc1;\n";
    if (is_image_buffer) {
      c += "  bool in_y0 = c0.y >= 0 && c0.y < src_size.y;\n";
      c += "  bool in_y1 = c1.y >= 0 && c1.y < src_size.y;\n";
    }
    c += "  for (int x = 0; x < kernel_size.x; ++x) {\n";
    c += "  c0.x = x * dilation.x + xc0;\n";
    c += "  c1.x = x * dilation.x + xc1;\n";
    if (is_image_buffer) {
      c += "  bool in_x0 = c0.x >= 0 && c0.x < src_size.x;\n";
      c += "  bool in_x1 = c1.x >= 0 && c1.x < src_size.x;\n";
      c += "  int addr_0 = select(-1, c0.y * src_size.x + c0.x, (in_x0 && "
           "in_y0));\n";
      c += "  int addr_1 = select(-1, c0.y * src_size.x + c1.x, (in_x1 && "
           "in_y0));\n";
      c += "  int addr_2 = select(-1, c1.y * src_size.x + c0.x, (in_x0 && "
           "in_y1));\n";
      c += "  int addr_3 = select(-1, c1.y * src_size.x + c1.x, (in_x1 && "
           "in_y1));\n";
      c += "  int dz_0 = select(0, src_size.x * src_size.y, (in_x0 && "
           "in_y0));\n";
      c += "  int dz_1 = select(0, src_size.x * src_size.y, (in_x1 && "
           "in_y0));\n";
      c += "  int dz_2 = select(0, src_size.x * src_size.y, (in_x0 && "
           "in_y1));\n";
      c += "  int dz_3 = select(0, src_size.x * src_size.y, (in_x1 && "
           "in_y1));\n";
    }
  } else if (is_image_buffer) {
    c += "  bool in_x0 = xc0 >= 0 && xc0 < src_size.x;\n";
    c += "  bool in_x1 = xc1 >= 0 && xc1 < src_size.x;\n";
    c += "  bool in_y0 = yc0 >= 0 && yc0 < src_size.y;\n";
    c += "  bool in_y1 = yc1 >= 0 && yc1 < src_size.y;\n";
    c += "  int addr_0 = select(-1, yc0 * src_size.x + xc0, (in_x0 && "
         "in_y0));\n";
    c += "  int addr_1 = select(-1, yc0 * src_size.x + xc1, (in_x1 && "
         "in_y0));\n";
    c += "  int addr_2 = select(-1, yc1 * src_size.x + xc0, (in_x0 && "
         "in_y1));\n";
    c += "  int addr_3 = select(-1, yc1 * src_size.x + xc1, (in_x1 && "
         "in_y1));\n";
    c += "  int dz_0 = select(0, src_size.x * src_size.y, (in_x0 && in_y0));\n";
    c += "  int dz_1 = select(0, src_size.x * src_size.y, (in_x1 && in_y0));\n";
    c += "  int dz_2 = select(0, src_size.x * src_size.y, (in_x0 && in_y1));\n";
    c += "  int dz_3 = select(0, src_size.x * src_size.y, (in_x1 && in_y1));\n";
  }
  c += "  for (int s = 0; s < src_size.w; ++s) {\n";
  if (is_image_buffer) {
    c += "    FLT4 src0 = " + src_tensor.Read("addr_0") + ";\n";
    c += "    FLT4 src1 = " + src_tensor.Read("addr_1") + ";\n";
    c += "    FLT4 src2 = " + src_tensor.Read("addr_2") + ";\n";
    c += "    FLT4 src3 = " + src_tensor.Read("addr_3") + ";\n";
  }
  std::string fc0 = "(int2)(Z, " + f_y + ")";
  std::string fc1 = "(int2)(Z + 1, " + f_y + ")";
  c += "    FLT4 f0 = READ_IMAGE(filters0, smp_none, " + fc0 + ");\n";
  c += "    FLT4 f1 = READ_IMAGE(filters1, smp_none, " + fc0 + ");\n";
  c += "    FLT4 f2 = READ_IMAGE(filters2, smp_none, " + fc0 + ");\n";
  c += "    FLT4 f3 = READ_IMAGE(filters3, smp_none, " + fc0 + ");\n";
  c += "    FLT4 f4 = READ_IMAGE(filters0, smp_none, " + fc1 + ");\n";
  c += "    FLT4 f5 = READ_IMAGE(filters1, smp_none, " + fc1 + ");\n";
  c += "    FLT4 f6 = READ_IMAGE(filters2, smp_none, " + fc1 + ");\n";
  c += "    FLT4 f7 = READ_IMAGE(filters3, smp_none, " + fc1 + ");\n";
  if (!is_image_buffer) {
    const auto mode = GetFastestZeroMode(device);
    c += "    FLT4 src0 = " + src_tensor.Read3D(s_x0, s_y0, "s", mode) + ";\n";
    c += "    FLT4 src1 = " + src_tensor.Read3D(s_x1, s_y0, "s", mode) + ";\n";
    c += "    FLT4 src2 = " + src_tensor.Read3D(s_x0, s_y1, "s", mode) + ";\n";
    c += "    FLT4 src3 = " + src_tensor.Read3D(s_x1, s_y1, "s", mode) + ";\n";
  }
  for (int i = 0; i < 4; ++i) {
    c += "    CONV1(r" + std::to_string(i) + ", src" + std::to_string(i) +
         ");\n";
  }
  for (int i = 0; i < 4; ++i) {
    c += "    CONV2(r" + std::to_string(i + 4) + ", src" + std::to_string(i) +
         ");\n";
  }
  if (!is1x1) {
    c += "    filter_offset++;\n";
  }
  if (is_image_buffer) {
    c += "     addr_0 += dz_0;\n";
    c += "     addr_1 += dz_1;\n";
    c += "     addr_2 += dz_2;\n";
    c += "     addr_3 += dz_3;\n";
  }
  c += "  }\n";  // src_size.w
  if (!is1x1) {
    c += "  }\n";  // kernel_size.x
    c += "  }\n";  // kernel_size.y
  }
  // when is1x1 && adreno4xx_optimization is true, xc0 == X and yc0 == Y
  std::string dst_x = is1x1 && adreno4xx_optimization ? "xc0" : "X";
  std::string dst_y = is1x1 && adreno4xx_optimization ? "yc0" : "Y";
  c += "  if (Z < dst_size.w) {\n";
  c += "    FLT4 bias_val = READ_IMAGE(biases, smp_none, (int2)(Z, 0));\n";
  for (int i = 0; i < 4; ++i) {
    c += "  {\n";
    c += "  int xc = " + dst_x + " + " + std::to_string(i % 2) + ";\n";
    c += "  int yc = " + dst_y + " + " + std::to_string(i / 2) + ";\n";
    c += "  if (xc < dst_size.x && yc < dst_size.y) {\n";
    c += "    FLT4 res = TO_FLT4(r" + std::to_string(i) + ") + bias_val;\n";
    const LinkingContext context{"res", "xc", "yc", "Z"};
    c += PostProcess(linked_operations, context);
    c += "  " + dst_tensor.Write3D("res", "xc", "yc", "Z") + "\n";
    c += "  }\n";
    c += "  }\n";
  }
  c += "  }\n";
  c += "  Z++;\n";
  c += "  if (Z < dst_size.w) {\n";
  c += "    FLT4 bias_val = READ_IMAGE(biases, smp_none, (int2)(Z, 0));\n";
  for (int i = 0; i < 4; ++i) {
    c += "  {\n";
    c += "  int xc = " + dst_x + " + " + std::to_string(i % 2) + ";\n";
    c += "  int yc = " + dst_y + " + " + std::to_string(i / 2) + ";\n";
    c += "  if (xc < dst_size.x && yc < dst_size.y) {\n";
    c += "    FLT4 res = TO_FLT4(r" + std::to_string(i + 4) + ") + bias_val;\n";
    const LinkingContext context{"res", "xc", "yc", "Z"};
    c += PostProcess(linked_operations, context);
    c += "  " + dst_tensor.Write3D("res", "xc", "yc", "Z") + "\n";
    c += "  }\n";
    c += "  }\n";
  }
  c += "  }\n";
  c += "}\n";
  return c;
}

bool UseFP16SIMD(const CLDevice& device, CalculationsPrecision precision,
                 bool kernel1x1) {
  if (!device.IsAdreno()) {
    return false;
  }
  switch (precision) {
    case CalculationsPrecision::F32:
    case CalculationsPrecision::F32_F16:
      return false;
    case CalculationsPrecision::F16:
      return device.IsAdreno3xx() && kernel1x1;
  }
}
}  // namespace

ConvTexture::ConvTexture(const OperationDef& definition,
                         const Convolution2DAttributes& attr)
    : GPUOperation(definition),
      kernel_size_(attr.weights.shape.w, attr.weights.shape.h),
      stride_(attr.strides.w, attr.strides.h),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h),
      dilation_(attr.dilations.w, attr.dilations.h),
      work_group_size_(4, 4, 2) {}

ConvTexture::ConvTexture(ConvTexture&& operation)
    : GPUOperation(std::move(operation)),
      weights_0_(std::move(operation.weights_0_)),
      weights_1_(std::move(operation.weights_1_)),
      weights_2_(std::move(operation.weights_2_)),
      weights_3_(std::move(operation.weights_3_)),
      biases_(std::move(operation.biases_)),
      kernel_size_(operation.kernel_size_),
      stride_(operation.stride_),
      padding_(operation.padding_),
      dilation_(operation.dilation_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

ConvTexture& ConvTexture::operator=(ConvTexture&& operation) {
  if (this != &operation) {
    weights_0_ = std::move(operation.weights_0_);
    weights_1_ = std::move(operation.weights_1_);
    weights_2_ = std::move(operation.weights_2_);
    weights_3_ = std::move(operation.weights_3_);
    biases_ = std::move(operation.biases_);
    std::swap(kernel_size_, operation.kernel_size_);
    std::swap(stride_, operation.stride_);
    std::swap(padding_, operation.padding_);
    std::swap(dilation_, operation.dilation_);
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status ConvTexture::Compile(const CreationContext& creation_context) {
  auto storage_type = definition_.GetPrimaryStorageType();
  bool is1x1 = kernel_size_.x == 1 && kernel_size_.y == 1;
  bool adreno4xx_optimization =
      stride_.x == 1 && stride_.y == 1 && padding_.x == 0 && padding_.y == 0 &&
      creation_context.device->IsAdreno4xx() &&
      storage_type == TensorStorageType::TEXTURE_ARRAY &&
      definition_.precision == CalculationsPrecision::F16;
  std::string code =
      GenerateConvCode(definition_, is1x1, adreno4xx_optimization,
                       *creation_context.device, linked_operations_);
  std::vector<CompilerOptions> options;
  if (UseFP16SIMD(*creation_context.device, definition_.precision, is1x1)) {
    options.push_back(CompilerOptions::ADRENO_FULL_SIMD_LINE);
  }
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", options, *creation_context.context,
      *creation_context.device, &kernel_);
}

Status ConvTexture::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_0_.GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_1_.GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_2_.GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_3_.GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));
  if (!(kernel_size_.x == 1 && kernel_size_.y == 1)) {
    RETURN_IF_ERROR(kernel_.SetBytesAuto(kernel_size_));
    RETURN_IF_ERROR(kernel_.SetBytesAuto(dilation_));
  }
  RETURN_IF_ERROR(kernel_.SetBytesAuto(stride_));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(padding_));
  return OkStatus();
}

int3 ConvTexture::GetGridSize() const {
  const int grid_x = IntegralDivideRoundUp(dst_[0]->Width(), 2);
  const int grid_y = IntegralDivideRoundUp(dst_[0]->Height(), 2);
  const int grid_z = IntegralDivideRoundUp(dst_[0]->Depth(), 2);
  return int3(grid_x, grid_y, grid_z);
}

Status ConvTexture::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroupConv(params, kernel_, GetGridSize(),
                              &work_group_size_);
}

Status ConvTexture::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Status CreateConvTexture(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const Convolution2DAttributes& attr,
                         ConvTexture* result) {
  *result = ConvTexture(definition, attr);
  RETURN_IF_ERROR(
      result->UploadWeights(attr.weights, creation_context.context));
  LinearStorageCreateInfo create_info;
  create_info.storage_type = LinearStorageType::TEXTURE_2D;
  create_info.data_type = definition.GetDataType();
  create_info.aligned_size = attr.weights.shape.o;
  RETURN_IF_ERROR(CreateLinearStorage(
      create_info, attr.bias, creation_context.context, &result->biases_));

  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
