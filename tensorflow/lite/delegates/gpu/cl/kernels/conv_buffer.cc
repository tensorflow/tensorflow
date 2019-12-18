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

#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_buffer.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GenerateConvBuffer(
    const OperationDef& op_def, bool stride_correction, int x_elements,
    int y_elements,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  std::string c = GetCommonDefines(op_def.precision);
  TensorCodeGenerator src_tensor("src_data",
                                 {"src_size.x", "src_size.y", "src_size.z"},
                                 op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor("dst_data",
                                 {"dst_size.x", "dst_size.y", "dst_size.z"},
                                 op_def.dst_tensors[0]);

  switch (op_def.precision) {
    case CalculationsPrecision::F32:
    case CalculationsPrecision::F16:
      c += "#define CONV(R, S)    \\\n";
      c += "R += S.x * f0.s0123; \\\n";
      c += "R += S.y * f0.s4567; \\\n";
      c += "R += S.z * f0.s89ab; \\\n";
      c += "R += S.w * f0.scdef;   \n";
      break;
    case CalculationsPrecision::F32_F16:
      c += "#define CONV(R, S) \\\n";
      c += "R += convert_float4(S.x * f0.s0123 + S.y * f0.s4567 + S.z * "
           "f0.s89ab + S.w * f0.scdef);\n";
      break;
  }

  switch (op_def.precision) {
    case CalculationsPrecision::F32:
      c += "#define FLT16 float16\n";
      break;
    case CalculationsPrecision::F32_F16:
    case CalculationsPrecision::F16:
      c += "#define FLT16 half16\n";
      break;
  }

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += "    __global FLT16* filters_buffer,   \n";
  c += "    __global FLT4* biases             \n";
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,                   \n";
  c += "    int4 dst_size,                   \n";
  c += "    int2 kernel_size,                \n";
  c += "    int2 dilation,                   \n";
  c += "    int2 stride,                     \n";
  c += "    int2 padding                     \n";
  c += ") {\n";
  c += "  int X = get_global_id(0) * " + std::to_string(x_elements) + ";\n";
  c += "  int Y = get_global_id(1) * " + std::to_string(y_elements) + ";\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;\n";
  c += "  __global FLT16* temp = filters_buffer + Z * src_size.z * "
       "kernel_size.x * kernel_size.y;\n";
  c += "  ACCUM_FLT4 bias_val = TO_ACCUM_TYPE(biases[Z]);\n";
  for (int i = 0; i < x_elements * y_elements; ++i) {
    c += "  ACCUM_FLT4 r" + std::to_string(i) + " = bias_val;\n";
  }
  for (int x = 0; x < x_elements; ++x) {
    std::string x_s = std::to_string(x);
    if (stride_correction) {
      c += "  int xc" + x_s + " = " +
           GetXStrideCorrected("X + " + x_s, "src_size.w", "stride.x",
                               "padding.x") +
           ";\n";
    } else {
      c += "  int xc" + x_s + " = (X + " + x_s + ") * stride.x + padding.x;\n";
    }
  }
  for (int y = 0; y < y_elements; ++y) {
    std::string y_s = std::to_string(y);
    c += "  int yc" + y_s + " = (Y + " + y_s + ") * stride.y + padding.y;\n";
  }
  c += "  for (int y = 0; y < kernel_size.y; ++y) {\n";
  for (int y = 0; y < y_elements; ++y) {
    std::string y_s = std::to_string(y);
    c += "  int c" + y_s + "y = y * dilation.y + yc" + y_s + ";\n";
    c += "  bool y" + y_s + "_in = c" + y_s + "y >= 0 && c" + y_s +
         "y < src_size.y;\n";
    c += "  c" + y_s + "y = clamp(c" + y_s + "y, 0, src_size.y - 1);\n";
  }
  c += "  for (int x = 0; x < kernel_size.x; ++x) {\n";
  for (int x = 0; x < x_elements; ++x) {
    std::string x_s = std::to_string(x);
    c += "  int c" + x_s + "x = x * dilation.x + xc" + x_s + ";\n";
    c += "  bool x" + x_s + "_in = c" + x_s + "x >= 0 && c" + x_s +
         "x < src_size.x;\n";
    c += "  c" + x_s + "x = clamp(c" + x_s + "x, 0, src_size.x - 1);\n";
  }
  for (int x = 0; x < x_elements; ++x) {
    std::string x_s = std::to_string(x);
    for (int y = 0; y < y_elements; ++y) {
      std::string y_s = std::to_string(y);
      std::string i_s = std::to_string(y * x_elements + x);
      c += "  int src_addr_" + i_s + " = c" + y_s + "y * src_size.x + c" + x_s +
           "x;\n";
    }
  }
  c += "  for (int s = 0; s < src_size.z; ++s) {\n";
  for (int x = 0; x < x_elements; ++x) {
    std::string x_s = std::to_string(x);
    for (int y = 0; y < y_elements; ++y) {
      std::string y_s = std::to_string(y);
      std::string i_s = std::to_string(y * x_elements + x);
      c += "    FLT4 s" + i_s + " = src_data[src_addr_" + i_s + "] * (FLT)(y" +
           y_s + "_in && x" + x_s + "_in);\n";
    }
  }
  c += "    FLT16 f0 = temp[0];\n";
  for (int i = 0; i < x_elements * y_elements; ++i) {
    std::string i_s = std::to_string(i);
    c += "    CONV(r" + i_s + ", s" + i_s + ");\n";
  }
  for (int i = 0; i < x_elements * y_elements; ++i) {
    std::string i_s = std::to_string(i);
    c += "    src_addr_" + i_s + " += src_size.x * src_size.y;\n";
  }
  c += "    temp += 1;\n";
  c += "  }\n";  // src_size.z - SRC_DEPTH
  c += "  }\n";  // kernel_size.x
  c += "  }\n";  // kernel_size.y

  for (int x = 0; x < x_elements; ++x) {
    std::string x_s = std::to_string(x);
    for (int y = 0; y < y_elements; ++y) {
      std::string y_s = std::to_string(y);
      std::string i_s = std::to_string(y * x_elements + x);
      c += "  if (X + " + x_s + " < dst_size.x && Y + " + y_s +
           " < dst_size.y) {\n";
      c += "    FLT4 res = TO_FLT4(r" + i_s + ");\n";
      const LinkingContext context{"res", "X + " + x_s, "Y + " + y_s, "Z"};
      c += PostProcess(linked_operations, context);
      c += "  " + dst_tensor.Write3D("res", "X + " + x_s, "Y + " + y_s, "Z") +
           "\n";
      c += "  }\n";
    }
  }
  c += "}\n";
  return c;
}
}  // namespace

ConvBuffer::ConvBuffer(const OperationDef& definition,
                       const Convolution2DAttributes& attr, int x_elements,
                       int y_elements)
    : GPUOperation(definition),
      kernel_size_(attr.weights.shape.w, attr.weights.shape.h),
      stride_(attr.strides.w, attr.strides.h),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h),
      dilation_(attr.dilations.w, attr.dilations.h),
      x_elements_(x_elements),
      y_elements_(y_elements),
      work_group_size_(4, 4, 4) {}

ConvBuffer::ConvBuffer(ConvBuffer&& operation)
    : GPUOperation(std::move(operation)),
      weights_(std::move(operation.weights_)),
      biases_(std::move(operation.biases_)),
      kernel_size_(operation.kernel_size_),
      stride_(operation.stride_),
      padding_(operation.padding_),
      dilation_(operation.dilation_),
      x_elements_(operation.x_elements_),
      y_elements_(operation.y_elements_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

ConvBuffer& ConvBuffer::operator=(ConvBuffer&& operation) {
  if (this != &operation) {
    weights_ = std::move(operation.weights_);
    biases_ = std::move(operation.biases_);
    std::swap(kernel_size_, operation.kernel_size_);
    std::swap(stride_, operation.stride_);
    std::swap(padding_, operation.padding_);
    std::swap(dilation_, operation.dilation_);
    std::swap(x_elements_, operation.x_elements_);
    std::swap(y_elements_, operation.y_elements_);
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status ConvBuffer::Compile(const CreationContext& creation_context) {
  const bool stride_correction = definition_.batch_support && stride_.x != 1;
  const std::string code =
      GenerateConvBuffer(definition_, stride_correction, x_elements_,
                         y_elements_, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status ConvBuffer::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_.GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWBatchedHDB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWBatchedHDB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(kernel_size_));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(int2(dilation_.x * src_[0]->Batch(), dilation_.y)));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(stride_));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(int2(padding_.x * src_[0]->Batch(), padding_.y)));
  return OkStatus();
}

int3 ConvBuffer::GetGridSize() const {
  const int grid_x =
      IntegralDivideRoundUp(dst_[0]->Width() * dst_[0]->Batch(), x_elements_);
  const int grid_y = IntegralDivideRoundUp(dst_[0]->Height(), y_elements_);
  const int grid_z = dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

Status ConvBuffer::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroupConv(params, kernel_, GetGridSize(),
                              &work_group_size_);
}

Status ConvBuffer::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Status CreateConvBuffer(const CreationContext& creation_context,
                        const OperationDef& definition,
                        const Convolution2DAttributes& attr,
                        ConvBuffer* result) {
  int x_elements = 2;
  int y_elements = 1;
  if (definition.precision != CalculationsPrecision::F16) {
    x_elements = 1;
    y_elements = 1;
  }
  *result = ConvBuffer(definition, attr, x_elements, y_elements);
  RETURN_IF_ERROR(
      result->UploadWeights(attr.weights, creation_context.context));
  LinearStorageCreateInfo create_info;
  create_info.storage_type = LinearStorageType::BUFFER;
  create_info.data_type = definition.GetDataType();
  create_info.aligned_size = attr.weights.shape.o;
  RETURN_IF_ERROR(CreateLinearStorage(
      create_info, attr.bias, creation_context.context, &result->biases_));

  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
