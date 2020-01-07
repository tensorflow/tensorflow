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

#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_buffer_1x1.h"

#include <array>
#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

// x_elements - amount of elements processed by thread in W dimension
// y_elements - amount of elements processed by thread in H dimension
// element_size must be 1, 2 or 4
// 1 - is FLT4
// 2 - is FLT8
// 4 - is FLT16
// This function generates code for arithmetic part of convolution
std::string GetComputationPart(int x_elements, int y_elements, int element_size,
                               CalculationsPrecision precision) {
  const std::string hexes[16] = {"0", "1", "2", "3", "4", "5", "6", "7",
                                 "8", "9", "a", "b", "c", "d", "e", "f"};
  std::string c;
  for (int y = 0; y < y_elements; ++y) {
    for (int x = 0; x < x_elements; ++x) {
      std::string s_index = std::to_string(y * x_elements + x);
      for (int e = 0; e < element_size; ++e) {
        std::string r_index =
            std::to_string((y * x_elements + x) * element_size + e);
        switch (precision) {
          case CalculationsPrecision::F32:
          case CalculationsPrecision::F16:
            c += "    r" + r_index + " += f0.s0123 * s" + s_index + ".s" +
                 hexes[e * 4 + 0] + ";\n";
            c += "    r" + r_index + " += f0.s4567 * s" + s_index + ".s" +
                 hexes[e * 4 + 1] + ";\n";
            c += "    r" + r_index + " += f0.s89ab * s" + s_index + ".s" +
                 hexes[e * 4 + 2] + ";\n";
            c += "    r" + r_index + " += f0.scdef * s" + s_index + ".s" +
                 hexes[e * 4 + 3] + ";\n";
            break;
          case CalculationsPrecision::F32_F16:
            c += "    r" + r_index + " += convert_float4(f0.s0123 * s" +
                 s_index + ".s" + hexes[e * 4 + 0] + " + f0.s4567 * s" +
                 s_index + ".s" + hexes[e * 4 + 1] + " + f0.s89ab * s" +
                 s_index + ".s" + hexes[e * 4 + 2] + " + f0.scdef * s" +
                 s_index + ".s" + hexes[e * 4 + 3] + ");\n";
            break;
        }
      }
    }
  }
  return c;
}

std::string GetShiftFromElementSize(int element_size) {
  if (element_size == 4) {
    return " >> 2";
  } else if (element_size == 2) {
    return " >> 1";
  } else {
    return "";
  }
}

std::string GenerateConvBuffer1x1(
    const OperationDef& op_def, int x_elements, int y_elements,
    int element_size,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  std::string c = GetCommonDefines(op_def.precision);
  TensorCodeGenerator dst_tensor("dst_data",
                                 {"dst_size.x", "dst_size.y", "dst_size.z"},
                                 op_def.dst_tensors[0]);

  switch (op_def.precision) {
    case CalculationsPrecision::F32:
      c += "#define FLT8 float8\n";
      c += "#define FLT16 float16\n";
      break;
    case CalculationsPrecision::F32_F16:
    case CalculationsPrecision::F16:
      c += "#define FLT8 half8\n";
      c += "#define FLT16 half16\n";
      break;
  }

  c += "__kernel void main_function(\n";
  c += "    __global FLT" + std::to_string(element_size * 4) + "* src_data,\n";
  c += "    __global FLT16* filters_buffer,   \n";
  c += "    __global FLT4* biases             \n";
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,                   \n";
  c += "    int4 dst_size                    \n";
  c += ") {\n";
  c += "  int X = get_global_id(0) * " +
       std::to_string(x_elements * element_size) + ";\n";
  c += "  int Y = get_global_id(1) * " + std::to_string(y_elements) + ";\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;\n";
  c += "  __global FLT16* temp = filters_buffer + Z * src_size.z;\n";
  c += "  ACCUM_FLT4 bias_val = TO_ACCUM_TYPE(biases[Z]);\n";
  for (int i = 0; i < x_elements * element_size * y_elements; ++i) {
    c += "  ACCUM_FLT4 r" + std::to_string(i) + " = bias_val;\n";
  }
  for (int x = 0; x < x_elements; ++x) {
    std::string x_s = std::to_string(x);
    c += "  int xc" + x_s + " = min(X + " + std::to_string(x * element_size) +
         ", src_size.x - 1);\n";
  }
  for (int y = 0; y < y_elements; ++y) {
    std::string y_s = std::to_string(y);
    c += "  int yc" + y_s + " = min(Y + " + y_s + ", src_size.y - 1);\n";
  }
  std::string shift = GetShiftFromElementSize(element_size);
  for (int y = 0; y < y_elements; ++y) {
    std::string y_s = std::to_string(y);
    for (int x = 0; x < x_elements; ++x) {
      std::string x_s = std::to_string(x);
      std::string i_s = std::to_string(y * x_elements + x);
      c += "  int src_addr_" + i_s + " = ((yc" + y_s + ") * src_size.x + (xc" +
           x_s + "))" + shift + ";\n";
    }
  }
  c += "  for (int s = 0; s < src_size.z; ++s) {\n";
  for (int y = 0; y < y_elements; ++y) {
    std::string y_s = std::to_string(y);
    for (int x = 0; x < x_elements; ++x) {
      std::string x_s = std::to_string(x);
      std::string i_s = std::to_string(y * x_elements + x);
      c += "    FLT" + std::to_string(element_size * 4) + " s" + i_s +
           " = src_data[src_addr_" + i_s + "];\n";
    }
  }
  c += "    FLT16 f0 = temp[0];\n";
  c += GetComputationPart(x_elements, y_elements, element_size,
                          op_def.precision);
  for (int i = 0; i < x_elements * y_elements; ++i) {
    std::string i_s = std::to_string(i);
    c += "    src_addr_" + i_s + " += src_size.w;\n";
  }
  c += "    temp += 1;\n";
  c += "  }\n";  // src_size.z = SRC_DEPTH

  for (int y = 0; y < y_elements; ++y) {
    std::string y_s = std::to_string(y);
    for (int x = 0; x < x_elements * element_size; ++x) {
      std::string x_s = std::to_string(x);
      std::string i_s = std::to_string(y * x_elements * element_size + x);
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

int GetGridWidth(int width) {
  if (width % 2 == 0) {  // using kernel_flt8_
    return width / 2;
  } else {  // using kernel_flt4_
    return width;
  }
}

}  // namespace

ConvBuffer1x1::ConvBuffer1x1(const OperationDef& definition,
                             int flt4_x_count, int flt4_y_count,
                             int flt8_x_count, int flt8_y_count)
    : GPUOperation(definition),
      flt4_x_count_(flt4_x_count),
      flt4_y_count_(flt4_y_count),
      flt8_x_count_(flt8_x_count),
      flt8_y_count_(flt8_y_count),
      work_group_size_(2, 4, 1) {}

ConvBuffer1x1::ConvBuffer1x1(ConvBuffer1x1&& operation)
    : GPUOperation(std::move(operation)),
      weights_(std::move(operation.weights_)),
      biases_(std::move(operation.biases_)),
      kernel_flt4_(std::move(operation.kernel_flt4_)),
      flt4_x_count_(operation.flt4_x_count_),
      flt4_y_count_(operation.flt4_y_count_),
      kernel_flt8_(std::move(operation.kernel_flt8_)),
      flt8_x_count_(operation.flt8_x_count_),
      flt8_y_count_(operation.flt8_y_count_),
      work_group_size_(operation.work_group_size_) {}

ConvBuffer1x1& ConvBuffer1x1::operator=(ConvBuffer1x1&& operation) {
  if (this != &operation) {
    weights_ = std::move(operation.weights_);
    biases_ = std::move(operation.biases_);
    kernel_flt4_ = std::move(operation.kernel_flt4_);
    std::swap(flt4_x_count_, operation.flt4_x_count_);
    std::swap(flt4_y_count_, operation.flt4_y_count_);
    kernel_flt8_ = std::move(operation.kernel_flt8_);
    std::swap(flt8_x_count_, operation.flt8_x_count_);
    std::swap(flt8_y_count_, operation.flt8_y_count_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status ConvBuffer1x1::Compile(const CreationContext& creation_context) {
  std::string code_flt4 = GenerateConvBuffer1x1(
      definition_, flt4_x_count_, flt4_y_count_, 1, linked_operations_);
  RETURN_IF_ERROR(creation_context.cache->GetOrCreateCLKernel(
      code_flt4, "main_function", *creation_context.context,
      *creation_context.device, &kernel_flt4_));
  std::string code_flt8 = GenerateConvBuffer1x1(
      definition_, flt8_x_count_, flt8_y_count_, 2, linked_operations_);
  RETURN_IF_ERROR(creation_context.cache->GetOrCreateCLKernel(
      code_flt8, "main_function", *creation_context.context,
      *creation_context.device, &kernel_flt8_));
  return OkStatus();
}

CLKernel* ConvBuffer1x1::GetKernel(int width) {
  if (width % 2 == 0) {
    return &kernel_flt8_;
  } else {
    return &kernel_flt4_;
  }
}

Status ConvBuffer1x1::BindArguments() {
  CLKernel* kernel = GetKernel(src_[0]->Width());
  kernel->ResetBindingCounter();
  RETURN_IF_ERROR(kernel->SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel->SetMemoryAuto(weights_.GetMemoryPtr()));
  RETURN_IF_ERROR(kernel->SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(kernel, linked_operations_));
  RETURN_IF_ERROR(kernel->SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  int4 src_size = int4(
      src_[0]->Width() * src_[0]->Batch(), src_[0]->Height(), src_[0]->Slices(),
      GetGridWidth(src_[0]->Width()) * src_[0]->Height() * src_[0]->Batch());
  RETURN_IF_ERROR(kernel->SetBytesAuto(src_size));
  RETURN_IF_ERROR(kernel->SetBytesAuto(dst_[0]->GetWBatchedHSB()));
  return OkStatus();
}

int3 ConvBuffer1x1::GetGridSize() const {
  const int fltx_count =
      dst_[0]->Width() % 2 == 0 ? flt8_x_count_ : flt4_x_count_;
  const int flty_count =
      dst_[0]->Width() % 2 == 0 ? flt8_y_count_ : flt4_y_count_;
  const int grid_x = IntegralDivideRoundUp(
      GetGridWidth(dst_[0]->Width()) * dst_[0]->Batch(), fltx_count);
  const int grid_y = IntegralDivideRoundUp(dst_[0]->Height(), flty_count);
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

Status ConvBuffer1x1::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroupConv(params, *GetKernel(src_[0]->Width()),
                              GetGridSize(), &work_group_size_);
}

Status ConvBuffer1x1::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(*GetKernel(src_[0]->Width()), GetGridSize(),
                                 work_group_size_);
}

bool IsConvBuffer1x1Supported(const OperationDef& definition,
                              const Convolution2DAttributes& attr) {
  auto src_storage_type = definition.src_tensors[0].storage_type;
  return src_storage_type == TensorStorageType::BUFFER &&
         attr.weights.shape.w == 1 && attr.weights.shape.h == 1 &&
         attr.dilations.w == 1 && attr.dilations.w == 1 &&
         attr.strides.w == 1 && attr.strides.h == 1 &&
         attr.padding.prepended.w == 0 && attr.padding.prepended.h == 0 &&
         attr.padding.appended.w == 0 && attr.padding.appended.h == 0;
}

Status CreateConvBuffer1x1(const CreationContext& creation_context,
                           const OperationDef& definition,
                           const Convolution2DAttributes& attr,
                           ConvBuffer1x1* result) {
  if (!IsConvBuffer1x1Supported(definition, attr)) {
    return InvalidArgumentError("ConvBuffer1x1 doesn't supported");
  }
  int flt4_x_count = 1;
  int flt4_y_count = 1;
  int flt8_x_count = 1;
  int flt8_y_count = 1;
  if (creation_context.device->vendor() == Vendor::MALI) {
    if (definition.precision == CalculationsPrecision::F16 &&
        creation_context.device->GetInfo().compute_units_count <= 4) {
      flt4_x_count = 2;
      flt8_x_count = 2;
    }
  }
  *result = ConvBuffer1x1(definition, flt4_x_count, flt4_y_count, flt8_x_count,
                          flt8_y_count);
  return result->UploadData(attr.weights, attr.bias, creation_context.context);
}

Status CreateConvBuffer1x1(const CreationContext& creation_context,
                           const OperationDef& definition,
                           const FullyConnectedAttributes& attr,
                           ConvBuffer1x1* result) {
  int flt4_x_count = 1;
  int flt4_y_count = 1;
  int flt8_x_count = 1;
  int flt8_y_count = 1;
  if (creation_context.device->vendor() == Vendor::MALI) {
    if (definition.precision == CalculationsPrecision::F16 &&
        creation_context.device->GetInfo().compute_units_count <= 4) {
      flt4_x_count = 2;
      flt8_x_count = 2;
    }
  }
  *result = ConvBuffer1x1(definition, flt4_x_count, flt4_y_count, flt8_x_count,
                          flt8_y_count);
  return result->UploadData(attr.weights, attr.bias, creation_context.context);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
