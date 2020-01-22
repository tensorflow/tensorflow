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

#include "tensorflow/lite/delegates/gpu/cl/kernels/convolution_transposed_4x4.h"

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
    const std::vector<ElementwiseOperation*>& linked_operations,
    ConvolutionTransposed4x4::WeightsUploadType weights_upload_type) {
  std::string c = GetCommonDefines(op_def.precision);

  TensorCodeGenerator src_tensor(
      "src_data", WHSPoint{"src_size.x", "src_size.y", "src_size.z"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data", WHSPoint{"dst_size.x", "dst_size.y", "dst_size.z"},
      op_def.dst_tensors[0]);

  const auto src_tensor_type = op_def.src_tensors[0].storage_type;
  const bool manual_clamp = src_tensor_type == TensorStorageType::BUFFER ||
                            src_tensor_type == TensorStorageType::IMAGE_BUFFER;

  const bool need_local_mem =
      weights_upload_type ==
          ConvolutionTransposed4x4::WeightsUploadType::LOCAL_MEM_BY_THREADS ||
      weights_upload_type ==
          ConvolutionTransposed4x4::WeightsUploadType::LOCAL_MEM_ASYNC;

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

  const std::string pixel_stride =
      op_def.IsBatchSupported() ? "dst_size.w" : "1";
  if (need_local_mem) {  // we use fixed workgroup size when use local mem
    c += "__attribute__((reqd_work_group_size(8, 4, 1)))\n";
  }
  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += "    __global FLT4* filters,\n";
  c += "    __read_only image2d_t biases";
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size,             \n";
  c += "    int filter_offset          \n";
  c += ") {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int X0 = linear_id / dst_size.w;\n";
    c += "  int B = linear_id % dst_size.w;\n";
  }
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  if (!need_local_mem) {
    if (op_def.IsBatchSupported()) {
      c += "  if (X0 * 2 * dst_size.w > dst_size.x || Y * 2 > dst_size.y || Z "
           ">= "
           "dst_size.z) return;\n";
    } else {
      c += "  if (X * 2 > dst_size.x || Y * 2 > dst_size.y || Z >= dst_size.z) "
           "return;\n";
    }
  }
  c += "  ACCUM_FLT4 r0 = (ACCUM_FLT4)(0.0f);\n";
  c += "  ACCUM_FLT4 r1 = (ACCUM_FLT4)(0.0f);\n";
  c += "  ACCUM_FLT4 r2 = (ACCUM_FLT4)(0.0f);\n";
  c += "  ACCUM_FLT4 r3 = (ACCUM_FLT4)(0.0f);\n";
  c += "  int f_offset = Z * filter_offset;\n";
  if (need_local_mem) {
    c += "  __local FLT4 weights_cache[64];\n";
  }
  if (weights_upload_type ==
      ConvolutionTransposed4x4::WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    c += "  int local_id = (int)(get_local_id(1) * 8 + get_local_id(0));\n";
  }
  if (manual_clamp) {
    const std::string prev_x = "X - " + pixel_stride;
    c += "  bool in_x0 = " + prev_x + " >= 0 && " + prev_x + " < src_size.x;\n";
    c += "  bool in_x1 = X >= 0 && X < src_size.x;\n";
    c += "  bool in_y0 = Y - 1 >= 0 && Y - 1 < src_size.y;\n";
    c += "  bool in_y1 = Y >= 0 && Y < src_size.y;\n";
    if (src_tensor_type == TensorStorageType::IMAGE_BUFFER) {
      c += "  int addr_0 = select(-1, (Y - 1) * src_size.x + " + prev_x +
           ", (in_x0 && in_y0));\n";
      c += "  int addr_1 = select(-1, (Y - 1) * src_size.x + X, (in_x1 && "
           "in_y0));\n";
      c += "  int addr_2 = select(-1, Y * src_size.x + " + prev_x +
           ", (in_x0 && in_y1));\n";
      c += "  int addr_3 = select(-1, Y * src_size.x + X, (in_x1 && "
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
    if (src_tensor_type == TensorStorageType::BUFFER) {
      c += "  int xc0 = clamp(" + prev_x + ", 0, src_size.x - 1);\n";
      c += "  int xc1 = clamp(X, 0, src_size.x - 1);\n";
      c += "  int yc0 = clamp(Y - 1, 0, src_size.y - 1);\n";
      c += "  int yc1 = clamp(Y, 0, src_size.y - 1);\n";
      c += "  int addr_0 = yc0 * src_size.x + xc0;\n";
      c += "  int addr_1 = yc0 * src_size.x + xc1;\n";
      c += "  int addr_2 = yc1 * src_size.x + xc0;\n";
      c += "  int addr_3 = yc1 * src_size.x + xc1;\n";
      c += "  int dz = src_size.x * src_size.y;\n";
    }
  }
  auto read_src = [&](int x, int y) {
    if (manual_clamp) {
      const std::string id = std::to_string(y * 2 + x);
      const std::string addr = "addr_" + std::to_string(y * 2 + x);
      if (src_tensor_type == TensorStorageType::IMAGE_BUFFER) {
        return src_tensor.Read(addr) + "; " + addr + " += dz_" + id + ";";
      } else {
        return src_tensor.Read(addr) + " * (FLT)(in_x" + std::to_string(x) +
               " && in_y" + std::to_string(y) + "); " + addr + " += dz;";
      }
    } else {
      return src_tensor.ReadWHS(
          "X + " + std::to_string(x - 1) + "*" + pixel_stride,
          "Y + " + std::to_string(y - 1), "s", TextureAddressMode::ZERO);
    }
  };
  c += "  for (int s = 0; s < src_size.z; ++s) {\n";
  if (need_local_mem) {
    c += "    barrier(CLK_LOCAL_MEM_FENCE);\n";
  }
  if (weights_upload_type ==
      ConvolutionTransposed4x4::WeightsUploadType::LOCAL_MEM_ASYNC) {
    c += "    async_work_group_copy(weights_cache, filters + f_offset, 64, "
         "0);\n";
  } else if (weights_upload_type ==
             ConvolutionTransposed4x4::WeightsUploadType::
                 LOCAL_MEM_BY_THREADS) {
    c += "    weights_cache[local_id] = filters[f_offset + local_id];\n";
    c += "    weights_cache[local_id + 32] = filters[f_offset + local_id + "
         "32];\n";
  } else {  // GLOBAL_MEM
    c += "    __global FLT4* weights_cache = filters + f_offset;\n";
  }
  c += "    FLT4 src0 = " + read_src(0, 0) + ";\n";
  c += "    FLT4 src1 = " + read_src(1, 0) + ";\n";
  c += "    FLT4 src2 = " + read_src(0, 1) + ";\n";
  c += "    FLT4 src3 = " + read_src(1, 1) + ";\n";
  c += "    f_offset += 64;\n";
  if (need_local_mem) {
    c += "    barrier(CLK_LOCAL_MEM_FENCE);\n";
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
      c += "  if (X0 * 2 * dst_size.w > dst_size.x || Y * 2 > dst_size.y || Z "
           ">= "
           "dst_size.z) return;\n";
    } else {
      c += "  if (X * 2 > dst_size.x || Y * 2 > dst_size.y || Z >= dst_size.z) "
           "return;\n";
    }
  }
  if (op_def.IsBatchSupported()) {
    c += "  X = X0 * 2 * dst_size.w + B - dst_size.w;\n";
  } else {
    c += "  X = X * 2 - 1;\n";
  }
  c += "  Y = Y * 2 - 1;\n";
  c += "\n";
  c += "  FLT4 bias_val = READ_IMAGE(biases, smp_none, (int2)(Z, 0));\n";
  c += "  if (X >= 0 && Y >= 0) {\n";
  c += "    FLT4 result = TO_FLT4(r0) + bias_val;\n";
  LinkingContext context{"result", "X", "Y", "Z"};
  c += PostProcess(linked_operations, context);
  c += "  " + dst_tensor.WriteWHS("result", "X", "Y", "Z") + "\n";
  c += "  }\n";
  c += "  if (X + " + pixel_stride + " < dst_size.x && Y >= 0) {\n";
  c += "    FLT4 result = TO_FLT4(r1) + bias_val;\n";
  context = {"result", "X + " + pixel_stride, "Y", "Z"};
  c += PostProcess(linked_operations, context);
  c += "  " + dst_tensor.WriteWHS("result", "X + " + pixel_stride, "Y", "Z") +
       "\n";
  c += "  }\n";
  c += "  if (X >= 0 && Y + 1 < dst_size.y) {\n";
  c += "    FLT4 result = TO_FLT4(r2) + bias_val;\n";
  context = {"result", "X", "Y + 1", "Z"};
  c += PostProcess(linked_operations, context);
  c += "  " + dst_tensor.WriteWHS("result", "X", "Y + 1", "Z") + "\n";
  c += "  }\n";
  c += "  if (X + " + pixel_stride + " < dst_size.x && Y + 1 < dst_size.y) {\n";
  c += "    FLT4 result = TO_FLT4(r3) + bias_val;\n";
  context = {"result", "X + " + pixel_stride, "Y + 1", "Z"};
  c += PostProcess(linked_operations, context);
  c += "  " +
       dst_tensor.WriteWHS("result", "X + " + pixel_stride, "Y + 1", "Z") +
       "\n";
  c += "  }\n";
  c += "}\n";
  return c;
}

}  // namespace

ConvolutionTransposed4x4::ConvolutionTransposed4x4(
    const OperationDef& definition, const CLDevice& device)
    : GPUOperation(definition) {
  if (device.IsPowerVR()) {
    weights_upload_type_ = WeightsUploadType::LOCAL_MEM_ASYNC;
  } else if (device.IsNvidia()) {
    weights_upload_type_ = WeightsUploadType::LOCAL_MEM_BY_THREADS;
  } else {
    weights_upload_type_ = WeightsUploadType::GLOBAL_MEM;
  }
}

ConvolutionTransposed4x4::ConvolutionTransposed4x4(
    ConvolutionTransposed4x4&& operation)
    : GPUOperation(std::move(operation)),
      weights_(std::move(operation.weights_)),
      weights_upload_type_(operation.weights_upload_type_),
      biases_(std::move(operation.biases_)),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

ConvolutionTransposed4x4& ConvolutionTransposed4x4::operator=(
    ConvolutionTransposed4x4&& operation) {
  if (this != &operation) {
    weights_ = std::move(operation.weights_);
    std::swap(weights_upload_type_, operation.weights_upload_type_);
    biases_ = std::move(operation.biases_);
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status ConvolutionTransposed4x4::Compile(
    const CreationContext& creation_context) {
  const auto code = GenerateConvolutionTransposedCode(
      definition_, linked_operations_, weights_upload_type_);

  std::vector<CompilerOptions> options;
  if (definition_.precision == CalculationsPrecision::F16 &&
      creation_context.device->IsPowerVR()) {
    options.push_back(CompilerOptions::POWERVR_FP16);
  }
  RETURN_IF_ERROR(creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", options, *creation_context.context,
      *creation_context.device, &kernel_));

  return OkStatus();
}

Status ConvolutionTransposed4x4::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_.GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWBatchedHSB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWBatchedHSB()));
  const int32_t filters_offset = 4 * 16 * src_[0]->Slices();
  RETURN_IF_ERROR(kernel_.SetBytesAuto(filters_offset));

  return OkStatus();
}

int3 ConvolutionTransposed4x4::GetGridSize() const {
  const int grid_x =
      IntegralDivideRoundUp(dst_[0]->Width() + 2, 2) * dst_[0]->Batch();
  const int grid_y = IntegralDivideRoundUp(dst_[0]->Height() + 2, 2);
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

Status ConvolutionTransposed4x4::Tune(const TuningParameters& params) {
  if (weights_upload_type_ == WeightsUploadType::LOCAL_MEM_ASYNC ||
      weights_upload_type_ == WeightsUploadType::LOCAL_MEM_BY_THREADS) {
    return OkStatus();
  }
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroupConv(params, kernel_, GetGridSize(),
                              &work_group_size_);
}

Status ConvolutionTransposed4x4::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

bool IsConvolutionTransposed4x4Supported(
    const CLDevice& device, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr) {
  return attr.weights.shape.w == 4 && attr.weights.shape.h == 4 &&
         attr.stride.w == 2 && attr.stride.h == 2 &&
         attr.padding.prepended.w == 1 && attr.padding.prepended.h == 1;
}

Status CreateConvolutionTransposed4x4(
    const CreationContext& creation_context, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr,
    ConvolutionTransposed4x4* result) {
  if (!IsConvolutionTransposed4x4Supported(*creation_context.device, definition,
                                           attr)) {
    return InvalidArgumentError(
        "ConvolutionTransposed4x4 doesn't support this attributes");
  }
  *result = ConvolutionTransposed4x4(definition, *creation_context.device);
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
