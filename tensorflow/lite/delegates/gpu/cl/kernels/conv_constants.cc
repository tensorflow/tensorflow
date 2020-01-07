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

#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_constants.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GenerateConvolutionConstantCode(
    const OperationDef& op_def, const int2& kernel_size, int src_channels,
    int dst_channels, bool stride_correction, const CLDevice& device,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor("src_data",
                                 {"src_size.x", "src_size.y", "src_size.z"},
                                 op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor("dst_data",
                                 {"dst_size.x", "dst_size.y", "dst_size.z"},
                                 op_def.dst_tensors[0]);

  std::string c = GetCommonDefines(op_def.precision);

  const int out_z = IntegralDivideRoundUp(dst_channels, 4);
  const std::string kOutZ = std::to_string(out_z);
  const int src_depth = IntegralDivideRoundUp(src_channels, 4);

  const auto src_tensor_type = op_def.src_tensors[0].storage_type;
  const bool manual_clamp = src_tensor_type == TensorStorageType::BUFFER ||
                            src_tensor_type == TensorStorageType::IMAGE_BUFFER;

  switch (op_def.precision) {
    case CalculationsPrecision::F32:
    case CalculationsPrecision::F16:
      c += "#define CONV4(R, SRC, F, i) \\\n";
      c += "  R += SRC.x * F[i + 0]; \\\n";
      c += "  R += SRC.y * F[i + 1]; \\\n";
      c += "  R += SRC.z * F[i + 2]; \\\n";
      c += "  R += SRC.w * F[i + 3];   \n";

      c += "#define CONV3(R, SRC, F, i) \\\n";
      c += "  R += SRC.x * F[i + 0]; \\\n";
      c += "  R += SRC.y * F[i + 1]; \\\n";
      c += "  R += SRC.z * F[i + 2]; \n";

      c += "#define CONV2(R, SRC, F, i) \\\n";
      c += "  R += SRC.x * F[i + 0]; \\\n";
      c += "  R += SRC.y * F[i + 1]; \n";

      c += "#define CONV1(R, SRC, F, i) \\\n";
      c += "  R += SRC * F[i + 0]; \n";
      break;
    case CalculationsPrecision::F32_F16:
      c += "#define CONV4(R, SRC, F, i) \\\n";
      c += "  R += convert_float4(SRC.x * F[i + 0] + SRC.y * F[i + 1]";
      c += " + SRC.z * F[i + 2] + SRC.w * F[i + 3]);\n";

      c += "#define CONV3(R, SRC, F, i) \\\n";
      c += "  R += convert_float4(SRC.x * F[i + 0] + SRC.y * F[i + 1]";
      c += " + SRC.z * F[i + 2]);\n";

      c += "#define CONV2(R, SRC, F, i) \\\n";
      c += "  R += convert_float4(SRC.x * F[i + 0] + SRC.y * F[i + 1]);\n";

      c += "#define CONV1(R, SRC, F, i) \\\n";
      c += "  R += convert_float4(SRC * F[i + 0]);\n";
      break;
  }

  const std::string postfixes[] = {".x", ".xy", ".xyz", ""};

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += "    __constant FLT4* filters,  \n";
  c += "    __constant FLT4* biases";
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int2 stride,               \n";
  c += "    int2 padding,              \n";
  c += "    int2 dilation,             \n";
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size              \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y) return;\n";
  if (stride_correction) {
    c += "  int start_x = " +
         GetXStrideCorrected("X", "src_size.w", "stride.x", "padding.x") +
         ";\n";
  } else {
    c += "  int start_x = X * stride.x + padding.x;\n";
  }
  c += "  int start_y = Y * stride.y + padding.y;\n";
  c += "  ACCUM_FLT4 r[" + kOutZ + "];\n";
  c += "  for (int i = 0; i < " + kOutZ + "; ++i) {\n";
  c += "    r[i] = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
  c += "  }\n";
  const auto address_mode = GetFastestZeroMode(device);
  int filters_counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    const int ch_count = std::min(4, src_channels - s * 4);
    const std::string s_conv = "CONV" + std::to_string(ch_count);
    const std::string s_count = ch_count == 1 ? "" : std::to_string(ch_count);
    const std::string s_type = absl::StrCat("FLT", s_count);
    const std::string s_postfix = postfixes[ch_count - 1];
    for (int ky = 0; ky < kernel_size.y; ++ky) {
      std::string s_y = absl::StrCat("(start_y + ", ky, " * dilation.y)");
      if (manual_clamp) {
        c += "  {\n";
        c += "  bool y_out = " + s_y + " < 0 || " + s_y + " >= src_size.y;\n";
      }
      for (int kx = 0; kx < kernel_size.x; ++kx) {
        c += "  {\n";
        std::string s_x = absl::StrCat("(start_x + ", kx, " * dilation.x)");
        if (manual_clamp) {
          c += "    bool x_out = " + s_x + "< 0 || " + s_x + ">= src_size.x;\n";
          c += "    " + s_type + " src = x_out || y_out ?";
          c += "(" + s_type + ")(0.0) : ";
          c += src_tensor.Read3D(s_x, s_y, std::to_string(s)) + s_postfix +
               ";\n";
        } else {
          c += "    " + s_type + " src = " +
               src_tensor.Read3D(s_x, s_y, std::to_string(s), address_mode) +
               s_postfix + ";\n";
        }
        for (int d = 0; d < out_z; ++d) {
          c += "    " + s_conv + "(r[" + std::to_string(d) + "], src, filters,";
          c += " " + std::to_string(filters_counter) + ");\n";
          filters_counter += ch_count;
        }
        c += "  }\n";
      }
      if (manual_clamp) {
        c += "  }\n";
      }
    }
  }
  for (int i = 0; i < out_z; ++i) {
    std::string s_i = std::to_string(i);
    c += "  {\n";
    c += "    FLT4 res = TO_FLT4(r[" + s_i + "]) + biases[" + s_i + "];\n";
    const LinkingContext context{"res", "X", "Y", s_i};
    c += PostProcess(linked_operations, context);
    c += "  " + dst_tensor.Write3D("res", "X", "Y", s_i);
    c += "  }\n";
  }
  c += "}\n";

  return c;
}

// Adreno can provide up to ~3-4KB of constant memory, but in some cases even
// 3KB can have very bad performance.
int GetAdrenoOptimalMaxConstantSize(int gpu_version) {
  if (gpu_version < 600) {
    return 256 * 10;  // 2.5KB
  } else {
    return 256 * 14;  // 3.5KB
  }
}

int GetOptimalMaxConstantSize(const DeviceInfo& info) {
  if (info.vendor != Vendor::QUALCOMM) {
    // In general we do not expect that this kernel will be used with non Adreno
    // so as it tuned for __constant memory that have big profit on Adreno
    return 1024;  // 1KB
  } else {
    return GetAdrenoOptimalMaxConstantSize(info.adreno_info.gpu_version);
  }
}
}  // namespace

ConvConstants::ConvConstants(ConvConstants&& kernel)
    : GPUOperation(std::move(kernel)),
      weights_(std::move(kernel.weights_)),
      biases_(std::move(kernel.biases_)),
      kernel_size_(kernel.kernel_size_),
      stride_(kernel.stride_),
      padding_(kernel.padding_),
      dilation_(kernel.dilation_),
      src_channels_(kernel.src_channels_),
      dst_channels_(kernel.dst_channels_),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

ConvConstants& ConvConstants::operator=(ConvConstants&& kernel) {
  if (this != &kernel) {
    weights_ = std::move(kernel.weights_);
    biases_ = std::move(kernel.biases_);
    std::swap(kernel_size_, kernel.kernel_size_);
    std::swap(stride_, kernel.stride_);
    std::swap(padding_, kernel.padding_);
    std::swap(dilation_, kernel.dilation_);
    std::swap(src_channels_, kernel.src_channels_);
    std::swap(dst_channels_, kernel.dst_channels_);
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

Status ConvConstants::Compile(const CreationContext& creation_context) {
  const bool stride_correction = definition_.batch_support && stride_.x != 1;
  const auto code = GenerateConvolutionConstantCode(
      definition_, kernel_size_, src_channels_, dst_channels_,
      stride_correction, *creation_context.device, linked_operations_);
  std::vector<CompilerOptions> options;
  if (definition_.precision == CalculationsPrecision::F16 &&
      creation_context.device->IsAdreno3xx()) {
    options.push_back(CompilerOptions::ADRENO_FULL_SIMD_LINE);
  }
  if (definition_.precision != CalculationsPrecision::F32 &&
      creation_context.device->IsPowerVR()) {
    // BUG, some PowerVRs (GE8320) produce incorrect result without it
    options.push_back(CompilerOptions::CL_OPT_DISABLE);
  }
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", options, *creation_context.context,
      *creation_context.device, &kernel_);
}

Status ConvConstants::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_.GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(stride_));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(int2(padding_.x * src_[0]->Batch(), padding_.y)));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(int2(dilation_.x * src_[0]->Batch(), dilation_.y)));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWBatchedHSB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWBatchedHSB()));
  return OkStatus();
}

int3 ConvConstants::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  return int3(grid_x, grid_y, 1);
}

Status ConvConstants::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status ConvConstants::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

bool IsConvConstantsSupported(const CLDevice& device,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr) {
  const auto& w_shape = attr.weights.shape;
  const int dst_channels = AlignByN(w_shape.o, 4);
  const int filters_count = w_shape.i * dst_channels * w_shape.h * w_shape.w;
  const int float_size = definition.precision == CalculationsPrecision::F32
                             ? sizeof(float)
                             : sizeof(half);
  const int filters_buffer_size = filters_count * float_size;
  const int kConstantMaxSize = GetOptimalMaxConstantSize(device.GetInfo());
  const int flt4_registers = IntegralDivideRoundUp(w_shape.o, 4);
  return filters_buffer_size <= kConstantMaxSize && flt4_registers <= 8;
}

Status CreateConvConstants(const CreationContext& creation_context,
                           const OperationDef& definition,
                           const Convolution2DAttributes& attr,
                           ConvConstants* result) {
  if (!IsConvConstantsSupported(*creation_context.device, definition, attr)) {
    return InvalidArgumentError("ConvConstants doesn't supported");
  }
  *result = ConvConstants(definition, attr);
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
