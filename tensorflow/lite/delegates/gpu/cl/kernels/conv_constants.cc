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
  if (!info.IsAdreno()) {
    // In general we do not expect that this kernel will be used with non Adreno
    // so as it tuned for __constant memory that have big profit on Adreno
    return 1024;  // 1KB
  } else {
    return GetAdrenoOptimalMaxConstantSize(info.adreno_info.gpu_version);
  }
}
}  // namespace

ConvConstants::ConvConstants(const OperationDef& definition,
                             const Convolution2DAttributes& attr,
                             const DeviceInfo& device_info)
    : GPUOperation(definition),
      kernel_size_(attr.weights.shape.w, attr.weights.shape.h),
      stride_(attr.strides.w, attr.strides.h),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h),
      dilation_(attr.dilations.w, attr.dilations.h),
      src_channels_(attr.weights.shape.i),
      dst_channels_(attr.weights.shape.o) {
  const bool stride_correction =
      definition_.IsBatchSupported() && stride_.x != 1;
  code_ = GenerateConvolutionConstantCode(definition_, kernel_size_,
                                          src_channels_, dst_channels_,
                                          stride_correction, device_info);
  if (definition_.precision == CalculationsPrecision::F16 &&
      device_info.IsAdreno3xx()) {
    compiler_options_.push_back(CompilerOptions::ADRENO_FULL_SIMD_LINE);
  }
  if (definition_.precision != CalculationsPrecision::F32 &&
      device_info.IsPowerVR()) {
    // BUG, some PowerVRs (GE8320) produce incorrect result without it
    compiler_options_.push_back(CompilerOptions::CL_OPT_DISABLE);
  }
}

ConvConstants::ConvConstants(ConvConstants&& kernel)
    : GPUOperation(std::move(kernel)),
      kernel_size_(kernel.kernel_size_),
      stride_(kernel.stride_),
      padding_(kernel.padding_),
      dilation_(kernel.dilation_),
      src_channels_(kernel.src_channels_),
      dst_channels_(kernel.dst_channels_) {}

ConvConstants& ConvConstants::operator=(ConvConstants&& kernel) {
  if (this != &kernel) {
    std::swap(kernel_size_, kernel.kernel_size_);
    std::swap(stride_, kernel.stride_);
    std::swap(padding_, kernel.padding_);
    std::swap(dilation_, kernel.dilation_);
    std::swap(src_channels_, kernel.src_channels_);
    std::swap(dst_channels_, kernel.dst_channels_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

std::string ConvConstants::GenerateConvolutionConstantCode(
    const OperationDef& op_def, const int2& kernel_size, int src_channels,
    int dst_channels, bool stride_correction, const DeviceInfo& device_info) {
  auto src_desc = op_def.src_tensors[0];
  src_desc.SetTextureAddressMode(GetFastestZeroMode(device_info));
  if (op_def.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  AddSrcTensor("src_tensor", src_desc);

  auto dst_desc = op_def.dst_tensors[0];
  if (op_def.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  AddDstTensor("dst_tensor", dst_desc);

  args_.AddInt("stride_x");
  args_.AddInt("stride_y");
  args_.AddInt("padding_x");
  args_.AddInt("padding_y");
  args_.AddInt("dilation_x");
  args_.AddInt("dilation_y");

  std::string c = GetCommonDefines(op_def.precision);

  const int out_z = DivideRoundUp(dst_channels, 4);
  const std::string kOutZ = std::to_string(out_z);
  const int src_depth = DivideRoundUp(src_channels, 4);

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
  c += "$0) {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
       "return;\n";
  if (stride_correction) {
    c += "  int start_x = " +
         GetXStrideCorrected("X", "args.src_tensor.Batch()", "args.stride_x",
                             "args.padding_x") +
         ";\n";
  } else {
    c += "  int start_x = X * args.stride_x + args.padding_x;\n";
  }
  c += "  int start_y = Y * args.stride_y + args.padding_y;\n";
  c += "  ACCUM_FLT4 r[" + kOutZ + "];\n";
  c += "  for (int i = 0; i < " + kOutZ + "; ++i) {\n";
  c += "    r[i] = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
  c += "  }\n";
  int filters_counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    const int ch_count = std::min(4, src_channels - s * 4);
    const std::string s_conv = "CONV" + std::to_string(ch_count);
    const std::string s_count = ch_count == 1 ? "" : std::to_string(ch_count);
    const std::string s_type = absl::StrCat("FLT", s_count);
    const std::string s_postfix = postfixes[ch_count - 1];
    for (int ky = 0; ky < kernel_size.y; ++ky) {
      std::string s_y = absl::StrCat("(start_y + ", ky, " * args.dilation_y)");
      if (manual_clamp) {
        c += "  {\n";
        c += "  bool y_out = " + s_y + " < 0 || " + s_y +
             " >= args.src_tensor.Height();\n";
      }
      for (int kx = 0; kx < kernel_size.x; ++kx) {
        c += "  {\n";
        std::string s_x =
            absl::StrCat("(start_x + ", kx, " * args.dilation_x)");
        if (manual_clamp) {
          c += "    bool x_out = " + s_x + "< 0 || " + s_x +
               ">= args.src_tensor.Width();\n";
          c += "    " + s_type + " src = x_out || y_out ?";
          c += "(" + s_type + ")(0.0) : args.src_tensor.Read(" + s_x + ", " +
               s_y + ", " + std::to_string(s) + ")" + s_postfix + ";\n";
        } else {
          c += "    " + s_type + " src = args.src_tensor.Read(" + s_x + ", " +
               s_y + ", " + std::to_string(s) + ")" + s_postfix + ";\n";
        }
        for (int d = 0; d < out_z; ++d) {
          c += "    " + s_conv + "(r[" + std::to_string(d) +
               "], src, args.weigths.GetPtr(),";
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
    c += "    FLT4 res = TO_FLT4(r[" + s_i + "]) + args.biases.Read(" + s_i +
         ");\n";
    c += "  args.dst_tensor.Write(res, X, Y, " + s_i + ");\n";
    c += "  }\n";
  }
  c += "}\n";
  return c;
}

absl::Status ConvConstants::BindArguments() {
  RETURN_IF_ERROR(args_.SetInt("stride_x", stride_.x));
  RETURN_IF_ERROR(args_.SetInt("stride_y", stride_.y));
  RETURN_IF_ERROR(args_.SetInt("padding_x", padding_.x * src_[0]->Batch()));
  RETURN_IF_ERROR(args_.SetInt("padding_y", padding_.y));
  RETURN_IF_ERROR(args_.SetInt("dilation_x", dilation_.x * src_[0]->Batch()));
  return args_.SetInt("dilation_y", dilation_.y);
}

int3 ConvConstants::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  return int3(grid_x, grid_y, 1);
}

bool IsConvConstantsSupported(const CLDevice& device,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr) {
  if (device.IsAMD() && definition.precision != CalculationsPrecision::F32 &&
      definition.src_tensors[0].storage_type != TensorStorageType::BUFFER) {
    // BUG, some AMD gpus crashe without it
    return false;
  }

  const auto& w_shape = attr.weights.shape;
  const int dst_channels = AlignByN(w_shape.o, 4);
  const int filters_count = w_shape.i * dst_channels * w_shape.h * w_shape.w;
  const int float_size = definition.precision == CalculationsPrecision::F32
                             ? sizeof(float)
                             : sizeof(half);
  const int filters_buffer_size = filters_count * float_size;
  const int kConstantMaxSize = GetOptimalMaxConstantSize(device.GetInfo());
  const int flt4_registers = DivideRoundUp(w_shape.o, 4);
  return filters_buffer_size <= kConstantMaxSize && flt4_registers <= 8;
}

absl::Status CreateConvConstants(const CreationContext& creation_context,
                                 const OperationDef& definition,
                                 const Convolution2DAttributes& attr,
                                 ConvConstants* result) {
  if (!IsConvConstantsSupported(*creation_context.device, definition, attr)) {
    return absl::InvalidArgumentError("ConvConstants doesn't supported");
  }
  *result = ConvConstants(definition, attr, creation_context.device->GetInfo());
  RETURN_IF_ERROR(
      result->UploadWeights(attr.weights, creation_context.context));

  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::BUFFER;
  desc.element_type = definition.GetDataType();
  desc.memory_type = MemoryType::CONSTANT;

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
