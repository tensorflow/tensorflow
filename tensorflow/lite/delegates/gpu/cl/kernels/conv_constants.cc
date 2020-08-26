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

std::string GenerateConvolutionConstantCode(const OperationDef& op_def,
                                            const OHWI& weights_shape,
                                            bool stride_correction,
                                            GPUOperation* op) {
  auto src_desc = op_def.src_tensors[0];
  src_desc.SetTextureAddressMode(TextureAddressMode::ZERO);
  if (op_def.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddSrcTensor("src_tensor", src_desc);

  auto dst_desc = op_def.dst_tensors[0];
  if (op_def.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddDstTensor("dst_tensor", dst_desc);

  std::string c = GetCommonDefines(op_def.precision);

  const int out_z = DivideRoundUp(weights_shape.o, 4);
  const std::string kOutZ = std::to_string(out_z);
  const int src_depth = DivideRoundUp(weights_shape.i, 4);

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
         GetXStrideCorrectedV2("X", "args.src_tensor.Batch()", "args.stride_x",
                               "args.padding_x") +
         ";\n";
  } else {
    if (op_def.IsBatchSupported()) {
      c += "  int start_x = X * args.stride_x + args.padding_x * "
           "args.src_tensor.Batch();\n";
    } else {
      c += "  int start_x = X * args.stride_x + args.padding_x;\n";
    }
  }
  c += "  int start_y = Y * args.stride_y + args.padding_y;\n";
  c += "  ACCUM_FLT4 r[" + kOutZ + "];\n";
  c += "  for (int i = 0; i < " + kOutZ + "; ++i) {\n";
  c += "    r[i] = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
  c += "  }\n";
  int filters_counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    const int ch_count = std::min(4, weights_shape.i - s * 4);
    const std::string s_conv = "CONV" + std::to_string(ch_count);
    const std::string s_count = ch_count == 1 ? "" : std::to_string(ch_count);
    const std::string s_type = absl::StrCat("FLT", s_count);
    const std::string s_postfix = postfixes[ch_count - 1];
    const std::string dilation_x =
        op_def.IsBatchSupported() ? "args.dilation_x * args.src_tensor.Batch()"
                                  : "args.dilation_x";
    for (int ky = 0; ky < weights_shape.h; ++ky) {
      std::string s_y = absl::StrCat("(start_y + ", ky, " * args.dilation_y)");
      if (manual_clamp) {
        c += "  {\n";
        c += "  bool y_out = " + s_y + " < 0 || " + s_y +
             " >= args.src_tensor.Height();\n";
      }
      for (int kx = 0; kx < weights_shape.w; ++kx) {
        c += "  {\n";
        std::string s_x =
            absl::StrCat("(start_x + ", kx, " * " + dilation_x + ")");
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

}  // namespace

bool IsConvConstantsSupported(const DeviceInfo& device_info,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr) {
  if (device_info.IsAMD() &&
      definition.precision != CalculationsPrecision::F32 &&
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
  const int kConstantMaxSize = GetOptimalMaxConstantSize(device_info);
  const int flt4_registers = DivideRoundUp(w_shape.o, 4);
  return filters_buffer_size <= kConstantMaxSize && flt4_registers <= 8;
}

GPUOperation CreateConvConstants(const DeviceInfo& device_info,
                                 const OperationDef& definition,
                                 const Convolution2DAttributes& attr) {
  GPUOperation op(definition);
  UploadWeightsForConvConstants(attr.weights, definition.precision, &op);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("dilation_x", attr.dilations.w);
  op.args_.AddInt("dilation_y", attr.dilations.h);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;

  const bool stride_correction =
      definition.IsBatchSupported() && attr.strides.w != 1;
  op.code_ = GenerateConvolutionConstantCode(definition, attr.weights.shape,
                                             stride_correction, &op);
  if (definition.precision == CalculationsPrecision::F16 &&
      device_info.IsAdreno3xx()) {
    op.compiler_options_.push_back(CompilerOptions::ADRENO_FULL_SIMD_LINE);
  }
  if (definition.precision != CalculationsPrecision::F32 &&
      device_info.IsPowerVR()) {
    // BUG, some PowerVRs (GE8320) produce incorrect result without it
    op.compiler_options_.push_back(CompilerOptions::CL_OPT_DISABLE);
  }

  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::BUFFER;
  desc.element_type = definition.GetDataType();
  desc.memory_type = MemoryType::CONSTANT;
  desc.UploadLinearData(attr.bias);
  op.args_.AddObject(
      "biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
  return op;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
