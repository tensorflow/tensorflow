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
#include <vector>

#include "absl/strings/substitute.h"
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
    const OperationDef& op_def, const int3& block_size, bool is1x1,
    bool adreno4xx_optimization, bool stride_correction,
    bool different_weights_for_height, const CLDevice& device,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  std::string c = GetCommonDefines(op_def.precision);
  TensorCodeGenerator src_tensor(
      "src_data", WHSPoint{"src_size.x", "src_size.y", "src_size.z"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data", WHSPoint{"dst_size.x", "dst_size.y", "dst_size.z"},
      op_def.dst_tensors[0]);

  const auto src_tensor_type = op_def.src_tensors[0].storage_type;
  const bool is_buffer = src_tensor_type == TensorStorageType::IMAGE_BUFFER ||
                         src_tensor_type == TensorStorageType::BUFFER;

  std::vector<std::string> xs(block_size.x);
  for (int x = 0; x < block_size.x; ++x) {
    xs[x] = std::to_string(x);
  }

  std::vector<std::string> ys(block_size.y);
  for (int y = 0; y < block_size.y; ++y) {
    ys[y] = std::to_string(y);
  }

  std::vector<std::string> zs(block_size.z);
  for (int z = 0; z < block_size.z; ++z) {
    zs[z] = std::to_string(z);
  }

  for (int z = 0; z < block_size.z; ++z) {
    const std::string f0 = std::to_string(z * 4 + 0);
    const std::string f1 = std::to_string(z * 4 + 1);
    const std::string f2 = std::to_string(z * 4 + 2);
    const std::string f3 = std::to_string(z * 4 + 3);
    switch (op_def.precision) {
      case CalculationsPrecision::F32:
      case CalculationsPrecision::F16:
        c += "#define CONV" + zs[z] + "(R, S)    \\\n";
        c += "R += S.x * f" + f0 + "; \\\n";
        c += "R += S.y * f" + f1 + "; \\\n";
        c += "R += S.z * f" + f2 + "; \\\n";
        c += "R += S.w * f" + f3 + ";   \n";
        break;
      case CalculationsPrecision::F32_F16:
        c += "#define CONV" + zs[z] + "(R, S) \\\n";
        c += "R += convert_float4(S.x * f" + f0 + " + S.y * f" + f1 +
             " + S.z * f" + f2 + " + S.w * f" + f3 + ");\n";
        break;
    }
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
  c += "  int X = get_global_id(0) * " + std::to_string(block_size.x) + ";\n";
  c += "  int Y = get_global_id(1) * " + std::to_string(block_size.y) + ";\n";
  c += "  int Z = get_global_id(2) * " + std::to_string(block_size.z) + ";\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;\n";
  std::vector<std::string> s_x(block_size.x);
  std::vector<std::string> s_y(block_size.y);
  for (int x = 0; x < block_size.x; ++x) {
    if (stride_correction) {
      c += "  int xc" + xs[x] + " = " +
           GetXStrideCorrected("X + " + xs[x], "src_size.w", "stride.x",
                               "padding.x") +
           ";\n";
    } else {
      c += "  int xc" + xs[x] + " = (X +" + xs[x] +
           ") * stride.x + padding.x;\n";
    }
    s_x[x] = is1x1 ? "xc" + xs[x] : "cx" + xs[x];
  }
  for (int y = 0; y < block_size.y; ++y) {
    c += "  int yc" + ys[y] + " = (Y +" + ys[y] + ") * stride.y + padding.y;\n";
    s_y[y] = is1x1 ? "yc" + ys[y] : "cy" + ys[y];
  }
  for (int i = 0; i < block_size.x * block_size.y * block_size.z; ++i) {
    c += "  ACCUM_FLT4 r" + std::to_string(i) +
         " = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
  }
  std::string f_y = is1x1 ? "s" : "filter_offset";
  if (different_weights_for_height) {
    f_y = "Y * src_size.z + s";
  }
  if (!is1x1) {
    for (int x = 0; x < block_size.x; ++x) {
      c += "  int cx" + xs[x] + ";\n";
    }
    for (int y = 0; y < block_size.y; ++y) {
      c += "  int cy" + ys[y] + ";\n";
    }
    c += "  int filter_offset = 0;\n";
    c += "  for (int y = 0; y < kernel_size.y; ++y) {\n";
    for (int y = 0; y < block_size.y; ++y) {
      c += "  cy" + ys[y] + " = y * dilation.y + yc" + ys[y] + ";\n";
    }
    if (is_buffer) {
      for (int y = 0; y < block_size.y; ++y) {
        c += "  bool in_y" + ys[y] + " = cy" + ys[y] + " >= 0 && cy" + ys[y] +
             " < src_size.y;\n";
        if (src_tensor_type == TensorStorageType::BUFFER) {
          c += "    cy" + ys[y] + " = clamp(cy" + ys[y] +
               ", 0, src_size.y - 1);\n";
        }
      }
    }
    c += "  for (int x = 0; x < kernel_size.x; ++x) {\n";
    for (int x = 0; x < block_size.x; ++x) {
      c += "  cx" + xs[x] + " = x * dilation.x + xc" + xs[x] + ";\n";
    }
    if (is_buffer) {
      for (int x = 0; x < block_size.x; ++x) {
        c += "  bool in_x" + xs[x] + " = cx" + xs[x] + " >= 0 && cx" + xs[x] +
             " < src_size.x;\n";
        if (src_tensor_type == TensorStorageType::BUFFER) {
          c += "    cx" + xs[x] + " = clamp(cx" + xs[x] +
               ", 0, src_size.x - 1);\n";
        }
      }
      for (int x = 0; x < block_size.x; ++x) {
        for (int y = 0; y < block_size.y; ++y) {
          const std::string id = std::to_string(y * block_size.x + x);
          if (src_tensor_type == TensorStorageType::IMAGE_BUFFER) {
            c += absl::Substitute(
                "  int addr_$0 = select(-1, cy$2 * src_size.x + cx$1, (in_x$1 "
                "&& "
                "in_y$2));\n",
                y * block_size.x + x, x, y);
            c += absl::Substitute(
                "  int dz_$0 = select(0, src_size.x * src_size.y, (in_x$1 && "
                "in_y$2));\n",
                y * block_size.x + x, x, y);
          } else {
            c += absl::Substitute("  int addr_$0 = cy$2 * src_size.x + cx$1;\n",
                                  y * block_size.x + x, x, y);
          }
        }
      }
      if (src_tensor_type == TensorStorageType::BUFFER) {
        c += "  int dz = src_size.x * src_size.y;\n";
      }
    }
  } else if (is_buffer) {
    for (int y = 0; y < block_size.y; ++y) {
      c += "  bool in_y" + ys[y] + " = yc" + ys[y] + " >= 0 && yc" + ys[y] +
           " < src_size.y;\n";
    }
    for (int x = 0; x < block_size.x; ++x) {
      c += "  bool in_x" + xs[x] + " = xc" + xs[x] + " >= 0 && xc" + xs[x] +
           " < src_size.x;\n";
    }
    for (int x = 0; x < block_size.x; ++x) {
      for (int y = 0; y < block_size.y; ++y) {
        const std::string id = std::to_string(y * block_size.x + x);
        if (src_tensor_type == TensorStorageType::IMAGE_BUFFER) {
          c += absl::Substitute(
              "  int addr_$0 = select(-1, yc$2 * src_size.x + xc$1, (in_x$1 && "
              "in_y$2));\n",
              y * block_size.x + x, x, y);
          c += absl::Substitute(
              "  int dz_$0 = select(0, src_size.x * src_size.y, (in_x$1 && "
              "in_y$2));\n",
              y * block_size.x + x, x, y);
        } else {
          c += absl::Substitute("  int addr_$0 = yc$2 * src_size.x + xc$1;\n",
                                y * block_size.x + x, x, y);
        }
      }
    }
    if (src_tensor_type == TensorStorageType::BUFFER) {
      c += "  int dz = src_size.x * src_size.y;\n";
    }
  }
  c += "  for (int s = 0; s < src_size.z; ++s) {\n";
  if (is_buffer) {
    if (src_tensor_type == TensorStorageType::IMAGE_BUFFER) {
      for (int index = 0; index < block_size.x * block_size.y; ++index) {
        const std::string id = std::to_string(index);
        c +=
            "    FLT4 src" + id + " = " + src_tensor.Read("addr_" + id) + ";\n";
      }
    } else {
      for (int x = 0; x < block_size.x; ++x) {
        for (int y = 0; y < block_size.y; ++y) {
          const std::string id = std::to_string(y * block_size.x + x);
          c += "    FLT4 src" + id + " = " + src_tensor.Read("addr_" + id) +
               " * (FLT)(in_x" + xs[x] + " && in_y" + ys[y] + "); addr_" + id +
               " += dz;\n";
        }
      }
    }
  }
  for (int z = 0; z < block_size.z; ++z) {
    const std::string fc = "(int2)(Z + " + zs[z] + ", " + f_y + ")";
    c += absl::Substitute(R"(    FLT4 f$1 = READ_IMAGE(filters0, smp_none, $0);
    FLT4 f$2 = READ_IMAGE(filters1, smp_none, $0);
    FLT4 f$3 = READ_IMAGE(filters2, smp_none, $0);
    FLT4 f$4 = READ_IMAGE(filters3, smp_none, $0);
)",
                          fc, z * 4 + 0, z * 4 + 1, z * 4 + 2, z * 4 + 3);
  }
  if (!is_buffer) {
    const auto mode = GetFastestZeroMode(device);
    for (int x = 0; x < block_size.x; ++x) {
      for (int y = 0; y < block_size.y; ++y) {
        const std::string id = std::to_string(y * block_size.x + x);
        c += "    FLT4 src" + id + " = " +
             src_tensor.ReadWHS(s_x[x], s_y[y], "s", mode) + ";\n";
      }
    }
  }
  for (int z = 0; z < block_size.z; ++z) {
    for (int i = 0; i < block_size.x * block_size.y; ++i) {
      c += "    CONV" + zs[z] + "(r" +
           std::to_string(i + z * block_size.x * block_size.y) + ", src" +
           std::to_string(i) + ");\n";
    }
  }
  if (!is1x1) {
    c += "    filter_offset++;\n";
  }
  if (is_buffer) {
    if (src_tensor_type == TensorStorageType::IMAGE_BUFFER) {
      for (int index = 0; index < block_size.x * block_size.y; ++index) {
        const std::string id = std::to_string(index);
        c += "     addr_" + id + " += dz_" + id + ";\n";
      }
    }
  }
  c += "  }\n";  // src_size.z
  if (!is1x1) {
    c += "  }\n";  // kernel_size.x
    c += "  }\n";  // kernel_size.y
  }
  // when is1x1 && adreno4xx_optimization is true, xc0 == X and yc0 == Y
  std::string dst_x = is1x1 && adreno4xx_optimization ? "xc0" : "X";
  std::string dst_y = is1x1 && adreno4xx_optimization ? "yc0" : "Y";
  for (int z = 0; z < block_size.z; ++z) {
    c += "  if (Z < dst_size.z) {\n";
    c += "    FLT4 bias_val = READ_IMAGE(biases, smp_none, (int2)(Z, 0));\n";
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x; ++x) {
        const std::string id =
            std::to_string((z * block_size.y + y) * block_size.x + x);
        c += "    {\n";
        c += "      int xc = " + dst_x + " + " + xs[x] + ";\n";
        c += "      int yc = " + dst_y + " + " + ys[y] + ";\n";
        c += "      if (xc < dst_size.x && yc < dst_size.y) {\n";
        c += "        FLT4 res = TO_FLT4(r" + id + ") + bias_val;\n";
        const LinkingContext context{"res", "xc", "yc", "Z"};
        c += PostProcess(linked_operations, context);
        c += "        " + dst_tensor.WriteWHS("res", "xc", "yc", "Z") + "\n";
        c += "      }\n";
        c += "    }\n";
      }
    }
    c += "  }\n";
    c += "  Z++;\n";
  }
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
      different_weights_for_height_(false),
      block_size_(2, 2, 2),
      work_group_size_(4, 4, 2) {}

ConvTexture::ConvTexture(const OperationDef& definition)
    : GPUOperation(definition),
      kernel_size_(1, 1),
      stride_(1, 1),
      padding_(0, 0),
      dilation_(1, 1),
      different_weights_for_height_(false),
      block_size_(4, 1, 2),
      work_group_size_(16, 1, 2) {}

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
      different_weights_for_height_(operation.different_weights_for_height_),
      block_size_(operation.block_size_),
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
    std::swap(different_weights_for_height_,
              operation.different_weights_for_height_);
    std::swap(block_size_, operation.block_size_);
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
  const bool stride_correction =
      definition_.IsBatchSupported() && stride_.x != 1;
  const std::string code =
      GenerateConvCode(definition_, block_size_, is1x1, adreno4xx_optimization,
                       stride_correction, different_weights_for_height_,
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
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWBatchedHSB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWBatchedHSB()));
  if (!(kernel_size_.x == 1 && kernel_size_.y == 1)) {
    RETURN_IF_ERROR(kernel_.SetBytesAuto(kernel_size_));
    RETURN_IF_ERROR(kernel_.SetBytesAuto(
        int2(dilation_.x * src_[0]->Batch(), dilation_.y)));
  }
  RETURN_IF_ERROR(kernel_.SetBytesAuto(stride_));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(int2(padding_.x * src_[0]->Batch(), padding_.y)));
  return OkStatus();
}

int3 ConvTexture::GetGridSize() const {
  const int grid_x =
      IntegralDivideRoundUp(dst_[0]->Width() * dst_[0]->Batch(), block_size_.x);
  const int grid_y = IntegralDivideRoundUp(dst_[0]->Height(), block_size_.y);
  const int grid_z = IntegralDivideRoundUp(dst_[0]->Slices(), block_size_.z);
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
  return result->UploadData(attr.weights, attr.bias, creation_context.context);
}

Status CreateConvTexture(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const FullyConnectedAttributes& attr,
                         ConvTexture* result) {
  *result = ConvTexture(definition);
  return result->UploadData(attr.weights, attr.bias, creation_context.context);
}

Status CreateConvTextureWino4x4To6x6(const CreationContext& creation_context,
                                     const OperationDef& definition,
                                     const Convolution2DAttributes& attr,
                                     ConvTexture* result) {
  *result = ConvTexture(definition);
  result->different_weights_for_height_ = true;
  result->block_size_ = {4, 1, 2};
  return result->UploadDataForWinograd4x4To6x6(
      attr.weights, *creation_context.device, creation_context.context);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
