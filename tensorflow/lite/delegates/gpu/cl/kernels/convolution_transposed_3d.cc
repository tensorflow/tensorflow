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

#include "tensorflow/lite/delegates/gpu/cl/kernels/convolution_transposed_3d.h"

#include <string>
#include <utility>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GenerateConvolutionTransposed3DCode(
    const OperationDef& op_def, const LinearStorage& biases,
    const CLDevice& device, bool weights_are_buffer, const int4& block_size,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor(
      "src_data",
      WHDSBPoint{"src_size.x", "src_size.y", "src_size.z", "src_size.w",
                 "batch_size"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data",
      WHDSBPoint{"dst_size.x", "dst_size.y", "dst_size.z", "dst_size.w",
                 "batch_size"},
      op_def.dst_tensors[0]);

  const auto src_tensor_type = op_def.src_tensors[0].storage_type;
  bool image_buffer = src_tensor_type == TensorStorageType::IMAGE_BUFFER;
  bool manual_clamp =
      image_buffer || src_tensor_type == TensorStorageType::BUFFER;

  const std::string batch_id = op_def.IsBatchSupported() ? "B" : "";
  std::string c = GetCommonDefines(op_def.precision);

  for (int s = 0; s < block_size.w; ++s) {
    const std::string f0 =
        weights_are_buffer ? "weights_cache[" + std::to_string(s) + "].s0123"
                           : "f" + std::to_string(s * 4 + 0);
    const std::string f1 =
        weights_are_buffer ? "weights_cache[" + std::to_string(s) + "].s4567"
                           : "f" + std::to_string(s * 4 + 1);
    const std::string f2 =
        weights_are_buffer ? "weights_cache[" + std::to_string(s) + "].s89ab"
                           : "f" + std::to_string(s * 4 + 2);
    const std::string f3 =
        weights_are_buffer ? "weights_cache[" + std::to_string(s) + "].scdef"
                           : "f" + std::to_string(s * 4 + 3);
    switch (op_def.precision) {
      case CalculationsPrecision::F32:
      case CalculationsPrecision::F16:
        c += "#define CONV" + std::to_string(s) + "(R, S)    \\\n";
        c += "R += S.x * " + f0 + "; \\\n";
        c += "R += S.y * " + f1 + "; \\\n";
        c += "R += S.z * " + f2 + "; \\\n";
        c += "R += S.w * " + f3 + ";   \n";
        break;
      case CalculationsPrecision::F32_F16:
        c += "#define CONV" + std::to_string(s) + "(R, S) \\\n";
        c += "R += convert_float4(S.x * " + f0 + " + S.y * " + f1 +
             " + S.z * " + f2 + " + S.w * " + f3 + ");\n";
        break;
    }
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
  if (weights_are_buffer) {
    c += "    __global FLT16* filters,  \n";
  } else {
    c += "    __read_only image2d_t filters0,  \n";
    c += "    __read_only image2d_t filters1,  \n";
    c += "    __read_only image2d_t filters2,  \n";
    c += "    __read_only image2d_t filters3,  \n";
  }
  c += biases.GetDeclaration();
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 kernel_size,          \n";
  c += "    int4 stride,               \n";
  c += "    int4 padding,              \n";
  if (op_def.IsBatchSupported()) {
    c += "    int batch_size,          \n";
  }
  c += "    int grid_size_s,           \n";
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size              \n";
  c += ") {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int dst_x = (linear_id / batch_size);\n";
    c += "  int B = linear_id % batch_size;\n";
  } else {
    c += "  int dst_x = get_global_id(0);\n";
  }
  c += "  int rem_x = dst_x % stride.x;\n";
  c += "  int ceil_x = dst_x / stride.x;\n";
  c += "  dst_x = ceil_x * stride.x * " + std::to_string(block_size.x) +
       " + rem_x;\n";
  c += "  int dst_y = get_global_id(1);\n";
  c += "  int rem_y = dst_y % stride.y;\n";
  c += "  int ceil_y = dst_y / stride.y;\n";
  c += "  dst_y = ceil_y * stride.y * " + std::to_string(block_size.y) +
       " + rem_y;\n";
  c += "  int linear_id_z = get_global_id(2);\n";
  c += "  int S = (linear_id_z % grid_size_s) * " +
       std::to_string(block_size.w) + ";\n";
  c += "  int dst_z = linear_id_z / grid_size_s;\n";
  c += "  int rem_z = dst_z % stride.z;\n";
  c += "  int ceil_z = dst_z / stride.z;\n";
  c += "  dst_z = ceil_z * stride.z * " + std::to_string(block_size.z) +
       " + rem_z;\n";
  c += "  if (dst_x >= dst_size.x || dst_y >= dst_size.y || dst_z >= "
       "dst_size.z) return;\n";
  if (weights_are_buffer) {
    c += "  int f_base = S * src_size.w * kernel_size.x * kernel_size.y * "
         "kernel_size.z;\n";
  }
  for (int i = 0; i < block_size.x * block_size.y * block_size.z * block_size.w;
       ++i) {
    c += "  ACCUM_FLT4 r" + std::to_string(i) +
         " = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
  }
  c += "  int kernel_first_dst_x = dst_x + padding.x;\n";
  c += "  int kernel_first_dst_y = dst_y + padding.y;\n";
  c += "  int kernel_first_dst_z = dst_z + padding.z;\n";
  c += "  int kernel_last_dst_x = kernel_first_dst_x - kernel_size.x;\n";
  c += "  int kernel_last_dst_y = kernel_first_dst_y - kernel_size.y;\n";
  c += "  int kernel_last_dst_z = kernel_first_dst_z - kernel_size.z;\n";
  c += "  int offset_x = abs(padding.x);\n";
  c += "  int offset_x_strided = offset_x * stride.x;\n";
  c += "  int src_x = (kernel_first_dst_x + offset_x_strided) / stride.x - "
       "offset_x;\n";
  c += "  int offset_y = abs(padding.y);\n";
  c += "  int offset_y_strided = offset_y * stride.y;\n";
  c += "  int src_y = (kernel_first_dst_y + offset_y_strided) / stride.y - "
       "offset_y;\n";
  c += "  int offset_z = abs(padding.z);\n";
  c += "  int offset_z_strided = offset_z * stride.z;\n";
  c += "  int src_z = (kernel_first_dst_z + offset_z_strided) / stride.z - "
       "offset_z;\n";
  c += "  int src_as_dst_z = src_z * stride.z;\n";
  c += "  for (;src_as_dst_z > kernel_last_dst_z; src_z -= 1, src_as_dst_z -= "
       "stride.z) {\n";
  for (int z = 0; z < block_size.z; ++z) {
    const std::string zindex = std::to_string(z);
    c += "    int sz" + zindex + " = src_z + " + zindex + ";\n";
    if (src_tensor_type != TensorStorageType::TEXTURE_3D) {
      c += "    bool in_z" + zindex + " = sz" + zindex + " >= 0 && sz" +
           zindex + " < src_size.z;\n";
    }
  }
  if (block_size.z == 1 && (src_tensor_type != TensorStorageType::TEXTURE_3D)) {
    c += "    if (!in_z0) continue;\n";
  }
  c += "    int kernel_z = kernel_first_dst_z - src_as_dst_z;\n";
  c += "    int src_as_dst_y = src_y * stride.y;\n";
  c += "    int src_y_copy = src_y;\n";
  c += "    for (;src_as_dst_y > kernel_last_dst_y; src_y_copy -= 1, "
       "src_as_dst_y -= "
       "stride.y) {\n";
  for (int y = 0; y < block_size.y; ++y) {
    const std::string yindex = std::to_string(y);
    c += "      int sy" + yindex + " = src_y_copy + " + yindex + ";\n";
    if (manual_clamp) {
      c += "      bool in_y" + yindex + " = sy" + yindex + " >= 0 && sy" +
           yindex + " < src_size.y;\n";
      if (!image_buffer) {
        c += "      sy" + yindex + " = clamp(sy" + yindex +
             ", 0, src_size.y - 1);\n";
      }
    }
  }
  c += "      int kernel_y = kernel_first_dst_y - src_as_dst_y;\n";
  c += "      int src_as_dst_x = src_x * stride.x;\n";
  c += "      int src_x_copy = src_x;\n";
  c += "      for (;src_as_dst_x > kernel_last_dst_x; src_x_copy -= 1, "
       "src_as_dst_x "
       "-= stride.x) {\n";
  for (int x = 0; x < block_size.x; ++x) {
    const std::string xindex = std::to_string(x);
    c += "        int sx" + xindex + " = src_x_copy + " + xindex + ";\n";
    if (manual_clamp) {
      c += "        bool in_x" + xindex + " = sx" + xindex + " >= 0 && sx" +
           xindex + " < src_size.x;\n";
      if (!image_buffer) {
        c += "        sx" + xindex + " = clamp(sx" + xindex +
             ", 0, src_size.x - 1);\n";
      }
    }
  }
  const std::string layer_offset =
      std::string("src_size.x * src_size.y") +
      (op_def.IsBatchSupported() ? " * batch_size" : "");
  for (int z = 0; z < block_size.z; ++z) {
    const std::string zindex = std::to_string(z);
    for (int y = 0; y < block_size.y; ++y) {
      const std::string yindex = std::to_string(y);
      for (int x = 0; x < block_size.x; ++x) {
        const std::string xindex = std::to_string(x);
        const std::string id =
            std::to_string((z * block_size.y + y) * block_size.x + x);
        if (image_buffer) {
          c += "        " + src_tensor.GetAddressWHDSB(
                                "addr_" + id, "sx" + xindex, "sy" + yindex,
                                "sz" + zindex, "0", batch_id);
          c += "        addr_" + id + " = select(-1, addr_" + id + ", (in_x" +
               xindex + " && in_y" + yindex + "));\n";
          c += absl::Substitute(
              "        int dz_$0 = select(0, $3, (in_x$1 && "
              "in_y$2));\n",
              id, x, y, layer_offset);
        } else {
          c += "        " + src_tensor.GetAddressWHDSB(
                                "addr_" + id, "sx" + xindex, "sy" + yindex,
                                "sz" + zindex, "0", batch_id);
        }
      }
    }
  }
  if (src_tensor_type == TensorStorageType::BUFFER) {
    c += "        int dz = " + layer_offset + ";\n";
  }
  if (block_size.x == 1 && block_size.y == 1 && manual_clamp) {
    c += "        if (!in_x0 || !in_y0) continue;\n";
  }
  c += "        int kernel_x = kernel_first_dst_x - src_as_dst_x;\n";
  c += "        int kernel_index = (kernel_z * kernel_size.y + kernel_y) * "
       "kernel_size.x + kernel_x;\n";
  if (weights_are_buffer) {
    c += "        int f_offset = f_base + kernel_index * src_size.w * " +
         std::to_string(block_size.w) + ";\n";
  } else {
    c += "        int x_c = kernel_index * src_size.w;\n";
  }
  c += "        for (int s = 0; s < src_size.w; ++s) {\n";
  const auto mode = GetFastestZeroMode(device);
  for (int y = 0; y < block_size.y; ++y) {
    const std::string yindex = std::to_string(y);
    for (int x = 0; x < block_size.x; ++x) {
      const std::string xindex = std::to_string(x);
      const std::string id = std::to_string(y * block_size.x + x);
      if (image_buffer) {
        c += "          FLT4 src" + id + " = " + src_tensor.Read("addr_" + id) +
             "; addr_" + id + " += dz_" + id + ";\n";
      } else if (manual_clamp) {
        c += "          FLT4 src" + id + " = " + src_tensor.Read("addr_" + id) +
             " * (FLT)(in_x" + xindex + " && in_y" + yindex + "); addr_" + id +
             " += dz;\n";
      } else {
        c += "          FLT4 src" + id + " = " +
             src_tensor.ReadWHDSB("sx" + xindex, "sy" + yindex, "sz0", "s",
                                  batch_id, mode) +
             ";\n";
      }
    }
  }
  if (weights_are_buffer) {
    c += "          __global FLT16* weights_cache = filters + f_offset;\n";
    c += "          f_offset += " + std::to_string(block_size.w) + ";\n";
  } else {
    for (int z = 0; z < block_size.w; ++z) {
      const std::string fc = "(int2)(S + " + std::to_string(z) + ", x_c)";
      c += absl::Substitute(
          R"(          FLT4 f$1 = READ_IMAGE(filters0, smp_none, $0);
          FLT4 f$2 = READ_IMAGE(filters1, smp_none, $0);
          FLT4 f$3 = READ_IMAGE(filters2, smp_none, $0);
          FLT4 f$4 = READ_IMAGE(filters3, smp_none, $0);
)",
          fc, z * 4 + 0, z * 4 + 1, z * 4 + 2, z * 4 + 3);
    }
    c += "          x_c++;\n";
  }
  for (int z = 0; z < block_size.w; ++z) {
    for (int i = 0; i < block_size.x * block_size.y * block_size.z; ++i) {
      c += "          CONV" + std::to_string(z) + "(r" +
           std::to_string(i + z * block_size.x * block_size.y * block_size.z) +
           ", src" + std::to_string(i) + ");\n";
    }
  }
  c += "        }\n";
  c += "      }\n";
  c += "    }\n";
  c += "  }\n";
  for (int s = 0; s < block_size.w; ++s) {
    c += "  if (S < dst_size.w) {\n";
    c += "    FLT4 bias_val = " + biases.ReadLinearFLT4("S") + ";\n";
    for (int z = 0; z < block_size.z; ++z) {
      for (int y = 0; y < block_size.y; ++y) {
        for (int x = 0; x < block_size.x; ++x) {
          const std::string id = std::to_string(
              ((s * block_size.z + z) * block_size.y + y) * block_size.x + x);
          c += "    {\n";
          c += "      int xc = dst_x + stride.x * " + std::to_string(x) + ";\n";
          c += "      int yc = dst_y + stride.y * " + std::to_string(y) + ";\n";
          c += "      int zc = dst_z + stride.z * " + std::to_string(z) + ";\n";
          c += "      if (xc < dst_size.x && yc < dst_size.y && zc < "
               "dst_size.z) {\n";
          c += "        FLT4 res = TO_FLT4(r" + id + ") + bias_val;\n";
          std::string x_3dcoord =
              op_def.IsBatchSupported() ? "xc * dst_size.w + B" : "xc";
          const LinkingContext context{"res", x_3dcoord, "yc", "S"};
          c += PostProcess(linked_operations, context);
          c += "        " +
               dst_tensor.WriteWHDSB("res", "xc", "yc", "zc", "S", batch_id) +
               "\n";
          c += "      }\n";
          c += "    }\n";
        }
      }
    }
    c += "  }\n";
    c += "  S++;\n";
  }
  c += "}\n";
  return c;
}
}  // namespace

ConvolutionTransposed3D::ConvolutionTransposed3D(
    const OperationDef& definition,
    const ConvolutionTransposed3DAttributes& attr, const CLDevice& device)
    : GPUOperation(definition),
      weights_are_buffer_(device.IsMali()),
      kernel_size_(attr.weights.shape.w, attr.weights.shape.h,
                   attr.weights.shape.d),
      stride_(attr.stride.w, attr.stride.h, attr.stride.d),
      padding_(attr.padding.prepended.w, attr.padding.prepended.h,
               attr.padding.prepended.d),
      block_size_(2, 2, 1, 2) {}

ConvolutionTransposed3D::ConvolutionTransposed3D(
    ConvolutionTransposed3D&& operation)
    : GPUOperation(std::move(operation)),
      biases_(std::move(operation.biases_)),
      weights_0_(std::move(operation.weights_0_)),
      weights_1_(std::move(operation.weights_1_)),
      weights_2_(std::move(operation.weights_2_)),
      weights_3_(std::move(operation.weights_3_)),
      weights_buf_(std::move(operation.weights_buf_)),
      weights_are_buffer_(operation.weights_are_buffer_),
      kernel_size_(operation.kernel_size_),
      stride_(operation.stride_),
      padding_(operation.padding_),
      block_size_(operation.block_size_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

ConvolutionTransposed3D& ConvolutionTransposed3D::operator=(
    ConvolutionTransposed3D&& operation) {
  if (this != &operation) {
    biases_ = std::move(operation.biases_);
    weights_0_ = std::move(operation.weights_0_);
    weights_1_ = std::move(operation.weights_1_);
    weights_2_ = std::move(operation.weights_2_);
    weights_3_ = std::move(operation.weights_3_);
    weights_buf_ = std::move(operation.weights_buf_);
    std::swap(weights_are_buffer_, operation.weights_are_buffer_);
    std::swap(kernel_size_, operation.kernel_size_);
    std::swap(stride_, operation.stride_);
    std::swap(padding_, operation.padding_);
    std::swap(block_size_, operation.block_size_);
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status ConvolutionTransposed3D::Compile(
    const CreationContext& creation_context) {
  const auto code = GenerateConvolutionTransposed3DCode(
      definition_, biases_, *creation_context.device, weights_are_buffer_,
      block_size_, linked_operations_);

  std::vector<CompilerOptions> options;
  if (creation_context.device->IsPowerVR() && block_size_.y != 1) {
    bool is_texture3d = definition_.src_tensors[0].storage_type ==
                        TensorStorageType::TEXTURE_3D;
    bool is_texture_array = definition_.src_tensors[0].storage_type ==
                            TensorStorageType::TEXTURE_ARRAY;
    if (is_texture3d || is_texture_array) {
      options.push_back(CompilerOptions::CL_OPT_DISABLE);
    }
  }
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", options, *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status ConvolutionTransposed3D::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  if (weights_are_buffer_) {
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_buf_.GetMemoryPtr()));
  } else {
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_0_.GetMemoryPtr()));
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_1_.GetMemoryPtr()));
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_2_.GetMemoryPtr()));
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_3_.GetMemoryPtr()));
  }
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(
      int4(kernel_size_.x, kernel_size_.y, kernel_size_.z, 1)));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(int4(stride_.x, stride_.y, stride_.z, 1)));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(int4(padding_.x, padding_.y, padding_.z, 1)));
  if (definition_.IsBatchSupported()) {
    RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->Batch()));
  }
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(DivideRoundUp(dst_[0]->Slices(), block_size_.w)));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWHDS()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWHDS()));
  return absl::OkStatus();
}

int3 ConvolutionTransposed3D::GetGridSize() const {
  const int aligned_w = AlignByN(dst_[0]->Width(), stride_.x * block_size_.x);
  const int aligned_h = AlignByN(dst_[0]->Height(), stride_.y * block_size_.y);
  const int aligned_d = AlignByN(dst_[0]->Depth(), stride_.z * block_size_.z);
  const int grid_x = DivideRoundUp(aligned_w, block_size_.x) * dst_[0]->Batch();
  const int grid_y = DivideRoundUp(aligned_h, block_size_.y);
  const int grid_z = DivideRoundUp(dst_[0]->Slices(), block_size_.w) *
                     DivideRoundUp(aligned_d, block_size_.z);
  return int3(grid_x, grid_y, grid_z);
}

absl::Status ConvolutionTransposed3D::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroupConv(params, kernel_, GetGridSize(),
                              &work_group_size_);
}

absl::Status ConvolutionTransposed3D::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

absl::Status CreateConvolutionTransposed3D(
    const CreationContext& creation_context, const OperationDef& definition,
    const ConvolutionTransposed3DAttributes& attr,
    ConvolutionTransposed3D* result) {
  *result = ConvolutionTransposed3D(definition, attr, *creation_context.device);
  RETURN_IF_ERROR(
      result->UploadWeights(attr.weights, creation_context.context));
  LinearStorageCreateInfo create_info;
  create_info.storage_type =
      DeduceLinearStorageType(definition.GetPrimaryStorageType());
  create_info.data_type = definition.GetDataType();
  create_info.name = "biases";
  create_info.aligned_size = attr.weights.shape.o;
  RETURN_IF_ERROR(CreateLinearStorage(
      create_info, attr.bias, creation_context.context, &result->biases_));
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
