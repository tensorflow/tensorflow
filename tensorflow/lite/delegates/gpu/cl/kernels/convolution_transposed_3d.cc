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

std::string GenerateConvolutionTransposed3DCode(const OperationDef& op_def,
                                                const CLDevice& device,
                                                bool weights_are_buffer,
                                                const int4& block_size,
                                                Arguments* args) {
  auto src_desc = absl::make_unique<TensorDescriptor>(op_def.src_tensors[0]);
  src_desc->SetTextureAddressMode(GetFastestZeroMode(device));
  args->AddObjectRef("src_tensor", AccessType::READ, std::move(src_desc));
  args->AddObjectRef(
      "dst_tensor", AccessType::WRITE,
      absl::make_unique<TensorDescriptor>(op_def.dst_tensors[0]));
  args->AddInt("stride_x");
  args->AddInt("stride_y");
  args->AddInt("stride_z");
  args->AddInt("padding_x");
  args->AddInt("padding_y");
  args->AddInt("padding_z");
  args->AddInt("kernel_size_x");
  args->AddInt("kernel_size_y");
  args->AddInt("kernel_size_z");
  args->AddInt("grid_size_s");

  const auto src_tensor_type = op_def.src_tensors[0].storage_type;
  bool image_buffer = src_tensor_type == TensorStorageType::IMAGE_BUFFER;
  bool manual_clamp =
      image_buffer || src_tensor_type == TensorStorageType::BUFFER;

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
  c += "$0) {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int dst_x = (linear_id / args.dst_tensor.Batch());\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int dst_x = get_global_id(0);\n";
  }
  c += "  int rem_x = dst_x % args.stride_x;\n";
  c += "  int ceil_x = dst_x / args.stride_x;\n";
  c += "  dst_x = ceil_x * args.stride_x * " + std::to_string(block_size.x) +
       " + rem_x;\n";
  c += "  int dst_y = get_global_id(1);\n";
  c += "  int rem_y = dst_y % args.stride_y;\n";
  c += "  int ceil_y = dst_y / args.stride_y;\n";
  c += "  dst_y = ceil_y * args.stride_y * " + std::to_string(block_size.y) +
       " + rem_y;\n";
  c += "  int linear_id_z = get_global_id(2);\n";
  c += "  int S = (linear_id_z % args.grid_size_s) * " +
       std::to_string(block_size.w) + ";\n";
  c += "  int dst_z = linear_id_z / args.grid_size_s;\n";
  c += "  int rem_z = dst_z % args.stride_z;\n";
  c += "  int ceil_z = dst_z / args.stride_z;\n";
  c += "  dst_z = ceil_z * args.stride_z * " + std::to_string(block_size.z) +
       " + rem_z;\n";
  c += "  if (dst_x >= args.dst_tensor.Width() || dst_y >= "
       "args.dst_tensor.Height() || dst_z >= "
       "args.dst_tensor.Depth()) return;\n";
  if (weights_are_buffer) {
    c += "  int f_base = S * args.src_tensor.Slices() * args.kernel_size_x * "
         "args.kernel_size_y * "
         "args.kernel_size_z;\n";
  }
  for (int i = 0; i < block_size.x * block_size.y * block_size.z * block_size.w;
       ++i) {
    c += "  ACCUM_FLT4 r" + std::to_string(i) +
         " = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
  }
  c += "  int kernel_first_dst_x = dst_x + args.padding_x;\n";
  c += "  int kernel_first_dst_y = dst_y + args.padding_y;\n";
  c += "  int kernel_first_dst_z = dst_z + args.padding_z;\n";
  c += "  int kernel_last_dst_x = kernel_first_dst_x - args.kernel_size_x;\n";
  c += "  int kernel_last_dst_y = kernel_first_dst_y - args.kernel_size_y;\n";
  c += "  int kernel_last_dst_z = kernel_first_dst_z - args.kernel_size_z;\n";
  c += "  int offset_x = abs(args.padding_x);\n";
  c += "  int offset_x_strided = offset_x * args.stride_x;\n";
  c +=
      "  int src_x = (kernel_first_dst_x + offset_x_strided) / args.stride_x - "
      "offset_x;\n";
  c += "  int offset_y = abs(args.padding_y);\n";
  c += "  int offset_y_strided = offset_y * args.stride_y;\n";
  c +=
      "  int src_y = (kernel_first_dst_y + offset_y_strided) / args.stride_y - "
      "offset_y;\n";
  c += "  int offset_z = abs(args.padding_z);\n";
  c += "  int offset_z_strided = offset_z * args.stride_z;\n";
  c +=
      "  int src_z = (kernel_first_dst_z + offset_z_strided) / args.stride_z - "
      "offset_z;\n";
  c += "  int src_as_dst_z = src_z * args.stride_z;\n";
  c += "  for (;src_as_dst_z > kernel_last_dst_z; src_z -= 1, src_as_dst_z -= "
       "args.stride_z) {\n";
  for (int z = 0; z < block_size.z; ++z) {
    const std::string zindex = std::to_string(z);
    c += "    int sz" + zindex + " = src_z + " + zindex + ";\n";
    if (src_tensor_type != TensorStorageType::TEXTURE_3D) {
      c += "    bool in_z" + zindex + " = sz" + zindex + " >= 0 && sz" +
           zindex + " < args.src_tensor.Depth();\n";
    }
  }
  if (block_size.z == 1 && (src_tensor_type != TensorStorageType::TEXTURE_3D)) {
    c += "    if (!in_z0) continue;\n";
  }
  c += "    int kernel_z = kernel_first_dst_z - src_as_dst_z;\n";
  c += "    int src_as_dst_y = src_y * args.stride_y;\n";
  c += "    int src_y_copy = src_y;\n";
  c += "    for (;src_as_dst_y > kernel_last_dst_y; src_y_copy -= 1, "
       "src_as_dst_y -= "
       "args.stride_y) {\n";
  for (int y = 0; y < block_size.y; ++y) {
    const std::string yindex = std::to_string(y);
    c += "      int sy" + yindex + " = src_y_copy + " + yindex + ";\n";
    if (manual_clamp) {
      c += "      bool in_y" + yindex + " = sy" + yindex + " >= 0 && sy" +
           yindex + " < args.src_tensor.Height();\n";
      if (!image_buffer) {
        c += "      sy" + yindex + " = clamp(sy" + yindex +
             ", 0, args.src_tensor.Height() - 1);\n";
      }
    }
  }
  c += "      int kernel_y = kernel_first_dst_y - src_as_dst_y;\n";
  c += "      int src_as_dst_x = src_x * args.stride_x;\n";
  c += "      int src_x_copy = src_x;\n";
  c += "      for (;src_as_dst_x > kernel_last_dst_x; src_x_copy -= 1, "
       "src_as_dst_x "
       "-= args.stride_x) {\n";
  for (int x = 0; x < block_size.x; ++x) {
    const std::string xindex = std::to_string(x);
    c += "        int sx" + xindex + " = src_x_copy + " + xindex + ";\n";
    if (manual_clamp) {
      c += "        bool in_x" + xindex + " = sx" + xindex + " >= 0 && sx" +
           xindex + " < args.src_tensor.Width();\n";
      if (!image_buffer) {
        c += "        sx" + xindex + " = clamp(sx" + xindex +
             ", 0, args.src_tensor.Width() - 1);\n";
      }
    }
  }
  const std::string layer_offset = "args.src_tensor.SliceStride()";
  for (int z = 0; z < block_size.z; ++z) {
    const std::string zindex = std::to_string(z);
    for (int y = 0; y < block_size.y; ++y) {
      const std::string yindex = std::to_string(y);
      for (int x = 0; x < block_size.x; ++x) {
        const std::string xindex = std::to_string(x);
        const std::string id =
            std::to_string((z * block_size.y + y) * block_size.x + x);
        c += "        args.src_tensor.GetAddress(addr_" + id + ", sx" + xindex +
             ", sy" + yindex + ", sz" + zindex + ", 0);";
        if (image_buffer) {
          c += "        addr_" + id + " = select(-1, addr_" + id + ", (in_x" +
               xindex + " && in_y" + yindex + "));\n";
          c += absl::Substitute(
              "        int dz_$0 = select(0, $3, (in_x$1 && "
              "in_y$2));\n",
              id, x, y, layer_offset);
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
  c += "        int kernel_index =(kernel_z * args.kernel_size_y + kernel_y) * "
       "args.kernel_size_x + kernel_x;\n";
  if (weights_are_buffer) {
    c += "        int f_offset = f_base + kernel_index * "
         "args.src_tensor.Slices() * " +
         std::to_string(block_size.w) + ";\n";
  } else {
    c += "        int x_c = kernel_index * args.src_tensor.Slices();\n";
  }
  c += "        for (int s = 0; s < args.src_tensor.Slices(); ++s) {\n";
  for (int y = 0; y < block_size.y; ++y) {
    const std::string yindex = std::to_string(y);
    for (int x = 0; x < block_size.x; ++x) {
      const std::string xindex = std::to_string(x);
      const std::string id = std::to_string(y * block_size.x + x);
      if (image_buffer) {
        c += "          FLT4 src" + id + " = args.src_tensor.Read(addr_" + id +
             "); addr_" + id + " += dz_" + id + ";\n";
      } else if (manual_clamp) {
        c += "          FLT4 src" + id + " = args.src_tensor.Read(addr_" + id +
             ") * (FLT)(in_x" + xindex + " && in_y" + yindex + "); addr_" + id +
             " += dz;\n";
      } else {
        c += "          FLT4 src" + id + " = args.src_tensor.Read(sx" + xindex +
             ", sy" + yindex + ", sz0, s);\n";
      }
    }
  }
  if (weights_are_buffer) {
    c += "          __global FLT16* weights_cache = "
         "args.weights.GetPtr(f_offset);\n";
    c += "          f_offset += " + std::to_string(block_size.w) + ";\n";
  } else {
    for (int z = 0; z < block_size.w; ++z) {
      c += absl::Substitute(
          R"(          FLT4 f$1 = args.weights0.Read(S + $0, x_c);
          FLT4 f$2 = args.weights1.Read(S + $0, x_c);
          FLT4 f$3 = args.weights2.Read(S + $0, x_c);
          FLT4 f$4 = args.weights3.Read(S + $0, x_c);
)",
          z, z * 4 + 0, z * 4 + 1, z * 4 + 2, z * 4 + 3);
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
    c += "  if (S < args.dst_tensor.Slices()) {\n";
    c += "    FLT4 bias_val = args.biases.Read(S);\n";
    for (int z = 0; z < block_size.z; ++z) {
      for (int y = 0; y < block_size.y; ++y) {
        for (int x = 0; x < block_size.x; ++x) {
          const std::string id = std::to_string(
              ((s * block_size.z + z) * block_size.y + y) * block_size.x + x);
          c += "    {\n";
          c += "      int xc = dst_x + args.stride_x * " + std::to_string(x) +
               ";\n";
          c += "      int yc = dst_y + args.stride_y * " + std::to_string(y) +
               ";\n";
          c += "      int zc = dst_z + args.stride_z * " + std::to_string(z) +
               ";\n";
          c += "      if (xc < args.dst_tensor.Width() && yc < "
               "args.dst_tensor.Height() && zc < args.dst_tensor.Depth()) {\n";
          c += "        FLT4 res = TO_FLT4(r" + id + ") + bias_val;\n";
          c += "        args.dst_tensor.Write(res, xc, yc, zc, S)\n";
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
  std::string code = GenerateConvolutionTransposed3DCode(
      definition_, *creation_context.device, weights_are_buffer_, block_size_,
      &args_);
  std::string element_wise_code;
  RETURN_IF_ERROR(
      MergeOperations(linked_operations_, &args_, &element_wise_code));
  RETURN_IF_ERROR(args_.TransformToCLCode(creation_context.device->GetInfo(),
                                          {{"dst_tensor", element_wise_code}},
                                          &code));

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
  RETURN_IF_ERROR(args_.SetObjectRef("src_tensor", src_[0]));
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  RETURN_IF_ERROR(args_.SetInt("stride_x", stride_.x));
  RETURN_IF_ERROR(args_.SetInt("stride_y", stride_.y));
  RETURN_IF_ERROR(args_.SetInt("stride_z", stride_.z));
  RETURN_IF_ERROR(args_.SetInt("padding_x", padding_.x));
  RETURN_IF_ERROR(args_.SetInt("padding_y", padding_.y));
  RETURN_IF_ERROR(args_.SetInt("padding_z", padding_.z));
  RETURN_IF_ERROR(args_.SetInt("kernel_size_x", kernel_size_.x));
  RETURN_IF_ERROR(args_.SetInt("kernel_size_y", kernel_size_.y));
  RETURN_IF_ERROR(args_.SetInt("kernel_size_z", kernel_size_.z));
  RETURN_IF_ERROR(args_.SetInt(
      "grid_size_s", DivideRoundUp(dst_[0]->Slices(), block_size_.w)));
  RETURN_IF_ERROR(SetArguments(linked_operations_, &args_));
  return args_.Bind(kernel_.kernel());
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

  TensorLinearDescriptor desc;
  desc.storage_type =
      DeduceLinearStorageType(definition.GetPrimaryStorageType());
  desc.element_type = definition.GetDataType();

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
