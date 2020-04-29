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

#include "tensorflow/lite/delegates/gpu/cl/kernels/pooling.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetAveragePoolingKernelCode(
    const OperationDef& op_def, bool stride_correction, const CLDevice& device,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor(
      "src_data", WHSPoint{"src_size.x", "src_size.y", "src_size.z"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data", WHSPoint{"dst_size.x", "dst_size.y", "dst_size.z"},
      op_def.dst_tensors[0]);

  const auto address_mode = GetFastestZeroMode(device);

  std::string c = GetCommonDefines(op_def.precision);

  const bool manual_clamp =
      op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER ||
      op_def.src_tensors[0].storage_type == TensorStorageType::IMAGE_BUFFER;

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size,             \n";
  c += "    int2 kernel_size,          \n";
  c += "    int2 padding,              \n";
  c += "    int2 stride                \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;\n";
  c += "  float4 r = (float4)(0.0f);\n";
  c += "  float window_size = 0.0;\n";
  if (stride_correction) {
    c += "  int xs = " +
         GetXStrideCorrected("X", "src_size.w", "stride.x", "padding.x") +
         ";\n";
  } else {
    c += "  int xs = X * stride.x + padding.x;\n";
  }
  c += "  int ys = Y * stride.y + padding.y;\n";
  c += "  for (int ky = 0; ky < kernel_size.y; ++ky) {\n";
  c += "    int y_c = ys + ky;\n";
  c += "    bool outside_y = y_c < 0 || y_c >= src_size.y;\n";
  c += "    for (int kx = 0; kx < kernel_size.x; ++kx) {\n";
  if (op_def.IsBatchSupported()) {
    c += "      int x_c = xs + kx * src_size.w;\n";
  } else {
    c += "      int x_c = xs + kx;\n";
  }
  c += "      bool outside = outside_y || x_c < 0 || x_c >= src_size.x;\n";
  if (manual_clamp) {
    c += "     r += !outside ? " +
         src_tensor.ReadAsFloatWHS("x_c", "y_c", "Z") + " : (float4)(0.0f);\n";
  } else {
    c += "      r += " +
         src_tensor.ReadAsFloatWHS("x_c", "y_c", "Z", address_mode) + ";\n";
  }
  c += "        window_size += !outside ? 1.0 : 0.0;\n";
  c += "    }\n";
  c += "  }\n";
  // If window_size==0, window covered nothing. This situation is a sign of
  // incorrectly constructed operation. NaNs are expected as output.
  c += "  FLT4 result = TO_FLT4(r / window_size);\n";
  const LinkingContext context{"result", "X", "Y", "Z"};
  c += PostProcess(linked_operations, context);
  c += "  " + dst_tensor.WriteWHS("result", "X", "Y", "Z");
  c += "}\n";

  return c;
}

std::string GetAveragePooling3DKernelCode(
    const OperationDef& op_def, bool stride_correction, const CLDevice& device,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor(
      "src_data",
      WHDSPoint{"src_size.x", "src_size.y", "src_size.z", "src_size.w"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data",
      WHDSPoint{"dst_size.x", "dst_size.y", "dst_size.z", "dst_size.w"},
      op_def.dst_tensors[0]);

  const auto address_mode = GetFastestZeroMode(device);

  std::string c = GetCommonDefines(op_def.precision);

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size,             \n";
  if (op_def.IsBatchSupported()) {
    c += "    int batch_size,          \n";
  }
  c += "    int4 kernel_size,          \n";
  c += "    int4 padding,              \n";
  c += "    int4 stride                \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int linear_id_z = get_global_id(2);\n";
  c += "  int S = linear_id_z % dst_size.w;\n";
  c += "  int Z = linear_id_z / dst_size.w;\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;\n";
  c += "  float4 r = (float4)(0.0f);\n";
  c += "  float window_size = 0.0;\n";
  if (stride_correction) {
    c += "  int xs = " +
         GetXStrideCorrected("X", "batch_size", "stride.x", "padding.x") +
         ";\n";
  } else {
    c += "  int xs = X * stride.x + padding.x;\n";
  }
  c += "  int ys = Y * stride.y + padding.y;\n";
  c += "  int zs = Z * stride.z + padding.z;\n";
  c += "  for (int kz = 0; kz < kernel_size.z; ++kz) {\n";
  c += "    int z_c = zs + kz;\n";
  c += "    if (z_c < 0 || z_c >= src_size.z) continue;\n";
  c += "    for (int ky = 0; ky < kernel_size.y; ++ky) {\n";
  c += "      int y_c = ys + ky;\n";
  c += "      if (y_c < 0 || y_c >= src_size.y) continue;\n";
  c += "      for (int kx = 0; kx < kernel_size.x; ++kx) {\n";
  if (op_def.IsBatchSupported()) {
    c += "        int x_c = xs + kx * batch_size;\n";
  } else {
    c += "        int x_c = xs + kx;\n";
  }
  c += "        if(x_c < 0 || x_c >= src_size.x) continue;\n";
  c += "        r += " +
       src_tensor.ReadAsFloatWHDS("x_c", "y_c", "z_c", "S", address_mode) +
       ";\n";
  c += "        window_size += 1.0;\n";
  c += "      }\n";
  c += "    }\n";
  c += "  }\n";
  // If window_size==0, window covered nothing. This situation is a sign of
  // incorrectly constructed operation. NaNs are expected as output.
  c += "  FLT4 result = TO_FLT4(r / window_size);\n";
  const LinkingContext context{"result", "X", "Y", "Z"};
  c += PostProcess(linked_operations, context);
  c += "  " + dst_tensor.WriteWHDS("result", "X", "Y", "Z", "S");
  c += "}\n";

  return c;
}

std::string GetMaxPoolingKernelCode(
    const OperationDef& op_def, bool stride_correction,
    const std::vector<ElementwiseOperation*>& linked_operations,
    bool output_indices) {
  TensorCodeGenerator src_tensor(
      "src_data", WHSPoint{"src_size.x", "src_size.y", "src_size.z"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data", WHSPoint{"dst_size.x", "dst_size.y", "dst_size.z"},
      op_def.dst_tensors[0]);
  const auto dst_ind_def =
      output_indices ? op_def.dst_tensors[1] : op_def.dst_tensors[0];
  TensorCodeGenerator indices_tensor(
      "dst_indices", WHSPoint{"dst_size.x", "dst_size.y", "dst_size.z"},
      dst_ind_def);

  std::string c = GetCommonDefines(op_def.precision);

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  if (output_indices) {
    c += indices_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  }
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size,             \n";
  c += "    int2 kernel_size,          \n";
  c += "    int2 padding,              \n";
  c += "    int2 stride                \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c +=
      "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return; \n";
  c += "  FLT4 maximum = (FLT4)(-10000.0f);\n";
  if (output_indices) {
    c += "  FLT4 indexes = (FLT4)(0.0f);\n";
    c += "  FLT index_counter = (FLT)(0.1f);\n";
  }
  if (stride_correction) {
    c += "  int xs = " +
         GetXStrideCorrected("X", "src_size.w", "stride.x", "padding.x") +
         ";\n";
  } else {
    c += "  int xs = X * stride.x + padding.x;\n";
  }
  c += "  int ys = Y * stride.y + padding.y;\n";
  c += "  for (int ky = 0; ky < kernel_size.y; ++ky) {\n";
  c += "    int y_c = ys + ky;\n";
  c += "    bool outside_y = y_c < 0 || y_c >= src_size.y;\n";
  c += "    for (int kx = 0; kx < kernel_size.x; ++kx) {\n";
  if (op_def.IsBatchSupported()) {
    c += "      int x_c = xs + kx * src_size.w;\n";
  } else {
    c += "      int x_c = xs + kx;\n";
  }
  c += "      bool outside_x = x_c < 0 || x_c >= src_size.x;\n";
  c += "      if (!outside_x && !outside_y) {\n";
  c += "        FLT4 src = " + src_tensor.ReadWHS("x_c", "y_c", "Z") + ";\n";
  if (output_indices) {
    c += "        if (src.x > maximum.x) {\n";
    c += "          indexes.x = index_counter;\n";
    c += "          maximum.x = src.x;\n";
    c += "        }\n";
    c += "        if (src.y > maximum.y) {\n";
    c += "          indexes.y = index_counter;\n";
    c += "          maximum.y = src.y;\n";
    c += "        }\n";
    c += "        if (src.z > maximum.z) {\n";
    c += "          indexes.z = index_counter;\n";
    c += "          maximum.z = src.z;\n";
    c += "        }\n";
    c += "        if (src.w > maximum.w) {\n";
    c += "          indexes.w = index_counter;\n";
    c += "          maximum.w = src.w;\n";
    c += "        }\n";
    c += "        index_counter += (FLT)(1.0f);\n";
  } else {
    c += "        maximum = max(src, maximum);\n";
  }
  c += "      }\n";
  c += "    }\n";
  c += "  }\n";
  const LinkingContext context{"maximum", "X", "Y", "Z"};
  c += PostProcess(linked_operations, context);
  c += "  " + dst_tensor.WriteWHS("maximum", "X", "Y", "Z");
  if (output_indices) {
    c += "  " + indices_tensor.WriteWHS("indexes", "X", "Y", "Z");
  }
  c += "}\n";

  return c;
}

std::string GetMaxPooling3DKernelCode(
    const OperationDef& op_def, bool stride_correction,
    const std::vector<ElementwiseOperation*>& linked_operations,
    bool output_indices) {
  TensorCodeGenerator src_tensor(
      "src_data",
      WHDSPoint{"src_size.x", "src_size.y", "src_size.z", "src_size.w"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data",
      WHDSPoint{"dst_size.x", "dst_size.y", "dst_size.z", "dst_size.w"},
      op_def.dst_tensors[0]);
  const auto dst_ind_def =
      output_indices ? op_def.dst_tensors[1] : op_def.dst_tensors[0];
  TensorCodeGenerator indices_tensor(
      "dst_indices",
      WHDSPoint{"dst_size.x", "dst_size.y", "dst_size.z", "dst_size.w"},
      dst_ind_def);

  std::string c = GetCommonDefines(op_def.precision);

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  if (output_indices) {
    c += indices_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  }
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size,             \n";
  if (op_def.IsBatchSupported()) {
    c += "    int batch_size,          \n";
  }
  c += "    int4 kernel_size,          \n";
  c += "    int4 padding,              \n";
  c += "    int4 stride                \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int linear_id_z = get_global_id(2);\n";
  c += "  int S = linear_id_z % dst_size.w;\n";
  c += "  int Z = linear_id_z / dst_size.w;\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;\n";
  c += "  FLT4 maximum = (FLT4)(-10000.0f);\n";
  if (output_indices) {
    c += "  FLT4 indexes = (FLT4)(0.0f);\n";
  }
  if (stride_correction) {
    c += "  int xs = " +
         GetXStrideCorrected("X", "batch_size", "stride.x", "padding.x") +
         ";\n";
  } else {
    c += "  int xs = X * stride.x + padding.x;\n";
  }
  c += "  int ys = Y * stride.y + padding.y;\n";
  c += "  int zs = Z * stride.z + padding.z;\n";
  c += "  for (int ky = 0; ky < kernel_size.y; ++ky) {\n";
  c += "    int y_c = ys + ky;\n";
  c += "    if (y_c < 0 || y_c >= src_size.y) continue;\n";
  c += "    for (int kx = 0; kx < kernel_size.x; ++kx) {\n";
  if (op_def.IsBatchSupported()) {
    c += "      int x_c = xs + kx * batch_size;\n";
  } else {
    c += "      int x_c = xs + kx;\n";
  }
  c += "      if (x_c < 0 || x_c >= src_size.x) continue;\n";
  c += "      for (int kz = 0; kz < kernel_size.z; ++kz) {\n";
  c += "        int z_c = zs + kz;\n";
  c += "        if (z_c < 0 || z_c >= src_size.z) continue;\n";
  c += "        FLT4 src = " + src_tensor.ReadWHDS("x_c", "y_c", "z_c", "S") +
       ";\n";
  if (output_indices) {
    c += "        FLT index_counter = (FLT)((ky * kernel_size.x + kx) * "
         "kernel_size.z + kz) + (FLT)(0.1f);\n";
    c += "        if (src.x > maximum.x) {\n";
    c += "          indexes.x = index_counter;\n";
    c += "          maximum.x = src.x;\n";
    c += "        }\n";
    c += "        if (src.y > maximum.y) {\n";
    c += "          indexes.y = index_counter;\n";
    c += "          maximum.y = src.y;\n";
    c += "        }\n";
    c += "        if (src.z > maximum.z) {\n";
    c += "          indexes.z = index_counter;\n";
    c += "          maximum.z = src.z;\n";
    c += "        }\n";
    c += "        if (src.w > maximum.w) {\n";
    c += "          indexes.w = index_counter;\n";
    c += "          maximum.w = src.w;\n";
    c += "        }\n";
  } else {
    c += "        maximum = max(src, maximum);\n";
  }
  c += "      };\n";
  c += "    }\n";
  c += "  }\n";
  const LinkingContext context{"maximum", "X", "Y", "Z"};
  c += PostProcess(linked_operations, context);
  c += "  " + dst_tensor.WriteWHDS("maximum", "X", "Y", "Z", "S");
  if (output_indices) {
    c += "  " + indices_tensor.WriteWHDS("indexes", "X", "Y", "Z", "S");
  }
  c += "}\n";
  return c;
}

}  // namespace

Pooling::Pooling(const OperationDef& definition,
                 const Pooling2DAttributes& attr)
    : GPUOperation(definition),
      stride_(attr.strides.w, attr.strides.h),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h),
      kernel_size_(attr.kernel.w, attr.kernel.h),
      type_(attr.type),
      output_indices_(attr.output_indices) {}

Pooling::Pooling(Pooling&& kernel)
    : GPUOperation(std::move(kernel)),
      stride_(kernel.stride_),
      padding_(kernel.padding_),
      kernel_size_(kernel.kernel_size_),
      type_(kernel.type_),
      output_indices_(kernel.output_indices_),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

Pooling& Pooling::operator=(Pooling&& kernel) {
  if (this != &kernel) {
    std::swap(stride_, kernel.stride_);
    std::swap(padding_, kernel.padding_);
    std::swap(kernel_size_, kernel.kernel_size_);
    std::swap(type_, kernel.type_);
    std::swap(output_indices_, kernel.output_indices_);
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

absl::Status Pooling::Compile(const CreationContext& creation_context) {
  std::string code;
  const bool stride_correction =
      definition_.IsBatchSupported() && stride_.x != 1;
  switch (type_) {
    case PoolingType::AVERAGE:
      code = GetAveragePoolingKernelCode(definition_, stride_correction,
                                         *creation_context.device,
                                         linked_operations_);
      break;
    case PoolingType::MAX:
      code = GetMaxPoolingKernelCode(definition_, stride_correction,
                                     linked_operations_, output_indices_);
      break;
    default:
      return absl::InvalidArgumentError(
          "You should create another kernel with this params");
      break;
  }
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status Pooling::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  if (output_indices_) {
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[1]->GetMemoryPtrForWriting()));
  }
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWBatchedHSB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWBatchedHSB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(kernel_size_));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(int2(padding_.x * src_[0]->Batch(), padding_.y)));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(stride_));

  return absl::OkStatus();
}

int3 Pooling::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status Pooling::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status Pooling::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Pooling CreatePooling(const OperationDef& definition,
                      const Pooling2DAttributes& attr) {
  return Pooling(definition, attr);
}

Pooling3D::Pooling3D(const OperationDef& definition,
                     const Pooling3DAttributes& attr)
    : GPUOperation(definition),
      stride_(attr.strides.w, attr.strides.h, attr.strides.d),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h,
               -attr.padding.prepended.d),
      kernel_size_(attr.kernel.w, attr.kernel.h, attr.kernel.d),
      type_(attr.type),
      output_indices_(attr.output_indices) {}

Pooling3D::Pooling3D(Pooling3D&& kernel)
    : GPUOperation(std::move(kernel)),
      stride_(kernel.stride_),
      padding_(kernel.padding_),
      kernel_size_(kernel.kernel_size_),
      type_(kernel.type_),
      output_indices_(kernel.output_indices_),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

Pooling3D& Pooling3D::operator=(Pooling3D&& kernel) {
  if (this != &kernel) {
    std::swap(stride_, kernel.stride_);
    std::swap(padding_, kernel.padding_);
    std::swap(kernel_size_, kernel.kernel_size_);
    std::swap(type_, kernel.type_);
    std::swap(output_indices_, kernel.output_indices_);
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

absl::Status Pooling3D::Compile(const CreationContext& creation_context) {
  std::string code;
  const bool stride_correction =
      definition_.IsBatchSupported() && stride_.x != 1;
  switch (type_) {
    case PoolingType::AVERAGE:
      code = GetAveragePooling3DKernelCode(definition_, stride_correction,
                                           *creation_context.device,
                                           linked_operations_);
      break;
    case PoolingType::MAX:
      code = GetMaxPooling3DKernelCode(definition_, stride_correction,
                                       linked_operations_, output_indices_);
      break;
    default:
      return absl::InvalidArgumentError(
          "You should create another kernel with this params");
      break;
  }
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status Pooling3D::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  if (output_indices_) {
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[1]->GetMemoryPtrForWriting()));
  }
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWBatchedHDS()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWBatchedHDS()));
  if (definition_.IsBatchSupported()) {
    RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->Batch()));
  }
  RETURN_IF_ERROR(kernel_.SetBytesAuto(
      int4(kernel_size_.x, kernel_size_.y, kernel_size_.z, 1)));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(
      int4(padding_.x * src_[0]->Batch(), padding_.y, padding_.z, 1)));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(int4(stride_.x, stride_.y, stride_.z, 1)));

  return absl::OkStatus();
}

int3 Pooling3D::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices() * dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status Pooling3D::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status Pooling3D::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Pooling3D CreatePooling3D(const OperationDef& definition,
                          const Pooling3DAttributes& attr) {
  return Pooling3D(definition, attr);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
