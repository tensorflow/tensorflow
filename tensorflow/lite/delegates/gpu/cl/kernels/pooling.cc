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

std::string GetAveragePoolingKernelCode(const OperationDef& op_def,
                                        bool stride_correction,
                                        const CLDevice& device,
                                        Arguments* args) {
  auto src_desc = absl::make_unique<TensorDescriptor>(op_def.src_tensors[0]);
  src_desc->SetTextureAddressMode(GetFastestZeroMode(device));
  if (op_def.IsBatchSupported()) {
    src_desc->SetStateVar("BatchedWidth", "true");
  }
  args->AddObjectRef("src_tensor", AccessType::READ, std::move(src_desc));
  auto dst_desc = absl::make_unique<TensorDescriptor>(op_def.dst_tensors[0]);
  if (op_def.IsBatchSupported()) {
    dst_desc->SetStateVar("BatchedWidth", "true");
  }
  args->AddObjectRef("dst_tensor", AccessType::WRITE, std::move(dst_desc));
  if (op_def.dst_tensors[0].HasAxis(Axis::WIDTH)) {
    args->AddInt("kernel_size_x");
    args->AddInt("padding_x");
    args->AddInt("stride_x");
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::HEIGHT)) {
    args->AddInt("kernel_size_y");
    args->AddInt("padding_y");
    args->AddInt("stride_y");
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    args->AddInt("kernel_size_z");
    args->AddInt("padding_z");
    args->AddInt("stride_z");
  }

  std::map<Axis, std::string> axis_to_src_coord = {
      {Axis::WIDTH, "x_c"},  {Axis::HEIGHT, "y_c"}, {Axis::DEPTH, "d_c"},
      {Axis::CHANNELS, "Z"}, {Axis::BATCH, "B"},
  };

  std::map<Axis, std::string> axis_to_dst_coord = {
      {Axis::WIDTH, "X"},    {Axis::HEIGHT, "Y"}, {Axis::DEPTH, "D"},
      {Axis::CHANNELS, "Z"}, {Axis::BATCH, "B"},
  };

  std::vector<std::string> src_coords;
  std::vector<std::string> dst_coords;
  for (auto axis : {Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH, Axis::CHANNELS}) {
    if (op_def.dst_tensors[0].HasAxis(axis)) {
      dst_coords.push_back(axis_to_dst_coord[axis]);
    }
    if (op_def.src_tensors[0].HasAxis(axis)) {
      src_coords.push_back(axis_to_src_coord[axis]);
    }
  }
  std::string src_coord = src_coords[0];
  for (int i = 1; i < src_coords.size(); ++i) {
    src_coord += ", " + src_coords[i];
  }
  std::string dst_coord = dst_coords[0];
  for (int i = 1; i < dst_coords.size(); ++i) {
    dst_coord += ", " + dst_coords[i];
  }

  const bool manual_clamp =
      op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER ||
      op_def.src_tensors[0].storage_type == TensorStorageType::IMAGE_BUFFER;

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  int X = get_global_id(0);\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = get_global_id(1);\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int D = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = get_global_id(1);\n";
  }
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  float4 r = (float4)(0.0f);\n";
  c += "  float window_size = 0.0;\n";
  if (stride_correction) {
    c += "  int xs = " +
         GetXStrideCorrected("X", "args.src_tensor.Batch()", "args.stride_x",
                             "args.padding_x") +
         ";\n";
  } else {
    c += "  int xs = X * args.stride_x + args.padding_x;\n";
  }
  c += "  int ys = Y * args.stride_y + args.padding_y;\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int ds = D * args.stride_z + args.padding_z;\n";
    c += "  for (int kz = 0; kz < args.kernel_size_z; ++kz) {\n";
    c += "    int d_c = ds + kz;\n";
    c += "    if (d_c < 0 || d_c >= args.src_tensor.Depth()) continue;\n";
  }
  c += "  for (int ky = 0; ky < args.kernel_size_y; ++ky) {\n";
  c += "    int y_c = ys + ky;\n";
  c += "    bool outside_y = y_c < 0 || y_c >= args.src_tensor.Height();\n";
  c += "    for (int kx = 0; kx < args.kernel_size_x; ++kx) {\n";
  if (op_def.IsBatchSupported()) {
    c += "      int x_c = xs + kx * args.src_tensor.Batch();\n";
  } else {
    c += "      int x_c = xs + kx;\n";
  }
  c += "      bool outside = outside_y || x_c < 0 || x_c >= "
       "args.src_tensor.Width();\n";
  if (manual_clamp) {
    c += "     r += !outside ? args.src_tensor.Read<float>(" + src_coord +
         ") : "
         "(float4)(0.0f);\n";
  } else {
    c += "      r += args.src_tensor.Read<float>(" + src_coord + ");\n";
  }
  c += "        window_size += !outside ? 1.0 : 0.0;\n";
  c += "    }\n";
  c += "  }\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  }  // Depth\n";
  }
  // If window_size==0, window covered nothing. This situation is a sign of
  // incorrectly constructed operation. NaNs are expected as output.
  c += "  FLT4 result = TO_FLT4(r / window_size);\n";
  c += "  args.dst_tensor.Write(result, " + dst_coord + ");\n";
  c += "}\n";

  return c;
}

std::string GetMaxPoolingKernelCode(const OperationDef& op_def,
                                    bool stride_correction, bool output_indices,
                                    Arguments* args) {
  auto src_desc = absl::make_unique<TensorDescriptor>(op_def.src_tensors[0]);
  if (op_def.IsBatchSupported()) {
    src_desc->SetStateVar("BatchedWidth", "true");
  }
  args->AddObjectRef("src_tensor", AccessType::READ, std::move(src_desc));
  auto dst_desc = absl::make_unique<TensorDescriptor>(op_def.dst_tensors[0]);
  if (op_def.IsBatchSupported()) {
    dst_desc->SetStateVar("BatchedWidth", "true");
  }
  args->AddObjectRef("dst_tensor", AccessType::WRITE, std::move(dst_desc));
  if (output_indices) {
    auto dst_ind_desc =
        absl::make_unique<TensorDescriptor>(op_def.dst_tensors[1]);
    if (op_def.IsBatchSupported()) {
      dst_ind_desc->SetStateVar("BatchedWidth", "true");
    }
    args->AddObjectRef("dst_indices", AccessType::WRITE,
                       std::move(dst_ind_desc));
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::WIDTH)) {
    args->AddInt("kernel_size_x");
    args->AddInt("padding_x");
    args->AddInt("stride_x");
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::HEIGHT)) {
    args->AddInt("kernel_size_y");
    args->AddInt("padding_y");
    args->AddInt("stride_y");
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    args->AddInt("kernel_size_z");
    args->AddInt("padding_z");
    args->AddInt("stride_z");
  }

  std::map<Axis, std::string> axis_to_src_coord = {
      {Axis::WIDTH, "x_c"},  {Axis::HEIGHT, "y_c"}, {Axis::DEPTH, "d_c"},
      {Axis::CHANNELS, "Z"}, {Axis::BATCH, "B"},
  };

  std::map<Axis, std::string> axis_to_dst_coord = {
      {Axis::WIDTH, "X"},    {Axis::HEIGHT, "Y"}, {Axis::DEPTH, "D"},
      {Axis::CHANNELS, "Z"}, {Axis::BATCH, "B"},
  };

  std::vector<std::string> src_coords;
  std::vector<std::string> dst_coords;
  for (auto axis : {Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH, Axis::CHANNELS}) {
    if (op_def.dst_tensors[0].HasAxis(axis)) {
      dst_coords.push_back(axis_to_dst_coord[axis]);
    }
    if (op_def.src_tensors[0].HasAxis(axis)) {
      src_coords.push_back(axis_to_src_coord[axis]);
    }
  }
  std::string src_coord = src_coords[0];
  for (int i = 1; i < src_coords.size(); ++i) {
    src_coord += ", " + src_coords[i];
  }
  std::string dst_coord = dst_coords[0];
  for (int i = 1; i < dst_coords.size(); ++i) {
    dst_coord += ", " + dst_coords[i];
  }

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  int X = get_global_id(0);\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = get_global_id(1);\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int D = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = get_global_id(1);\n";
  }
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT4 maximum = (FLT4)(-10000.0f);\n";
  if (output_indices) {
    c += "  FLT4 indexes = (FLT4)(0.0f);\n";
  }
  if (stride_correction) {
    c += "  int xs = " +
         GetXStrideCorrected("X", "args.src_tensor.Batch()", "args.stride_x",
                             "args.padding_x") +
         ";\n";
  } else {
    c += "  int xs = X * args.stride_x + args.padding_x;\n";
  }
  c += "  int ys = Y * args.stride_y + args.padding_y;\n";
  c += "  for (int ky = 0; ky < args.kernel_size_y; ++ky) {\n";
  c += "    int y_c = ys + ky;\n";
  c += "    if (y_c < 0 || y_c >= args.src_tensor.Height()) continue;\n";
  c += "    for (int kx = 0; kx < args.kernel_size_x; ++kx) {\n";
  if (op_def.IsBatchSupported()) {
    c += "      int x_c = xs + kx * args.src_tensor.Batch();\n";
  } else {
    c += "      int x_c = xs + kx;\n";
  }
  c += "      if (x_c < 0 || x_c >= args.src_tensor.Width()) continue;\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "    int ds = D * args.stride_z + args.padding_z;\n";
    c += "    for (int kz = 0; kz < args.kernel_size_z; ++kz) {\n";
    c += "    int d_c = ds + kz;\n";
    c += "      if (d_c < 0 || d_c >= args.src_tensor.Depth()) continue;\n";
  }
  c += "      FLT4 src = args.src_tensor.Read(" + src_coord + ");\n";
  if (output_indices) {
    if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
      c += "      FLT index_counter = (FLT)((ky * args.kernel_size_x + kx) * "
           "args.kernel_size_z + kz) + (FLT)(0.1f);\n";
    } else {
      c += "      FLT index_counter = (FLT)(ky * args.kernel_size_x + kx) + "
           "(FLT)(0.1f);\n";
    }
    c += "      if (src.x > maximum.x) {\n";
    c += "        indexes.x = index_counter;\n";
    c += "        maximum.x = src.x;\n";
    c += "      }\n";
    c += "      if (src.y > maximum.y) {\n";
    c += "        indexes.y = index_counter;\n";
    c += "        maximum.y = src.y;\n";
    c += "      }\n";
    c += "      if (src.z > maximum.z) {\n";
    c += "        indexes.z = index_counter;\n";
    c += "        maximum.z = src.z;\n";
    c += "      }\n";
    c += "      if (src.w > maximum.w) {\n";
    c += "        indexes.w = index_counter;\n";
    c += "        maximum.w = src.w;\n";
    c += "      }\n";
  } else {
    c += "      maximum = max(src, maximum);\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "    }  // Depth\n";
  }
  c += "    }\n";
  c += "  }\n";
  c += "  args.dst_tensor.Write(maximum, " + dst_coord + ");\n";
  if (output_indices) {
    c += "  args.dst_indices.Write(indexes, " + dst_coord + ");\n";
  }
  c += "}\n";

  return c;
}
}  // namespace

Pooling::Pooling(const OperationDef& definition,
                 const Pooling2DAttributes& attr)
    : GPUOperation(definition),
      stride_(attr.strides.w, attr.strides.h, 0, 0),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h, 0, 0),
      kernel_size_(attr.kernel.w, attr.kernel.h, 0, 0),
      type_(attr.type),
      output_indices_(attr.output_indices) {}

Pooling::Pooling(const OperationDef& definition,
                 const Pooling3DAttributes& attr)
    : GPUOperation(definition),
      stride_(attr.strides.w, attr.strides.h, attr.strides.d, 0),
      padding_(-attr.padding.prepended.w, -attr.padding.prepended.h,
               -attr.padding.prepended.d, 0),
      kernel_size_(attr.kernel.w, attr.kernel.h, attr.kernel.d, 0),
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
                                         *creation_context.device, &args_);
      break;
    case PoolingType::MAX:
      code = GetMaxPoolingKernelCode(definition_, stride_correction,
                                     output_indices_, &args_);
      break;
    default:
      return absl::InvalidArgumentError(
          "You should create another kernel with this params");
      break;
  }
  std::string element_wise_code;
  RETURN_IF_ERROR(
      MergeOperations(linked_operations_, &args_, &element_wise_code));
  RETURN_IF_ERROR(args_.TransformToCLCode(creation_context.device->GetInfo(),
                                          {{"dst_tensor", element_wise_code}},
                                          &code));
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status Pooling::BindArguments() {
  RETURN_IF_ERROR(args_.SetObjectRef("src_tensor", src_[0]));
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  if (definition_.dst_tensors[0].HasAxis(Axis::WIDTH)) {
    RETURN_IF_ERROR(args_.SetInt("stride_x", stride_.x));
    RETURN_IF_ERROR(args_.SetInt("padding_x", padding_.x * src_[0]->Batch()));
    RETURN_IF_ERROR(args_.SetInt("kernel_size_x", kernel_size_.x));
  }
  if (definition_.dst_tensors[0].HasAxis(Axis::HEIGHT)) {
    RETURN_IF_ERROR(args_.SetInt("stride_y", stride_.y));
    RETURN_IF_ERROR(args_.SetInt("padding_y", padding_.y));
    RETURN_IF_ERROR(args_.SetInt("kernel_size_y", kernel_size_.y));
  }
  if (definition_.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    RETURN_IF_ERROR(args_.SetInt("stride_z", stride_.z));
    RETURN_IF_ERROR(args_.SetInt("padding_z", padding_.z));
    RETURN_IF_ERROR(args_.SetInt("kernel_size_z", kernel_size_.z));
  }
  if (output_indices_) {
    RETURN_IF_ERROR(args_.SetObjectRef("dst_indices", dst_[1]));
  }
  RETURN_IF_ERROR(SetArguments(linked_operations_, &args_));
  return args_.Bind(kernel_.kernel());
}

int3 Pooling::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height() * dst_[0]->Depth();
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

Pooling CreatePooling(const OperationDef& definition,
                      const Pooling3DAttributes& attr) {
  return Pooling(definition, attr);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
