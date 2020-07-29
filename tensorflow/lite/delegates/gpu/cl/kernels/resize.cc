/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/kernels/resize.h"

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetResizeCode(const OperationDef& op_def,
                          SamplingType sampling_type, bool half_pixel_centers,
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
  args->AddInt("border_x");
  args->AddInt("border_y");
  args->AddFloat("scale_factor_x");
  args->AddFloat("scale_factor_y");

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  if (linear_id >= args.dst_tensor.Width() || Y >= "
         "args.dst_tensor.Height() || Z >= args.dst_tensor.Slices()) return;\n";
  } else {
    c += "  int X = get_global_id(0);\n";
    c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
         "|| Z >= args.dst_tensor.Slices()) return;\n";
  }
  if (sampling_type == SamplingType::NEAREST) {
    c += "  int2 coord = (int2)(X * args.scale_factor_x, Y * "
         "args.scale_factor_y);\n";
    if (op_def.IsBatchSupported()) {
      c += "  coord.x = coord.x * args.src_tensor.Batch() + B;\n";
      c += "  X = X * args.src_tensor.Batch() + B;\n";
    }
    c += "  FLT4 r0 = args.src_tensor.Read(coord.x, coord.y, Z);\n";
  } else {
    if (half_pixel_centers) {
      c += "  float2 f_coords = ((float2)(X, Y) + 0.5f) * "
           "(float2)(args.scale_factor_x, args.scale_factor_y) - "
           "0.5f;\n";
    } else {
      c += "  float2 f_coords = (float2)(X, Y) * (float2)(args.scale_factor_x, "
           "args.scale_factor_y);\n";
    }
    c += "  float2 f_coords_floor = floor(f_coords);\n";
    c += "  int2 coords_floor = (int2)(f_coords_floor.x, f_coords_floor.y);\n";
    c += "  int4 st;\n";
    c += "  st.xy = max(coords_floor, (int2)(0, 0));\n";
    c += "  st.zw = min(coords_floor + (int2)(1, 1), (int2)(args.border_x, "
         "args.border_y));\n";
    c += "  float2 t = f_coords - f_coords_floor;\n";
    if (op_def.IsBatchSupported()) {
      c += "  st.x = st.x * args.src_tensor.Batch() + B;\n";
      c += "  st.z = st.z * args.src_tensor.Batch() + B;\n";
      c += "  X = X * args.src_tensor.Batch() + B;\n";
    }
    c += "  float4 src0 = args.src_tensor.Read<float>(st.x, st.y, Z);\n";
    c += "  float4 src1 = args.src_tensor.Read<float>(st.z, st.y, Z);\n";
    c += "  float4 src2 = args.src_tensor.Read<float>(st.x, st.w, Z);\n";
    c += "  float4 src3 = args.src_tensor.Read<float>(st.z, st.w, Z);\n";
    c += "  FLT4 r0 = TO_FLT4(mix(mix(src0, src1, t.x), mix(src2, src3, t.x), "
         "t.y));\n";
  }
  c += "  args.dst_tensor.Write(r0, X, Y, Z);\n";
  c += "}\n";
  return c;
}

std::string GetResize3DCode(const OperationDef& op_def,
                            SamplingType sampling_type, Arguments* args) {
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
  args->AddInt("border_x");
  args->AddInt("border_y");
  args->AddInt("border_z");
  args->AddFloat("scale_factor_x");
  args->AddFloat("scale_factor_y");
  args->AddFloat("scale_factor_z");

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int linear_id_z = get_global_id(2);\n";
  c += "  int S = linear_id_z % args.dst_tensor.Slices();\n";
  c += "  int Z = linear_id_z / args.dst_tensor.Slices();\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  if (linear_id >= args.dst_tensor.Width() || Y >= "
         "args.dst_tensor.Height() || Z >= args.dst_tensor.Depth()) return;\n";
  } else {
    c += "  int X = get_global_id(0);\n";
    c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
         "|| Z >= args.dst_tensor.Depth()) return;\n";
  }
  if (sampling_type == SamplingType::NEAREST) {
    c += "  int4 coord = (int4)(X * args.scale_factor_x, Y * "
         "args.scale_factor_y, Z * "
         "args.scale_factor_z, 0);\n";
    if (op_def.IsBatchSupported()) {
      c += "  coord.x = coord.x * args.src_tensor.Batch() + B;\n";
      c += "  X = X * args.src_tensor.Batch() + B;\n";
    }
    c += "  FLT4 r0 = args.src_tensor.Read(coord.x, coord.y, coord.z, S);\n";
  } else {
    c += "  float4 f_coords;\n";
    c += "  f_coords.x = (float)(X) * args.scale_factor_x;\n";
    c += "  f_coords.y = (float)(Y) * args.scale_factor_y;\n";
    c += "  f_coords.z = (float)(Z) * args.scale_factor_z;\n";
    c += "  int4 start = (int4)(f_coords.x, f_coords.y, f_coords.z, 0);\n";
    c += "  int4 end;\n";
    c += "  end.x = min(start.x + 1, args.border_x);\n";
    c += "  end.y = min(start.y + 1, args.border_y);\n";
    c += "  end.z = min(start.z + 1, args.border_z);\n";
    c += "  float4 t = f_coords - (float4)(start.x, start.y, start.z, 0.0f);\n";
    if (op_def.IsBatchSupported()) {
      c += "  start.x = start.x * args.src_tensor.Batch() + B;\n";
      c += "  end.x = end.x * args.src_tensor.Batch() + B;\n";
      c += "  X = X * args.src_tensor.Batch() + B;\n";
    }
    c += "  float4 src0 = args.src_tensor.Read<float>(start.x, start.y, "
         "start.z, S);\n";
    c += "  float4 src1 = args.src_tensor.Read<float>(end.x, start.y, start.z, "
         "S);\n";
    c += "  float4 src2 = args.src_tensor.Read<float>(start.x, end.y, start.z, "
         "S);\n";
    c += "  float4 src3 = args.src_tensor.Read<float>(end.x, end.y, start.z, "
         "S);\n";
    c += "  float4 src4 = args.src_tensor.Read<float>(start.x, start.y, end.z, "
         "S);\n";
    c += "  float4 src5 = args.src_tensor.Read<float>(end.x, start.y, end.z, "
         "S);\n";
    c += "  float4 src6 = args.src_tensor.Read<float>(start.x, end.y, end.z, "
         "S);\n";
    c += "  float4 src7 = args.src_tensor.Read<float>(end.x, end.y, end.z, "
         "S);\n";
    c +=
        "  float4 t0 = mix(mix(src0, src1, t.x), mix(src2, src3, t.x), t.y);\n";
    c +=
        "  float4 t1 = mix(mix(src4, src5, t.x), mix(src6, src7, t.x), t.y);\n";
    c += "  FLT4 r0 = TO_FLT4(mix(t0, t1, t.z));\n";
  }
  c += "  args.dst_tensor.Write(r0, X, Y, Z, S);\n";
  c += "}\n";
  return c;
}

}  // namespace

Resize::Resize(Resize&& operation)
    : GPUOperation(std::move(operation)),
      attr_(operation.attr_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

Resize& Resize::operator=(Resize&& operation) {
  if (this != &operation) {
    attr_ = operation.attr_;
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status Resize::Compile(const CreationContext& creation_context) {
  std::string code =
      GetResizeCode(definition_, attr_.type, attr_.half_pixel_centers, &args_);
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

absl::Status Resize::BindArguments() {
  RETURN_IF_ERROR(args_.SetObjectRef("src_tensor", src_[0]));
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  RETURN_IF_ERROR(args_.SetInt("border_x", src_[0]->Width() - 1));
  RETURN_IF_ERROR(args_.SetInt("border_y", src_[0]->Height() - 1));
  RETURN_IF_ERROR(args_.SetFloat(
      "scale_factor_x",
      CalculateResizeScale(src_[0]->Width(), dst_[0]->Width(), attr_)));
  RETURN_IF_ERROR(args_.SetFloat(
      "scale_factor_y",
      CalculateResizeScale(src_[0]->Height(), dst_[0]->Height(), attr_)));
  RETURN_IF_ERROR(SetArguments(linked_operations_, &args_));
  return args_.Bind(kernel_.kernel());
}

int3 Resize::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status Resize::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

absl::Status Resize::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Resize CreateResize(const OperationDef& definition,
                    const Resize2DAttributes& attr) {
  return Resize(definition, attr);
}

Resize3D::Resize3D(Resize3D&& operation)
    : GPUOperation(std::move(operation)),
      attr_(operation.attr_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

Resize3D& Resize3D::operator=(Resize3D&& operation) {
  if (this != &operation) {
    attr_ = operation.attr_;
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status Resize3D::Compile(const CreationContext& creation_context) {
  std::string code = GetResize3DCode(definition_, attr_.type, &args_);
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

absl::Status Resize3D::BindArguments() {
  RETURN_IF_ERROR(args_.SetObjectRef("src_tensor", src_[0]));
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  RETURN_IF_ERROR(args_.SetInt("border_x", src_[0]->Width() - 1));
  RETURN_IF_ERROR(args_.SetInt("border_y", src_[0]->Height() - 1));
  RETURN_IF_ERROR(args_.SetInt("border_z", src_[0]->Depth() - 1));
  RETURN_IF_ERROR(args_.SetFloat(
      "scale_factor_x",
      CalculateResizeScale(src_[0]->Width(), dst_[0]->Width(), attr_)));
  RETURN_IF_ERROR(args_.SetFloat(
      "scale_factor_y",
      CalculateResizeScale(src_[0]->Height(), dst_[0]->Height(), attr_)));
  RETURN_IF_ERROR(args_.SetFloat(
      "scale_factor_z",
      CalculateResizeScale(src_[0]->Depth(), dst_[0]->Depth(), attr_)));
  RETURN_IF_ERROR(SetArguments(linked_operations_, &args_));
  return args_.Bind(kernel_.kernel());
}

int3 Resize3D::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices() * dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status Resize3D::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

absl::Status Resize3D::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Resize3D CreateResize3D(const OperationDef& definition,
                        const Resize3DAttributes& attr) {
  return Resize3D(definition, attr);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
