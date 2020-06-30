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

#include "tensorflow/lite/delegates/gpu/cl/kernels/strided_slice.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetStridedSliceCode(const OperationDef& op_def, bool alignedx4,
                                Arguments* args) {
  args->AddObjectRef(
      "src_tensor", AccessType::READ,
      absl::make_unique<TensorDescriptor>(op_def.src_tensors[0]));
  args->AddObjectRef(
      "dst_tensor", AccessType::WRITE,
      absl::make_unique<TensorDescriptor>(op_def.dst_tensors[0]));
  args->AddInt("offset_x");
  args->AddInt("offset_y");
  args->AddInt("offset_z");
  args->AddInt("offset_b");
  args->AddInt("stride_x");
  args->AddInt("stride_y");
  args->AddInt("stride_z");
  args->AddInt("stride_b");

  const std::string batch_id =
      op_def.dst_tensors[0].HasAxis(Axis::BATCH) ? "B" : "0";
  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = get_global_id(0);\n";
  }
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  int s_x = X * args.stride_x + args.offset_x;\n";
  c += "  int s_y = Y * args.stride_y + args.offset_y;\n";
  if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int s_b = " + batch_id + " * args.stride_b + args.offset_b;\n";
    c += "  args.src_tensor.SetBatchRef(s_b);\n";
  }
  if (alignedx4) {
    c += "  int s_z = Z + args.offset_z;\n";
    c += "  FLT4 result = args.src_tensor.Read(s_x, s_y, s_z);\n";
  } else {
    c += "  FLT4 result;\n";
    const std::string postfixes[] = {"x", "y", "z", "w"};
    for (int i = 0; i < 4; ++i) {
      c += "  {\n";
      const std::string channel = "(Z * 4 + " + std::to_string(i) + ")";
      c += "    int s_ch = " + channel + " * args.stride_z + args.offset_z;\n";
      c += "    int s_z = min(s_ch >> 2, args.src_tensor.Slices() - 1);\n";
      c += "    int s_z_rem = s_ch & 3;\n";
      c += "    FLT4 t = args.src_tensor.Read(s_x, s_y, s_z);\n";
      c += "    FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
      c += "    result." + postfixes[i] + " = t_ar[s_z_rem];\n";
      c += "  }\n";
    }
  }
  c += "  args.dst_tensor.Write(result, X, Y, Z);\n";
  c += "}\n";
  return c;
}

bool Is4Aligned(const SliceAttributes& attr) {
  return attr.strides.c == 1 && attr.starts.c % 4 == 0;
}

int4 GetOffset(const SliceAttributes& attr, int src_width, int src_height,
               int src_channels, int src_batch) {
  int4 offset;
  if (attr.strides.w > 0) {
    offset.x = attr.starts.w;
  } else {
    if (attr.ends.w > 0) {
      offset.x = attr.ends.w;
    } else {
      offset.x = src_width + attr.ends.w;
    }
  }
  if (attr.strides.h > 0) {
    offset.y = attr.starts.h;
  } else {
    if (attr.ends.h > 0) {
      offset.y = attr.ends.h;
    } else {
      offset.y = src_height + attr.ends.h;
    }
  }
  if (attr.strides.c > 0) {
    offset.z = attr.starts.c;
  } else {
    if (attr.ends.c > 0) {
      offset.z = attr.ends.c;
    } else {
      offset.z = src_channels + attr.ends.c;
    }
  }
  if (Is4Aligned(attr)) {
    offset.z /= 4;
  }
  if (attr.strides.b > 0) {
    offset.w = attr.starts.b;
  } else {
    if (attr.ends.b > 0) {
      offset.w = attr.ends.b;
    } else {
      offset.w = src_batch + attr.ends.b;
    }
  }
  return offset;
}

}  // namespace

StridedSlice::StridedSlice(const OperationDef& definition,
                           const SliceAttributes& attr)
    : GPUOperation(definition), attributes_(attr), work_group_size_(8, 4, 1) {}

StridedSlice::StridedSlice(StridedSlice&& operation)
    : GPUOperation(std::move(operation)),
      attributes_(operation.attributes_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

StridedSlice& StridedSlice::operator=(StridedSlice&& operation) {
  if (this != &operation) {
    attributes_ = operation.attributes_;
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status StridedSlice::Compile(const CreationContext& creation_context) {
  std::string code =
      GetStridedSliceCode(definition_, Is4Aligned(attributes_), &args_);
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

absl::Status StridedSlice::BindArguments() {
  RETURN_IF_ERROR(args_.SetObjectRef("src_tensor", src_[0]));
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  int4 offset = GetOffset(attributes_, src_[0]->Width(), src_[0]->Height(),
                          src_[0]->Channels(), src_[0]->Batch());
  RETURN_IF_ERROR(args_.SetInt("offset_x", offset.x));
  RETURN_IF_ERROR(args_.SetInt("offset_y", offset.y));
  RETURN_IF_ERROR(args_.SetInt("offset_z", offset.z));
  RETURN_IF_ERROR(args_.SetInt("offset_b", offset.w));
  RETURN_IF_ERROR(args_.SetInt("stride_x", attributes_.strides.w));
  RETURN_IF_ERROR(args_.SetInt("stride_y", attributes_.strides.h));
  RETURN_IF_ERROR(args_.SetInt("stride_z", attributes_.strides.c));
  RETURN_IF_ERROR(args_.SetInt("stride_b", attributes_.strides.b));
  RETURN_IF_ERROR(SetArguments(linked_operations_, &args_));
  return args_.Bind(kernel_.kernel());
}

int3 StridedSlice::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status StridedSlice::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status StridedSlice::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

StridedSlice CreateStridedSlice(const OperationDef& definition,
                                const SliceAttributes& attr) {
  return StridedSlice(definition, attr);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
