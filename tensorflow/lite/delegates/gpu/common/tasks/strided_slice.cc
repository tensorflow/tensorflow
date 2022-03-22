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

#include "tensorflow/lite/delegates/gpu/common/tasks/strided_slice.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
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
    : GPUOperation(definition), attributes_(attr) {
  work_group_size_ = int3(8, 4, 1);
  code_ = GetStridedSliceCode(definition_, Is4Aligned(attributes_));
}

StridedSlice::StridedSlice(StridedSlice&& operation)
    : GPUOperation(std::move(operation)), attributes_(operation.attributes_) {}

StridedSlice& StridedSlice::operator=(StridedSlice&& operation) {
  if (this != &operation) {
    attributes_ = operation.attributes_;
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string StridedSlice::GetStridedSliceCode(const OperationDef& op_def,
                                              bool alignedx4) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddInt("offset_x");
  args_.AddInt("offset_y");
  args_.AddInt("offset_z");
  args_.AddInt("offset_b");
  args_.AddInt("stride_x");
  args_.AddInt("stride_y");
  args_.AddInt("stride_z");
  args_.AddInt("stride_b");

  const std::string batch_id =
      op_def.dst_tensors[0].HasAxis(Axis::BATCH) ? "B" : "0";
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  int s_x = X * args.stride_x + args.offset_x;\n";
  c += "  int s_y = Y * args.stride_y + args.offset_y;\n";
  if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int s_b = " + batch_id + " * args.stride_b + args.offset_b;\n";
    c += "  args.src_tensor.SetBatchRef(s_b);\n";
  }
  if (alignedx4) {
    c += "  int s_z = S + args.offset_z;\n";
    c += "  args.src_tensor::type result = args.src_tensor.Read(s_x, s_y, "
         "s_z);\n";
  } else {
    c += "  args.src_tensor::type result;\n";
    const std::string postfixes[] = {"x", "y", "z", "w"};
    for (int i = 0; i < 4; ++i) {
      c += "  {\n";
      const std::string channel = "(S * 4 + " + std::to_string(i) + ")";
      c += "    int s_ch = min(" + channel +
           " * args.stride_z + args.offset_z, args.src_tensor.Channels() - "
           "1);\n";
      c += "    args.src_tensor.ReadPerChannel(result." + postfixes[i] +
           ", s_x, s_y, s_ch);\n";
      c += "  }\n";
    }
  }
  c += "  args.dst_tensor.Write(result, X, Y, S);\n";
  c += "}\n";
  return c;
}

absl::Status StridedSlice::BindArguments(ArgumentsBinder* args) {
  int4 offset = GetOffset(attributes_, src_[0]->Width(), src_[0]->Height(),
                          src_[0]->Channels(), src_[0]->Batch());
  RETURN_IF_ERROR(args->SetInt("offset_x", offset.x));
  RETURN_IF_ERROR(args->SetInt("offset_y", offset.y));
  RETURN_IF_ERROR(args->SetInt("offset_z", offset.z));
  RETURN_IF_ERROR(args->SetInt("offset_b", offset.w));
  RETURN_IF_ERROR(args->SetInt("stride_x", attributes_.strides.w));
  RETURN_IF_ERROR(args->SetInt("stride_y", attributes_.strides.h));
  RETURN_IF_ERROR(args->SetInt("stride_z", attributes_.strides.c));
  RETURN_IF_ERROR(args->SetInt("stride_b", attributes_.strides.b));
  return absl::OkStatus();
}

int3 StridedSlice::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

StridedSlice CreateStridedSlice(const OperationDef& definition,
                                const SliceAttributes& attr) {
  return StridedSlice(definition, attr);
}

}  // namespace gpu
}  // namespace tflite
