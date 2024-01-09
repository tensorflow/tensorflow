/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/gather.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tflite {
namespace gpu {

namespace {
std::string GetGatherCode(const OperationDef& op_def, GatherAttributes attr) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  int idx;\n";
  c += "  args.src_tensor::type result;\n";
  switch (attr.axis) {
    case Axis::BATCH:
      c += "  idx = args.indices.Read<int>(0, 0, 0, B).x;\n";
      c += "  result = args.src_tensor.Read(X, Y, "
           "S, idx);\n";
      break;
    case Axis::HEIGHT:
      c += "  idx = args.indices.Read<int>(0, 0, 0, Y).x;\n";
      c += "  result = args.src_tensor.Read(X, idx, "
           "S, B);\n";
      break;
    case Axis::WIDTH:
      c += "  idx = args.indices.Read<int>(0, 0, 0, X).x;\n";
      c += "  result = args.src_tensor.Read(idx, Y, "
           ", S, B);\n";
      break;
    case Axis::CHANNELS:
      c += "  idx = args.indices.Read<int>(0, 0, 0, S * 4).x;\n";
      c += "  args.src_tensor.ReadPerChannel(result.x, X, Y, idx, B);\n";
      c += "  idx = args.indices.Read<int>(0, 0, 0, S * 4 + 1).x;\n";
      c += "  args.src_tensor.ReadPerChannel(result.y, X, Y, idx, B);\n";
      c += "  idx = args.indices.Read<int>(0, 0, 0, S * 4 + 2).x;\n";
      c += "  args.src_tensor.ReadPerChannel(result.z, X, Y, idx, B);\n";
      c += "  idx = args.indices.Read<int>(0, 0, 0, S * 4 + 3).x;\n";
      c += "  args.src_tensor.ReadPerChannel(result.w, X, Y, idx, B);\n";
      break;
    default:
      c += "  return;\n";
  }
  c += "  args.dst_tensor.Write(result, X, Y, S);\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateGather(const GpuInfo& gpu_info, const OperationDef& op_def,
                          const GatherAttributes& attr) {
  GPUOperation op(op_def);
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  if (op_def.src_tensors.size() == 1) {  // Constant indices
    BHWC shape = BHWC(attr.indices.shape.v, 1, 1, 1);
    TensorStorageType storage_type = GetStorageTypeForLinearTensor(
        gpu_info, DataType::INT32, attr.indices.shape);
    TensorDescriptor indices =
        CreateBhwcTensorDescriptor(DataType::INT32, storage_type, shape);
    indices.UploadData(attr.indices);
    op.args_.AddObject("indices",
                       std::make_unique<TensorDescriptor>(std::move(indices)));
  } else {  // Runtime indices
    op.AddSrcTensor("indices", op_def.src_tensors[1]);
  }
  op.code_ = GetGatherCode(op_def, attr);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
