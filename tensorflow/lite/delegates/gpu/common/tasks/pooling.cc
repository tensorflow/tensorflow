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

#include "tensorflow/lite/delegates/gpu/common/tasks/pooling.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
std::string GetAveragePoolingKernelCode(const OperationDef& op_def,
                                        bool stride_correction,
                                        GPUOperation* op) {
  auto src_desc = op_def.src_tensors[0];
  src_desc.SetAddressMode(AddressMode::kZero);
  if (op_def.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddSrcTensor("src_tensor", src_desc);
  auto dst_desc = op_def.dst_tensors[0];
  if (op_def.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddDstTensor("dst_tensor", dst_desc);

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

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int X = GLOBAL_ID_0;\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int D = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
  }
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  float4 r = INIT_FLOAT4(0.0f);\n";
  c += "  float window_size = 0.0;\n";
  if (stride_correction) {
    c += "  int xs = " +
         GetXStrideCorrectedV2("X", "args.src_tensor.Batch()", "args.stride_x",
                               "args.padding_x") +
         ";\n";
  } else {
    if (op_def.IsBatchSupported()) {
      c += "  int xs = X * args.stride_x + args.padding_x * "
           "args.src_tensor.Batch();\n";
    } else {
      c += "  int xs = X * args.stride_x + args.padding_x;\n";
    }
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
         "INIT_FLOAT4(0.0f);\n";
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
                                    GPUOperation* op) {
  auto src_desc = op_def.src_tensors[0];
  if (op_def.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddSrcTensor("src_tensor", src_desc);
  auto dst_desc = op_def.dst_tensors[0];
  if (op_def.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  op->AddDstTensor("dst_tensor", dst_desc);
  if (output_indices) {
    auto dst_ind_desc = op_def.dst_tensors[1];
    if (op_def.IsBatchSupported()) {
      dst_ind_desc.SetStateVar("BatchedWidth", "true");
    }
    op->AddDstTensor("dst_indices", dst_ind_desc);
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

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int X = GLOBAL_ID_0;\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int D = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
  }
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT4 maximum = INIT_FLT4(-10000.0f);\n";
  if (output_indices) {
    c += "  int4 indexes = INIT_INT4v4(0, 0, 0, 0);\n";
  }
  if (stride_correction) {
    c += "  int xs = " +
         GetXStrideCorrectedV2("X", "args.src_tensor.Batch()", "args.stride_x",
                               "args.padding_x") +
         ";\n";
  } else {
    if (op_def.IsBatchSupported()) {
      c += "  int xs = X * args.stride_x + args.padding_x * "
           "args.src_tensor.Batch();\n";
    } else {
      c += "  int xs = X * args.stride_x + args.padding_x;\n";
    }
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
      c += "      int index_counter = (ky * args.kernel_size_x + kx) * "
           "args.kernel_size_z + kz;\n";
    } else {
      c += "      int index_counter = ky * args.kernel_size_x + kx;\n";
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
    c += "  args.dst_indices.Write<int>(indexes, " + dst_coord + ");\n";
  }
  c += "}\n";

  return c;
}
}  // namespace

GPUOperation CreatePooling(const OperationDef& definition,
                           const Pooling2DAttributes& attr) {
  GPUOperation op(definition);
  op.args_.AddInt("kernel_size_x", attr.kernel.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("kernel_size_y", attr.kernel.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  const bool stride_correction =
      definition.IsBatchSupported() && attr.strides.w != 1;
  if (attr.type == PoolingType::AVERAGE) {
    op.code_ = GetAveragePoolingKernelCode(definition, stride_correction, &op);
  } else if (attr.type == PoolingType::MAX) {
    op.code_ = GetMaxPoolingKernelCode(definition, stride_correction,
                                       attr.output_indices, &op);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

GPUOperation CreatePooling(const OperationDef& definition,
                           const Pooling3DAttributes& attr) {
  GPUOperation op(definition);
  op.args_.AddInt("kernel_size_x", attr.kernel.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("kernel_size_y", attr.kernel.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("kernel_size_z", attr.kernel.d);
  op.args_.AddInt("padding_z", -attr.padding.prepended.d);
  op.args_.AddInt("stride_z", attr.strides.d);
  const bool stride_correction =
      definition.IsBatchSupported() && attr.strides.w != 1;
  if (attr.type == PoolingType::AVERAGE) {
    op.code_ = GetAveragePoolingKernelCode(definition, stride_correction, &op);
  } else if (attr.type == PoolingType::MAX) {
    op.code_ = GetMaxPoolingKernelCode(definition, stride_correction,
                                       attr.output_indices, &op);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
