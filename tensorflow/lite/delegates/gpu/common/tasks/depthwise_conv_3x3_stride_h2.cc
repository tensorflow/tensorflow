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

#include "tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3_stride_h2.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace {
std::string GetKernelDepthWiseConv3x3StrideH2(const GpuInfo& gpu_info,
                                              const OperationDef& definition,
                                              bool weights_are_buffer,
                                              bool local_mem_uploads) {
  const auto src_tensor_type = definition.src_tensors[0].GetStorageType();

  std::string c = "MAIN_FUNCTION($0) {\n";
  if (definition.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += R"(
  int Y = GLOBAL_ID_1 * 2;
  int S = GLOBAL_ID_2;

  ACCUM_FLT4 r0 = INIT_ACCUM_FLT4(0.0f);
  ACCUM_FLT4 l0 = INIT_ACCUM_FLT4(0.0f);
)";
  if (local_mem_uploads) {
    c += "  __local FLT4 f[10];\n";
    c += "  int local_id = LOCAL_ID_1 * 8 + LOCAL_ID_0;\n";
    c += "  if (local_id < 10) {\n";
    c += "    f[local_id] = args.weights.Read(S * 10 + local_id);\n";
    c += "  }\n";
    c += "  LOCAL_MEM_BARRIER;\n";
  } else if (weights_are_buffer && gpu_info.SupportsPointersInKernels()) {
    c += "  __global FLT4* f = args.weights.GetPtr() + S * 10;\n";
  }
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
       "|| S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT4 s0, s1, s2;\n";
  c += "  int x0 = X * args.stride_x + args.padding_x;\n";
  c += "  int x1 = X * args.stride_x + args.padding_x + args.dilation_x;\n";
  c += "  int x2 = X * args.stride_x + args.padding_x + 2 * args.dilation_x;\n";
  c += "  int y0 = Y * 2 + args.padding_y;\n";
  c += "  int y1 = Y * 2 + args.padding_y + 1;\n";
  c += "  int y2 = Y * 2 + args.padding_y + 2;\n";
  c += "  int y3 = Y * 2 + args.padding_y + 3;\n";
  c += "  int y4 = Y * 2 + args.padding_y + 4;\n";
  std::string W[9] = {"f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"};
  std::string bias = "bias";
  if (!weights_are_buffer) {
    c += "   FLT4 f0 = args.weights.Read(0, S);\n";
    c += "   FLT4 f1 = args.weights.Read(1, S);\n";
    c += "   FLT4 f2 = args.weights.Read(2, S);\n";
    c += "   FLT4 f3 = args.weights.Read(3, S);\n";
    c += "   FLT4 f4 = args.weights.Read(4, S);\n";
    c += "   FLT4 f5 = args.weights.Read(5, S);\n";
    c += "   FLT4 f6 = args.weights.Read(6, S);\n";
    c += "   FLT4 f7 = args.weights.Read(7, S);\n";
    c += "   FLT4 f8 = args.weights.Read(8, S);\n";
  }
  if (!definition.src_tensors[0].SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
    c += "  bool x0_in = x0 >= 0 && x0 < args.src_tensor.Width();\n";
    c += "  bool x1_in = x1 >= 0 && x1 < args.src_tensor.Width();\n";
    c += "  bool x2_in = x2 >= 0 && x2 < args.src_tensor.Width();\n";
    c += "  x0 = clamp(x0, 0, args.src_tensor.Width() - 1);\n";
    c += "  x1 = clamp(x1, 0, args.src_tensor.Width() - 1);\n";
    c += "  x2 = clamp(x2, 0, args.src_tensor.Width() - 1);\n";
  }
  if (!definition.src_tensors[0].SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
    c += "  bool y0_in = y0 >= 0 && y0 < args.src_tensor.Height();\n";
    c += "  bool y1_in = y1 >= 0 && y1 < args.src_tensor.Height();\n";
    c += "  bool y2_in = y2 >= 0 && y2 < args.src_tensor.Height();\n";
    c += "  bool y3_in = y3 >= 0 && y3 < args.src_tensor.Height();\n";
    c += "  bool y4_in = y4 >= 0 && y4 < args.src_tensor.Height();\n";
    c += "  y0 = clamp(y0, 0, args.src_tensor.Height() - 1);\n";
    c += "  y1 = clamp(y1, 0, args.src_tensor.Height() - 1);\n";
    c += "  y2 = clamp(y2, 0, args.src_tensor.Height() - 1);\n";
    c += "  y3 = clamp(y3, 0, args.src_tensor.Height() - 1);\n";
    c += "  y4 = clamp(y4, 0, args.src_tensor.Height() - 1);\n";
  }

  if (src_tensor_type == TensorStorageType::BUFFER &&
      gpu_info.SupportsPointersInKernels()) {
    c += "  __global FLT4* src_loc = "
         "args.src_tensor.GetPtrWithSliceOffset(S);\n";
  }

  if (local_mem_uploads || weights_are_buffer) {
    const bool use_direct_buffer =
        !local_mem_uploads && !gpu_info.SupportsPointersInKernels();
    const std::string fetch_start =
        use_direct_buffer ? "args.weights.Read(S * 10 + " : "f[";
    const std::string fetch_end = use_direct_buffer ? ")" : "]";
    W[0] = fetch_start + "0" + fetch_end;
    W[1] = fetch_start + "1" + fetch_end;
    W[2] = fetch_start + "2" + fetch_end;
    W[3] = fetch_start + "3" + fetch_end;
    W[4] = fetch_start + "4" + fetch_end;
    W[5] = fetch_start + "5" + fetch_end;
    W[6] = fetch_start + "6" + fetch_end;
    W[7] = fetch_start + "7" + fetch_end;
    W[8] = fetch_start + "8" + fetch_end;
    bias = fetch_start + "9" + fetch_end;
  }
  auto read_3x_line = [&](int y) {
    std::string s0_check, s1_check, s2_check;
    if (!definition.src_tensors[0].SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
      s0_check += "x0_in";
      s1_check += "x1_in";
      s2_check += "x2_in";
    }
    if (!definition.src_tensors[0].SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
      const std::string y_in = "y" + std::to_string(y) + "_in";
      s0_check += s0_check.empty() ? y_in : (" && " + y_in);
      s1_check += s1_check.empty() ? y_in : (" && " + y_in);
      s2_check += s2_check.empty() ? y_in : (" && " + y_in);
    }
    if (!s0_check.empty()) {
      s0_check = " * INIT_FLT(" + s0_check + ")";
    }
    if (!s1_check.empty()) {
      s1_check = " * INIT_FLT(" + s1_check + ")";
    }
    if (!s2_check.empty()) {
      s2_check = " * INIT_FLT(" + s2_check + ")";
    }

    const std::string yc = "y" + std::to_string(y);
    if (src_tensor_type == TensorStorageType::BUFFER &&
        gpu_info.SupportsPointersInKernels()) {
      c += "    s0 = src_loc[args.src_tensor.GetWHOffset(x0, " + yc + ")]" +
           s0_check + ";\n";
      c += "    s1 = src_loc[args.src_tensor.GetWHOffset(x1, " + yc + ")]" +
           s1_check + ";\n";
      c += "    s2 = src_loc[args.src_tensor.GetWHOffset(x2, " + yc + ")]" +
           s2_check + ";\n";
    } else {
      c +=
          "    s0 = args.src_tensor.Read(x0, " + yc + ", S)" + s0_check + ";\n";
      c +=
          "    s1 = args.src_tensor.Read(x1, " + yc + ", S)" + s1_check + ";\n";
      c +=
          "    s2 = args.src_tensor.Read(x2, " + yc + ", S)" + s2_check + ";\n";
    }
  };
  read_3x_line(0);
  c += "    r0 += TO_ACCUM_TYPE(" + W[0] + " * s0);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[1] + " * s1);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[2] + " * s2);\n";
  read_3x_line(1);
  c += "    r0 += TO_ACCUM_TYPE(" + W[3] + " * s0);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[4] + " * s1);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[5] + " * s2);\n";
  read_3x_line(2);
  c += "    r0 += TO_ACCUM_TYPE(" + W[6] + " * s0);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[7] + " * s1);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[8] + " * s2);\n";
  c += "    l0 += TO_ACCUM_TYPE(" + W[0] + " * s0);\n";
  c += "    l0 += TO_ACCUM_TYPE(" + W[1] + " * s1);\n";
  c += "    l0 += TO_ACCUM_TYPE(" + W[2] + " * s2);\n";
  read_3x_line(3);
  c += "    l0 += TO_ACCUM_TYPE(" + W[3] + " * s0);\n";
  c += "    l0 += TO_ACCUM_TYPE(" + W[4] + " * s1);\n";
  c += "    l0 += TO_ACCUM_TYPE(" + W[5] + " * s2);\n";
  read_3x_line(4);
  c += "    l0 += TO_ACCUM_TYPE(" + W[6] + " * s0);\n";
  c += "    l0 += TO_ACCUM_TYPE(" + W[7] + " * s1);\n";
  c += "    l0 += TO_ACCUM_TYPE(" + W[8] + " * s2);\n";
  if (!weights_are_buffer) {
    c += "   FLT4 bias = args.weights.Read(9, S);\n";
  }
  c += "  r0 += TO_ACCUM_TYPE(" + bias + ");\n";
  c += "  l0 += TO_ACCUM_TYPE(" + bias + ");\n";
  c += R"(
  if (Y < args.dst_tensor.Height()) {
    FLT4 value = TO_FLT4(r0);
    args.dst_tensor.Write(value, X, Y, S);
  }
  if (Y + 1 < args.dst_tensor.Height()) {
    FLT4 value = TO_FLT4(l0);
    args.dst_tensor.Write(value, X, Y + 1, S);
  }
}
)";

  return c;
}

}  // namespace

int3 DepthWiseConv3x3StrideH2::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = DivideRoundUp(dst_[0]->Height(), 2);
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

void DepthWiseConv3x3StrideH2::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info, std::vector<int3>* work_groups) const {
  if (local_mem_uploads_) {
    work_groups->push_back(work_group_size_);
  } else {
    GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info, grid_size_,
                          work_groups);
  }
}

DepthWiseConv3x3StrideH2 CreateDepthWiseConv3x3StrideH2(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info) {
  bool weights_are_buffer = !gpu_info.SupportsImages() ||
                            gpu_info.IsPowerVR() || gpu_info.IsMali() ||
                            gpu_info.IsApple();

  DepthWiseConv3x3StrideH2 desc(definition);
  desc.local_mem_uploads_ = weights_are_buffer && gpu_info.IsPowerVR();
  if (gpu_info.IsApple() &&
      gpu_info.apple_info.IsLocalMemoryPreferredOverGlobal()) {
    desc.local_mem_uploads_ = true;
  }
  desc.work_group_size_ = int3(8, 4, 1);
  desc.code_ = GetKernelDepthWiseConv3x3StrideH2(
      gpu_info, definition, weights_are_buffer, desc.local_mem_uploads_);
  auto src_desc = definition.src_tensors[0];
  desc.AddSrcTensor("src_tensor", src_desc);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.args_.AddInt("padding_x", -attr.padding.prepended.w);
  desc.args_.AddInt("padding_y", -attr.padding.prepended.h);
  desc.args_.AddInt("stride_x", attr.strides.w);
  desc.args_.AddInt("dilation_x", attr.dilations.w);

  desc.UploadWeightsAndBiases(attr.weights, attr.bias, weights_are_buffer);
  return desc;
}

bool IsDepthWiseConv3x3StrideH2Supported(
    const DepthwiseConvolution2DAttributes& attr) {
  return attr.weights.shape.o == 1 && attr.weights.shape.h == 3 &&
         attr.weights.shape.w == 3 && attr.strides.h == 2 &&
         attr.dilations.h == 1;
}

}  // namespace gpu
}  // namespace tflite
