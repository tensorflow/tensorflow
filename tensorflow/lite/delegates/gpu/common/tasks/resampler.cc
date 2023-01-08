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

#include "tensorflow/lite/delegates/gpu/common/tasks/resampler.h"

#include <string>

namespace tflite {
namespace gpu {
namespace {

std::string GetResamplerCode(const GpuInfo& gpu_info,
                             const OperationDef& op_def) {
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
  c += "  float2 f_coords = args.warp_tensor.Read<float>(X, Y, 0).xy;\n";
  c += "  float2 f_coords_floor = floor(f_coords);\n";
  c += "  int4 st;\n";
  c += "  st.xy = INIT_INT2v2(f_coords_floor.x, f_coords_floor.y);\n";
  c += "  st.zw = st.xy + INIT_INT2v2(1, 1);\n";
  c += "  float2 t = f_coords - f_coords_floor;\n";
  bool supports_hw_zero_clamp =
      op_def.src_tensors[0].SupportsZeroClamp(Axis::WIDTH, gpu_info) &&
      op_def.src_tensors[0].SupportsZeroClamp(Axis::HEIGHT, gpu_info);
  if (supports_hw_zero_clamp) {
    c += R"(
  float4 src0 = args.src_tensor.Read<float>(st.x, st.y, S);
  float4 src1 = args.src_tensor.Read<float>(st.z, st.y, S);
  float4 src2 = args.src_tensor.Read<float>(st.x, st.w, S);
  float4 src3 = args.src_tensor.Read<float>(st.z, st.w, S);
)";
  } else {
    c += R"(
  bool stx_in = st.x >= 0 && st.x < args.src_tensor.Width();
  bool stz_in = st.z >= 0 && st.z < args.src_tensor.Width();
  bool sty_in = st.y >= 0 && st.y < args.src_tensor.Height();
  bool stw_in = st.w >= 0 && st.w < args.src_tensor.Height();
  float4 src0 = (stx_in && sty_in) ? args.src_tensor.Read<float>(st.x, st.y, S) : INIT_FLOAT4(0.0f);
  float4 src1 = (stz_in && sty_in) ? args.src_tensor.Read<float>(st.z, st.y, S) : INIT_FLOAT4(0.0f);
  float4 src2 = (stx_in && stw_in) ? args.src_tensor.Read<float>(st.x, st.w, S) : INIT_FLOAT4(0.0f);
  float4 src3 = (stz_in && stw_in) ? args.src_tensor.Read<float>(st.z, st.w, S) : INIT_FLOAT4(0.0f);
    )";
  }
  c += "  FLT4 r0 = TO_FLT4(mix(mix(src0, src1, t.x), mix(src2, src3, t.x), "
       "t.y));\n";
  c += "  args.dst_tensor.Write(r0, X, Y, S);\n";
  c += "}\n";
  return c;
}

}  // namespace

GPUOperation CreateResampler(const GpuInfo& gpu_info,
                             const OperationDef& definition) {
  GPUOperation op(definition);
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddSrcTensor("warp_tensor", definition.src_tensors[1]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GetResamplerCode(gpu_info, definition);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
