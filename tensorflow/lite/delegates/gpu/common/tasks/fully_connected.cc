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

#include "tensorflow/lite/delegates/gpu/common/tasks/fully_connected.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

namespace {
bool UseBufferForWeights(const GpuInfo& gpu_info) {
  return gpu_info.IsAdreno() || gpu_info.IsAMD() || gpu_info.IsMali() ||
         gpu_info.IsApple();
}

void RearrangeFCWeightsToOIO4I4(
    const tflite::gpu::Tensor<OHWI, DataType::INT8>& weights, uint8_t* dst) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);

  int counter = 0;
  for (int d = 0; d < dst_depth; ++d) {
    for (int s = 0; s < src_depth; ++s) {
      for (int i = 0; i < 4; ++i) {
        const int src_ch = s * 4 + i;
        for (int j = 0; j < 4; ++j) {
          const int dst_ch = d * 4 + j;
          if (src_ch < weights.shape.i && dst_ch < weights.shape.o) {
            int t =
                127 +
                weights.data[weights.shape.LinearIndex({dst_ch, 0, 0, src_ch})];
            if (t < 0) {
              t = 0;
            }
            dst[counter++] = t;
          } else {
            dst[counter++] = 127;
          }
        }
      }
    }
  }
}

}  // namespace

FullyConnected::FullyConnected(const OperationDef& definition,
                               const GpuInfo& gpu_info)
    : GPUOperation(definition) {
  if (gpu_info.IsAdreno()) {
    if (gpu_info.adreno_info.IsAdreno3xx()) {
      work_group_size_ = int3(16, 4, 1);
    } else if (gpu_info.adreno_info.IsAdreno4xx()) {
      work_group_size_ = int3(32, 4, 1);
    } else {
      work_group_size_ = int3(32, 4, 1);
    }
  } else if (gpu_info.IsIntel() || gpu_info.IsNvidia() ||
             gpu_info.IsPowerVR() || gpu_info.IsApple()) {
    work_group_size_ = int3(8, 4, 1);
  } else {
    work_group_size_ = int3(16, 4, 1);
  }
}

FullyConnected::FullyConnected(FullyConnected&& kernel)
    : GPUOperation(std::move(kernel)) {}

FullyConnected& FullyConnected::operator=(FullyConnected&& kernel) {
  if (this != &kernel) {
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

// We split vec vec dot (every thread do vec vec dot product in basic
// vec mat mult) on 4 parts to create more threads
// tid.y thread process every 4-th element in vec vec dot
// Good results for ~1024 x 1024 sizes, for other can be written more
// optimized shaders

std::string FullyConnected::GetFullyConnectedKernelCode(
    const OperationDef& op_def, const GpuInfo& gpu_info,
    bool weights_are_buffer, bool quantized) {
  const int wg_total_size = work_group_size_.x * work_group_size_.y;
  const std::string barrier =
      wg_total_size == 32 && gpu_info.IsWaveSizeEqualTo32()
          ? "SIMD_LOCAL_MEM_BARRIER"
          : "LOCAL_MEM_BARRIER";
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  std::string c;
  switch (op_def.precision) {
    case CalculationsPrecision::F32:
      c += "#define FLT16 float16\n";
      break;
    case CalculationsPrecision::F32_F16:
    case CalculationsPrecision::F16:
      c += "#define FLT16 half16\n";
      break;
  }

  c += "#define WG_X " + std::to_string(work_group_size_.x) + "\n";
  c += "#define WG_Y " + std::to_string(work_group_size_.y) + "\n";

  c += R"(MAIN_FUNCTION($0) {
  int gid = GLOBAL_ID_0;
  int2 tid = INIT_INT2v2(LOCAL_ID_0, LOCAL_ID_1);
  ACCUM_FLT4 s = INIT_ACCUM_FLT4(0.0f);
  if (gid < args.dst_tensor.Slices()) {
    for (int c = tid.y; c < args.src_tensor.Slices(); c += WG_Y) {
      FLT4 v = args.src_tensor.Read(0, 0, c);
)";
  if (weights_are_buffer) {
    c += R"(FLT16 w = args.weights.Read(c * args.dst_tensor.Slices() + gid);
      FLT4 partial = v.x * FLT16_0123(w);
      partial += v.y * FLT16_4567(w);
      partial += v.z * FLT16_89ab(w);
      partial += v.w * FLT16_cdef(w);
      s += TO_ACCUM_TYPE(partial);
)";
  } else {
    c += R"(FLT4 w0 = args.weights.Read(c * 4 + 0, gid);
      FLT4 w1 = args.weights.Read(c * 4 + 1, gid);
      FLT4 w2 = args.weights.Read(c * 4 + 2, gid);
      FLT4 w3 = args.weights.Read(c * 4 + 3, gid);
      )";
    if (quantized) {
      c += R"(w0 = w0 * args.q0 + args.q1;
      w1 = w1 * args.q0 + args.q1;
      w2 = w2 * args.q0 + args.q1;
      w3 = w3 * args.q0 + args.q1;
)";
    }
    c += R"(FLT4 partial = v.x * w0;
      partial += v.y * w1;
      partial += v.z * w2;
      partial += v.w * w3;
      s += TO_ACCUM_TYPE(partial);
)";
  }
  c += R"(    }
  }
  __local ACCUM_FLT4 temp[WG_X][WG_Y];
  temp[tid.x][tid.y] = s;
)";
  c += "  " + barrier + ";\n";
  c += R"(
  if (gid >= args.dst_tensor.Slices()) {
    return;
  }
  if (tid.y == 0) {
)";
  for (int i = 1; i < work_group_size_.y; ++i) {
    c += "    s += temp[tid.x][" + std::to_string(i) + "];\n";
  }
  c += R"(    FLT4 r0 = TO_FLT4(s) + args.biases.Read(gid);
    args.dst_tensor.Write(r0, 0, 0, gid);
  }
})";

  return c;
}

int3 FullyConnected::GetGridSize() const {
  return int3(dst_[0]->Slices(), 1, 1);
}

void FullyConnected::UploadQuantizedWeights(
    const tflite::gpu::Tensor<OHWI, DataType::INT8>& weights, float scale,
    float zero_point) {
  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  Texture2DDescriptor desc;
  desc.element_type = DataType::UINT8;
  desc.normalized = true;
  desc.normalized_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
  desc.size = int2(src_depth * 4, dst_depth);
  desc.data.resize(src_depth * 4 * dst_depth * 4);
  RearrangeFCWeightsToOIO4I4(weights, desc.data.data());

  if (definition_.precision == CalculationsPrecision::F32) {
    args_.AddFloat("q0", scale * 255.0f);
    args_.AddFloat("q1", -scale * (127.0 + zero_point));
  } else {
    args_.AddHalf("q0", half(scale * 255.0f));
    args_.AddHalf("q1", half(-scale * (127.0 + zero_point)));
  }
  args_.AddObject("weights",
                  std::make_unique<Texture2DDescriptor>(std::move(desc)));
}

FullyConnected CreateFullyConnected(const GpuInfo& gpu_info,
                                    const OperationDef& definition,
                                    const FullyConnectedAttributes& attr) {
  FullyConnected result(definition, gpu_info);
  result.UploadWeights(attr.weights, UseBufferForWeights(gpu_info));
  result.code_ = result.GetFullyConnectedKernelCode(
      definition, gpu_info, UseBufferForWeights(gpu_info), false);

  TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
      gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
  result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                       std::move(bias_tensor_desc)));

  return result;
}

FullyConnected CreateFullyConnected(const GpuInfo& gpu_info,
                                    const OperationDef& definition,
                                    const FullyConnectedInt8Attributes& attr) {
  FullyConnected result(definition, gpu_info);
  result.UploadQuantizedWeights(attr.weights, attr.scale, attr.zero_point);
  result.code_ =
      result.GetFullyConnectedKernelCode(definition, gpu_info, false, true);

  TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
      gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
  result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                       std::move(bias_tensor_desc)));

  return result;

  return result;
}

}  // namespace gpu
}  // namespace tflite
