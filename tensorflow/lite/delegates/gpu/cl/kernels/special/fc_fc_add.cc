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

#include "tensorflow/lite/delegates/gpu/cl/kernels/special/fc_fc_add.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/cl/device_info.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
bool UseBufferForWeights(const GpuInfo& gpu_info) {
  return gpu_info.IsAdreno() || gpu_info.IsAMD() || gpu_info.IsMali();
}
}  // namespace

FCFCAdd::FCFCAdd(const OperationDef& definition, const GpuInfo& gpu_info)
    : GPUOperation(definition) {
  if (gpu_info.IsAdreno()) {
    if (gpu_info.adreno_info.IsAdreno3xx()) {
      work_group_size_ = int3(16, 4, 1);
    } else if (gpu_info.adreno_info.IsAdreno4xx()) {
      work_group_size_ = int3(32, 4, 1);
    } else {
      work_group_size_ = int3(32, 4, 1);
    }
  } else if (gpu_info.IsIntel()) {
    work_group_size_ = int3(8, 4, 1);
  } else if (gpu_info.IsNvidia()) {
    work_group_size_ = int3(8, 4, 1);
  } else if (gpu_info.IsPowerVR()) {
    work_group_size_ = int3(8, 4, 1);
  } else {
    work_group_size_ = int3(16, 4, 1);
  }
  code_ = GetFCFCAddKernelCode(definition_, gpu_info);
}

FCFCAdd::FCFCAdd(FCFCAdd&& kernel) : GPUOperation(std::move(kernel)) {}

FCFCAdd& FCFCAdd::operator=(FCFCAdd&& kernel) {
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

std::string FCFCAdd::GetFCFCAddKernelCode(const OperationDef& op_def,
                                          const GpuInfo& gpu_info) {
  AddSrcTensor("src_tensor_0", op_def.src_tensors[0]);
  AddSrcTensor("src_tensor_1", op_def.src_tensors[1]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  const bool weights_are_buffer = UseBufferForWeights(gpu_info);

  std::string c = GetCommonDefines(op_def.precision);
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

  c += R"(__kernel void main_function($0) {
  int gid = get_global_id(0);
  int2 tid = (int2)(get_local_id(0), get_local_id(1));
  ACCUM_FLT4 s = (ACCUM_FLT4)(0.0f);
  if (gid < args.dst_tensor.Slices()) {
    for (int c = tid.y; c < args.src_tensor_0.Slices(); c += WG_Y) {
      FLT4 v = args.src_tensor_0.Read(0, 0, c);
)";
  if (weights_are_buffer) {
    c += R"(FLT16 w = args.weights0.Read(c * args.dst_tensor.Slices() + gid);
      FLT4 partial = v.s0 * w.s0123;
      partial = mad(v.s1, w.s4567, partial);
      partial = mad(v.s2, w.s89ab, partial);
      partial = mad(v.s3, w.scdef, partial);
      s += TO_ACCUM_TYPE(partial);
)";
  } else {
    c += R"(FLT4 w0 = args.weights0.Read(c * 4 + 0, gid);
      FLT4 w1 = args.weights0.Read(c * 4 + 1, gid);
      FLT4 w2 = args.weights0.Read(c * 4 + 2, gid);
      FLT4 w3 = args.weights0.Read(c * 4 + 3, gid);
      FLT4 partial = v.s0 * w0;
      partial = mad(v.s1, w1, partial);
      partial = mad(v.s2, w2, partial);
      partial = mad(v.s3, w3, partial);
      s += TO_ACCUM_TYPE(partial);
)";
  }
  c += R"(    }
    for (int c = tid.y; c < args.src_tensor_1.Slices(); c += WG_Y) {
      FLT4 v = args.src_tensor_1.Read(0, 0, c);
      )";
  if (weights_are_buffer) {
    c += R"(FLT16 w = args.weights1.Read(c * args.dst_tensor.Slices() + gid);
      FLT4 partial = v.s0 * w.s0123;
      partial = mad(v.s1, w.s4567, partial);
      partial = mad(v.s2, w.s89ab, partial);
      partial = mad(v.s3, w.scdef, partial);
      s += TO_ACCUM_TYPE(partial);
)";
  } else {
    c += R"(FLT4 w0 = args.weights1.Read(c * 4 + 0, gid);
      FLT4 w1 = args.weights1.Read(c * 4 + 1, gid);
      FLT4 w2 = args.weights1.Read(c * 4 + 2, gid);
      FLT4 w3 = args.weights1.Read(c * 4 + 3, gid);
      FLT4 partial = v.s0 * w0;
      partial = mad(v.s1, w1, partial);
      partial = mad(v.s2, w2, partial);
      partial = mad(v.s3, w3, partial);
      s += TO_ACCUM_TYPE(partial);
)";
  }
  c += R"(    }
  }
  __local ACCUM_FLT4 temp[WG_X][WG_Y];
  temp[tid.x][tid.y] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (gid >= args.dst_tensor.Slices()) {
    return;
  }
  if (tid.y == 0) {
)";
  for (int i = 1; i < work_group_size_.y; ++i) {
    c += "    s += temp[tid.x][" + std::to_string(i) + "];\n";
  }
  c +=
      R"(    FLT4 r0 = TO_FLT4(s) + args.biases0.Read(gid) + args.biases1.Read(gid);
    args.dst_tensor.Write(r0, 0, 0, gid);
  }
})";

  return c;
}

int3 FCFCAdd::GetGridSize() const { return int3(dst_[0]->Slices(), 1, 1); }

FCFCAdd CreateFCFCAdd(const GpuInfo& gpu_info, const OperationDef& definition,
                      const FullyConnectedAttributes& attr0,
                      const FullyConnectedAttributes& attr1) {
  FCFCAdd result(definition, gpu_info);
  result.UploadWeights(attr0.weights, "weights0",
                       UseBufferForWeights(gpu_info));
  result.UploadWeights(attr1.weights, "weights1",
                       UseBufferForWeights(gpu_info));

  TensorLinearDescriptor desc0;
  desc0.storage_type = LinearStorageType::TEXTURE_2D;
  desc0.element_type = definition.GetDataType();
  desc0.UploadLinearData(attr0.bias);
  result.args_.AddObject(
      "biases0", absl::make_unique<TensorLinearDescriptor>(std::move(desc0)));

  TensorLinearDescriptor desc1;
  desc1.storage_type = LinearStorageType::TEXTURE_2D;
  desc1.element_type = definition.GetDataType();
  desc1.UploadLinearData(attr1.bias);
  result.args_.AddObject(
      "biases1", absl::make_unique<TensorLinearDescriptor>(std::move(desc1)));

  return result;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
