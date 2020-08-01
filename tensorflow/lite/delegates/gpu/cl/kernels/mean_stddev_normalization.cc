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

#include "tensorflow/lite/delegates/gpu/cl/kernels/mean_stddev_normalization.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"

namespace tflite {
namespace gpu {
namespace cl {

MeanStdDevNormalization::MeanStdDevNormalization(const OperationDef& definition)
    : GPUOperation(definition) {
  code_ = GetNormalizationCode();
}

std::string MeanStdDevNormalization::GetNormalizationCode() {
  AddSrcTensor("src_tensor", definition_.src_tensors[0]);
  AddDstTensor("dst_tensor", definition_.dst_tensors[0]);

  std::string c = GetCommonDefines(definition_.precision);
  c += R"(__kernel void main_function(
$0) {
  if (get_global_id(0) > 0) { return; }
  size_t B = get_global_id(1);
  if (get_global_id(2) > 0) { return; }
  if (B >= args.src_tensor.Batch()) { return; }
  // Calculate the total sum of the input tensor.
  // First, get a local sum of input[local_id_x + N*local_size_x] for all N.
  float sum = 0.0f;
  for (int S = 0; S < args.src_tensor.Slices(); ++S) {
    const float4 t = args.src_tensor.Read<float>(0, 0, S, B);
    // Filter out reads beyond the end of the tensor.
    const int4 is_after_end_of_tensor = (int4)(0, 1, 2, 3) >= (args.src_tensor.Channels() - S * 4);
    const float4 filtered_t = select(t, (float4)(0.0f), is_after_end_of_tensor);
    sum += filtered_t.x + filtered_t.y + filtered_t.z + filtered_t.w;
  }
  // Calculate the mean
  const float mean = sum / args.src_tensor.Channels();
  // Calculate the squared sum of the difference from the mean.
  float sum_diff_sq = 0.0f;
  for (int S = 0; S < args.src_tensor.Slices(); ++S) {
    const float4 t = args.src_tensor.Read<float>(0, 0, S, B);
    const float4 diff = t - mean;
    // Filter out reads beyond the end of the tensor.
    const int4 is_after_end_of_tensor = (int4)(0, 1, 2, 3) >= (args.src_tensor.Channels() - S * 4);
    const float4 filtered_diff = select(diff, (float4)(0.0f), is_after_end_of_tensor);
    float dotprod = dot(filtered_diff, filtered_diff);
    sum_diff_sq += dotprod;
  }
  // Calculate 1/stddev (with the 'regulazing constant' as in tensor_utils.cc)
  const float variance = sum_diff_sq / args.src_tensor.Channels();
  const float stddev_inv =  rsqrt(variance + 1.0e-8f);
  // Calculate (t-mean)/stddev for each element
  for (int S = 0; S < args.src_tensor.Slices(); ++S) {
    const float4 t = args.src_tensor.Read<float>(0, 0, S, B);
    FLT4 result = TO_FLT4((t - mean) * stddev_inv);
    args.dst_tensor.Write(result, 0, 0, S, B);
  }
})";
  return c;
}

int3 MeanStdDevNormalization::GetGridSize() const {
  const int grid_x = 1;
  const int grid_y = src_[0]->Batch();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

MeanStdDevNormalization CreateMeanStdDevNormalization(
    const OperationDef& definition) {
  return MeanStdDevNormalization(definition);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
