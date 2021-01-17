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

#include "tensorflow/lite/delegates/gpu/metal/kernels/prelu.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {

ComputeTaskDescriptor PReLU(const OperationDef& definition,
                            const PReLUAttributes& attr) {
  auto alpha_buffer =
      absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.alpha);
  if (!alpha_buffer) {
    return {};
  }
  ComputeTaskDescriptor desc(definition);
  desc.is_linkable = true;
  if (attr.clip != 0) {
    desc.args.AddFloat("clip", attr.clip);
    desc.shader_source =
        R"(
  in_out_value = FLT4(clamp(in_out_value, FLT4(0.0f), FLT4(args.clip)) + args.alpha.Read(S_COORD) * min(FLT4(0.0f), in_out_value));
)";
  } else {
    desc.shader_source =
        R"(
  in_out_value = FLT4(max(FLT4(0.0f), in_out_value) + args.alpha.Read(S_COORD) * min(FLT4(0.0f), in_out_value));
)";
  }
  auto data_type = DeduceDataTypeFromPrecision(definition.precision);
  const int dst_channels_aligned = AlignByN(alpha_buffer->shape.v, 4);
  BufferDescriptor alpha_desc;
  alpha_desc.element_type = data_type;
  alpha_desc.element_size = 4;
  alpha_desc.data = GetByteBufferConvertedResized(alpha_buffer->data, data_type,
                                                  dst_channels_aligned);
  alpha_desc.size = alpha_desc.data.size();
  desc.args.AddObject(
      "alpha", absl::make_unique<BufferDescriptor>(std::move(alpha_desc)));
  return desc;
}

ComputeTaskDescriptor PReLUFull(const OperationDef& definition,
                                const PReLUAttributes& attr) {
  auto alpha = absl::get_if<Tensor<HWC, DataType::FLOAT32>>(&attr.alpha);
  if (!alpha) {
    return {};
  }
  ComputeTaskDescriptor desc(definition);
  desc.is_linkable = true;
  if (attr.clip != 0) {
    desc.args.AddFloat("clip", attr.clip);
    desc.shader_source =
        R"(
  in_out_value = FLT4(clamp(in_out_value, FLT4(0.0f), FLT4(args.clip)) + args.alpha.Read(X_COORD, Y_COORD, S_COORD) * min(FLT4(0.0f), in_out_value));
)";
  } else {
    desc.shader_source =
        R"(
  in_out_value = FLT4(max(FLT4(0.0f), in_out_value) + args.alpha.Read(X_COORD, Y_COORD, S_COORD) * min(FLT4(0.0f), in_out_value));
)";
  }
  TensorDescriptor alpha_desc{definition.GetDataType(),
                              TensorStorageType::BUFFER, Layout::HWC};
  alpha_desc.UploadData(*alpha);
  desc.args.AddObject(
      "alpha", absl::make_unique<TensorDescriptor>(std::move(alpha_desc)));
  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
