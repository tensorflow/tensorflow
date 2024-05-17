/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/conversion.h"

#include <memory>
#include <string>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"

namespace tflite {
namespace gpu {

GPUOperation CreateTensorToTensorOp(const GpuInfo& gpu_info,
                                    const TensorDescriptor& src_desc,
                                    const TensorDescriptor& dst_desc) {
  GPUOperation op;
  op.args_.AddObjectRef("src_tensor", AccessType::READ,
                        std::make_unique<TensorDescriptor>(src_desc));
  op.args_.AddObjectRef("dst_tensor", AccessType::WRITE,
                        std::make_unique<TensorDescriptor>(dst_desc));
  op.code_ +=
      R"(MAIN_FUNCTION($0) {
  int linear_id = get_global_id(0);
  int x = linear_id / args.dst_tensor.Batch();
  int b = linear_id % args.dst_tensor.Batch();
  int y = get_global_id(1);
  int d = get_global_id(2);
  if (x >= args.dst_tensor.Width() || y >= args.dst_tensor.Height() || d >= args.dst_tensor.Slices()) return;
  args.src_tensor::type in_value = args.src_tensor.Read(x, y, d, b);
)";
  const std::string conversion = GetTypeConversion(
      gpu_info, src_desc.GetDataType(), dst_desc.GetDataType(), 4);
  op.code_ += "  args.dst_tensor::type out_value = " +
              absl::Substitute(conversion, "in_value") + ";\n";
  op.code_ += "args.dst_tensor.Write(out_value, x, y, d, b);\n";
  op.code_ += "}\n";
  return op;
}

GPUOperation CreateTensorToBhwcBufferOp(const GpuInfo& gpu_info,
                                        const TensorDescriptor& src_desc,
                                        const BufferDescriptor& dst_desc) {
  GPUOperation op;
  op.args_.AddObjectRef("tensor", AccessType::READ,
                        std::make_unique<TensorDescriptor>(src_desc));
  op.args_.AddObjectRef("buffer", AccessType::WRITE,
                        std::make_unique<BufferDescriptor>(dst_desc));

  op.code_ += R"(MAIN_FUNCTION($0) {
  int linear_id = get_global_id(0);
  int x = linear_id / args.tensor.Batch();
  int b = linear_id % args.tensor.Batch();
  int y = get_global_id(1);
  int d = get_global_id(2);
  if (x >= args.tensor.Width() || y >= args.tensor.Height() || d >= args.tensor.Slices()) return;
  args.tensor::type in_value = args.tensor.Read(x, y, d, b);)";
  const std::string conversion = GetTypeConversion(
      gpu_info, src_desc.GetDataType(), dst_desc.element_type, 4);
  op.code_ += "  " + GetTypeDeclaration(gpu_info, dst_desc.element_type, 4) +
              " out_value = " + absl::Substitute(conversion, "in_value") +
              ";\n";
  op.code_ += R"(
  int c = d * 4;
  int index = ((b * args.tensor.Height() + y) * args.tensor.Width() + x) * args.tensor.Channels() + c;

  args.buffer.Write(out_value.x, index);
  if (c + 1 < args.tensor.Channels()) {
    args.buffer.Write(out_value.y, index + 1);
  }
  if (c + 2 < args.tensor.Channels()) {
    args.buffer.Write(out_value.z, index + 2);
  }
  if (c + 3 < args.tensor.Channels()) {
    args.buffer.Write(out_value.w, index + 3);
  }
})";
  return op;
}

GPUOperation CreateBhwcBufferToTensorOp(const GpuInfo& gpu_info,
                                        const BufferDescriptor& src_desc,
                                        const TensorDescriptor& dst_desc) {
  GPUOperation op;
  op.args_.AddObjectRef("buffer", AccessType::READ,
                        std::make_unique<BufferDescriptor>(src_desc));
  op.args_.AddObjectRef("tensor", AccessType::WRITE,
                        std::make_unique<TensorDescriptor>(dst_desc));

  op.code_ += R"(MAIN_FUNCTION($0) {
  int linear_id = get_global_id(0);
  int x = linear_id / args.tensor.Batch();
  int b = linear_id % args.tensor.Batch();
  int y = get_global_id(1);
  int d = get_global_id(2);

  if (x >= args.tensor.Width() || y >= args.tensor.Height() || d >= args.tensor.Slices()) return;
  int c = d * 4;
  int index = ((b * args.tensor.Height() + y) * args.tensor.Width() + x) * args.tensor.Channels() + c;
  args.tensor::type result;
  result.x = args.buffer.Read(index);
  result.y = c + 1 < args.tensor.Channels() ? args.buffer.Read(index + 1) : 1;
  result.z = c + 2 < args.tensor.Channels() ? args.buffer.Read(index + 2) : 2;
  result.w = c + 3 < args.tensor.Channels() ? args.buffer.Read(index + 3) : 3;
  args.tensor.Write(result, x, y, d, b);
})";
  return op;
}

}  // namespace gpu
}  // namespace tflite
