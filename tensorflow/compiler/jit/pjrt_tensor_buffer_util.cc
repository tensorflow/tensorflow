/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/pjrt_tensor_buffer_util.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/jit/pjrt_tensor_buffer.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace tensorflow {

// This method currently only supports PjRtBuffer implemented with
// StreamExecutor.
Tensor MakeTensorFromPjRtStreamExecutorBuffer(
    const DataType dtype, const TensorShape& shape,
    std::unique_ptr<xla::PjRtBuffer> pjrt_buffer) {
  const se::DeviceMemoryBase se_device_buffer =
      tensorflow::down_cast<xla::PjRtStreamExecutorBuffer*>(pjrt_buffer.get())
          ->AsShapedBuffer()
          ->root_buffer();
  const size_t expected_size = shape.num_elements() * DataTypeSize(dtype);
  auto* tensor_buffer = new PjRtTensorBuffer(
      se_device_buffer.opaque(), expected_size, std::move(pjrt_buffer));
  Tensor result(dtype, shape, tensor_buffer);
  tensor_buffer->Unref();
  return result;
}

// If output_tensor is using se::DeviceMemoryBase and the buffer is the same,
// the tensor should be reused so that the same device memory will not be
// double-freed.
static bool ShouldReuseTensor(const se::DeviceMemoryBase& output_device_memory,
                              const Tensor* existing_tensor) {
  const PjRtTensorBuffer* input_pjrt_tensor_buffer =
      dynamic_cast<const PjRtTensorBuffer*>(DMAHelper::buffer(existing_tensor));
  if (input_pjrt_tensor_buffer != nullptr) {
    return false;
  }
  se::DeviceMemoryBase input_device_memory = se::DeviceMemoryBase(
      const_cast<char*>(existing_tensor->tensor_data().data()),
      existing_tensor->tensor_data().size());
  return input_device_memory.IsSameAs(output_device_memory);
}

// This method currently only supports PjRtBuffer implemented with
// StreamExecutor.
void PjRtTensorBufferUtil::UpdateOrMakeTensorWithPjRtStreamExecutorBuffer(
    const DataType dtype, const TensorShape& shape,
    std::unique_ptr<xla::PjRtBuffer> pjrt_buffer, Tensor* output_tensor) {
  const se::DeviceMemoryBase se_device_buffer =
      tensorflow::down_cast<xla::PjRtStreamExecutorBuffer*>(pjrt_buffer.get())
          ->AsShapedBuffer()
          ->root_buffer();
  const size_t expected_size = shape.num_elements() * DataTypeSize(dtype);
  auto* tensor_buffer = new PjRtTensorBuffer(
      se_device_buffer.opaque(), expected_size, std::move(pjrt_buffer));

  if (ShouldReuseTensor(se_device_buffer, output_tensor)) {
    output_tensor->buf_ = tensor_buffer;
    return;
  }

  Tensor result(dtype, shape, tensor_buffer);
  tensor_buffer->Unref();
  *output_tensor = result;
}

}  // namespace tensorflow
