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

}  // namespace tensorflow
