/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_JIT_XLA_TENSOR_H_
#define TENSORFLOW_COMPILER_JIT_XLA_TENSOR_H_

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// The implementation of a Tensor for an XlaDevice. All device tensors are
// actually one of these.
//
// To distinguish between "normal" device tensors and XlaTensors, the raw
// pointer data stored in the TensorBuffer is a tagged pointer.
class XlaTensor {
 public:
  // Downcast from a Tensor to an XlaTensor. Return nullptr if the downcast
  // fails.
  static XlaTensor* FromTensor(const Tensor* tensor);

  static bool RefCountIsOne(const Tensor& tensor);

  // Create a DeviceMemoryBase from a Tensor. The Tensor can be an XlaTensor, in
  // which case the returned value is shaped_buffer()->root_buffer(), or a
  // normal Tensor in which case the returned value is
  // {tensor.tensor_data().data(), tensor.tensor_data().size}.
  static se::DeviceMemoryBase DeviceMemoryFromTensor(const Tensor& tensor);

  // Assign the internal ShapedBuffer to new memory for the given dtype and
  // shape. If a ShapedBuffer exists already (has_shaped_buffer() == true), it
  // is replaced and the managed memory deallocated.
  Status AllocateShapedBuffer(DataType dtype, const TensorShape& shape,
                              xla::LocalClient* client, int device_ordinal);

  // Some Tensors can have complex on-device shapes, including tuple shapes. To
  // manage the memory for these tensors a ShapedBuffer may be required.

  // Return true if this XlaTensor contains a ShapedBuffer.
  bool has_shaped_buffer() const { return shaped_buffer_ != nullptr; }
  // Return the contained ShapedBuffer.
  // REQUIRES: has_shaped_buffer()
  const xla::ShapedBuffer& shaped_buffer() const {
    CHECK(has_shaped_buffer());
    return *shaped_buffer_;
  }
  xla::ShapedBuffer& shaped_buffer() {
    CHECK(has_shaped_buffer());
    return *shaped_buffer_;
  }
  // Mutates the XlaTensor to set the ShapedBuffer.
  void set_shaped_buffer(xla::ScopedShapedBuffer shaped_buffer) {
    shaped_buffer_ =
        xla::MakeUnique<xla::ScopedShapedBuffer>(std::move(shaped_buffer));
  }

  // Some tensors on the device may have known values on the host. We use these
  // in on-demand mode to avoid re-copying values from the device if we know the
  // host value already.

  // Return true if this XlaTensor contains a host tensor.
  bool has_host_tensor() const { return host_tensor_ != nullptr; }
  // Return the contained host tensor.
  // REQUIRES: has_host_tensor()
  const Tensor& host_tensor() const { return *host_tensor_; }
  // Sets the contained host tensor.
  void set_host_tensor(const Tensor& tensor) {
    host_tensor_.reset(new Tensor(tensor));
  }

  // Convert from a raw pointer to an XlaTensor, removing the pointer tag.
  static XlaTensor* FromOpaquePointer(void* ptr);
  // Convert to a raw pointer from an XlaTensor, adding the pointer tag.
  static void* ToOpaquePointer(XlaTensor* tensor);

 private:
  // The optional contained ShapedBuffer.
  std::unique_ptr<xla::ScopedShapedBuffer> shaped_buffer_;
  // An optional host tensor value.
  std::unique_ptr<Tensor> host_tensor_;
};

}  // namespace tensorflow

#endif
