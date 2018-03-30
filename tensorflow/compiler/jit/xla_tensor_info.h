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

#ifndef TENSORFLOW_COMPILER_JIT_XLA_TENSOR_INFO_H_
#define TENSORFLOW_COMPILER_JIT_XLA_TENSOR_INFO_H_

#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// Information about a tensor. The XlaTensorInfoManager can maintain one of
// these per device Tensor.
class XlaTensorInfo {
 public:
  XlaTensorInfo() {}

  // Some Tensors can have complex on-device shapes, including tuple shapes. To
  // manage the memory for these tensors a ShapedBuffer may be required.

  // Return true if this TensorInfo contains a ShapedBuffer.
  bool has_shaped_buffer() const { return shaped_buffer_ != nullptr; }
  // Return the contained ShapedBuffer.
  // REQUIRES: has_shaped_buffer()
  const xla::ShapedBuffer& shaped_buffer() const { return *shaped_buffer_; }
  // Mutates the TensorInfo to set the ShapedBuffer.
  void set_shaped_buffer(xla::ShapedBuffer shaped_buffer) {
    shaped_buffer_.reset(new xla::ShapedBuffer(std::move(shaped_buffer)));
  }

  // Some tensors on the device may have known values on the host. We use these
  // in on-demand mode to avoid re-copying values from the device if we know the
  // host value already.

  // Return true if this TensorInfo contains a host tensor.
  bool has_host_tensor() const { return host_tensor_ != nullptr; }
  // Return the contained host tensor.
  // REQUIRES: has_host_tensor()
  const Tensor& host_tensor() const { return *host_tensor_; }
  // Sets the contained host tensor.
  void set_host_tensor(const Tensor& tensor) {
    host_tensor_.reset(new Tensor(tensor));
  }

 private:
  // The optional contained ShapedBuffer.
  std::unique_ptr<xla::ShapedBuffer> shaped_buffer_;
  // An optional host tensor value.
  std::unique_ptr<Tensor> host_tensor_;
};

// Manages XlaTensorInfo objects. This class is also an Allocator, so that
// XlaTensorInfo objects can be deleted when their Tensor is deallocated.
class XlaTensorInfoManager : public AllocatorWrapper {
 public:
  // Creates a new XlaTensorInfoManager, delegating all DeallocateRaw calls to
  // allocator.
  XlaTensorInfoManager(Allocator* allocator) : AllocatorWrapper(allocator) {}
  ~XlaTensorInfoManager() {
    // Destroy the tensor info hashtable under the lock, to ensure all accesses
    // to the hashtable are properly sequenced.
    mutex_lock lock(lock_);
    tensor_infos_.clear();
  }

  // Returns the XlaTensorInfo for the given device memory pointer or nullptr if
  // none exists.
  const XlaTensorInfo* GetTensorInfo(const void* device_ptr) const;
  // Returns the XlaTensorInfo for the device memory pointer extracted from
  // tensor or nullptr if none exists.
  const XlaTensorInfo* GetTensorInfo(const Tensor& tensor);

  // Returns the XlaTensorInfo for the given device memory pointer, creating one
  // if necessary.
  XlaTensorInfo* GetOrCreateTensorInfo(const Tensor& tensor);
  // Returns the XlaTensorInfo for the device memory pointer extracted from
  // tensor, creating one if necessary.
  XlaTensorInfo* GetOrCreateTensorInfo(const void* device_ptr);

  // Allocator interface
  void DeallocateRaw(void* ptr) override;

 private:
  mutable mutex lock_;
  // The managed tensor infos. The mapped value is a unique_ptr so that returned
  // references are stable over rehashes.
  std::unordered_map<const void*, std::unique_ptr<XlaTensorInfo>> tensor_infos_
      GUARDED_BY(lock_);
};
}  // namespace tensorflow

#endif
