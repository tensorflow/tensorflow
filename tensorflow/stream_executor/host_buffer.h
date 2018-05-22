/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_HOST_BUFFER_H_
#define TENSORFLOW_STREAM_EXECUTOR_HOST_BUFFER_H_

#include "tensorflow/stream_executor/dnn.h"

namespace stream_executor {

// A HostBuffer is a block of memory in host memory containing the data for a
// dnn::BatchDescriptor using a device-dependent memory layout.
// Derived classes provide methods to construct a HostBuffer for a specific
// device, and to copy data in and out of the buffer.
class HostBuffer {
 public:
  const dnn::BatchDescriptor& descriptor() const { return descriptor_; }

  // Returns a string describing the HostBuffer.
  virtual string AsString() const = 0;

 protected:
  // Construct a HostBuffer from the supplied dnn::BatchDescriptor.
  explicit HostBuffer(const dnn::BatchDescriptor& descriptor)
      : descriptor_(descriptor) {}
  virtual ~HostBuffer() {}

 private:
  const dnn::BatchDescriptor descriptor_;
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_HOST_BUFFER_H_
