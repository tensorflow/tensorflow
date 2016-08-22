/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_TENSOR_CODING_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_TENSOR_CODING_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

class Allocator;
class DeviceBase;
class TensorProto;

// TensorResponse can be used as the destination of an RPC that returns
// a RecvTensorResponse.  It efficiently decodes the incoming data
// into Tensor contents as well as associated metadata.
class TensorResponse {
 public:
  TensorResponse() {}

  // Reset to initial state.
  void Clear();

  // Clear just tensor_ and meta_ members without setting allocation
  // related members.
  void ClearTensor();

  // Initialize memory allocation related members.
  void InitAlloc(DeviceBase* d, const AllocatorAttributes& aa);

  // Source provides a way for a particular RPC implementation to provide
  // received data to ParseFrom.
  class Source {
   public:
    virtual ~Source();

    // Return the stream that contains the data to be parsed.
    // Note that this method might be invoked more than once if
    // ParseFrom needs to fall back to a more expensive parsing method.
    // Every call must return a stream pointing at the beginning of
    // the serialized RecvTensorResponse.
    //
    // Note that a subsequent call to contents() invalidates previous
    // results of contents().
    //
    // Ownership of the returned stream is retained by the Source and
    // should not be deleted by the caller.
    virtual ::tensorflow::protobuf::io::ZeroCopyInputStream* contents() = 0;
  };

  // Parse the RecvTensorResponse encoded in the data yielded by
  // source->contents() into *this.
  Status ParseFrom(Source* source);

  // Initialize tensor from *response.
  // Leaves *response with unspecified contents.
  Status InitFrom(RecvTensorResponse* response);

  // Initialize tensor metadata from response and allocate
  // uninitialized backing storage for actual contents.
  void InitPartial(const RecvTensorResponse& response);

  // Return a reference to the parsed tensor.  The tensor will remain
  // live only until *this is destroyed or modified.
  const Tensor& tensor() const { return tensor_; }

  // Return a reference to the parsed tensor metadata (no contents).
  // The result will remain live only until *this is destroyed or
  // modified.
  const RecvTensorResponse& metadata() const { return meta_; }

 private:
  bool ParseTensorSubmessage(protobuf::io::CodedInputStream* input,
                             TensorProto* tensor_meta);
  bool ParseFast(Source* source);
  bool ParseSlow(Source* source);

  bool on_host_ = false;
  DeviceBase* device_ = nullptr;
  AllocatorAttributes alloc_attrs_;
  Allocator* allocator_ = nullptr;
  bool already_used_ = false;
  Tensor tensor_;
  RecvTensorResponse meta_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_TENSOR_CODING_H_
