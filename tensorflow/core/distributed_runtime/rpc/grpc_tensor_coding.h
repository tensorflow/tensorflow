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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_TENSOR_CODING_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_TENSOR_CODING_H_

#include "grpcpp/impl/codegen/byte_buffer.h"
#include "absl/status/status.h"

namespace tensorflow {
class Tensor;
class RecvTensorResponse;

// TODO(jeff,sanjay): this should not be grpc specific.  Instead of
// grpc::ByteBuffer*, it should accept an object of an interface type
// to which owned byte-arrays can be added.
namespace grpc {

// Encode a RecvTensorResponse protocol buffer into a byte buffer in a
// format that is parseable as a RecvTensorResponse protocol buffer
// holding "proto".
//
// Discards original contents of *result.
void EncodeRecvTensorResponseToByteBuffer(const RecvTensorResponse& proto,
                                          ::grpc::ByteBuffer* result);

// Encode a Tensor into a byte buffer in a format that is parseable
// as a RecvTensorResponse protocol buffer holding "val".
//
// "is_dead" is the value to encode for "RecvTensorResponse::is_dead"
// (tensor is the output of a dead node and content is invalid because
// control flow operations elsewhere caused the path on which this
// Tensor exists to not be taken).
//
// "val" holds the tensor value to be encoded.
//
// Discards original contents of *result.
absl::Status EncodeTensorToByteBuffer(bool is_dead, const Tensor& val,
                                      bool require_ack,
                                      ::grpc::ByteBuffer* result);

}  // namespace grpc
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_TENSOR_CODING_H_
