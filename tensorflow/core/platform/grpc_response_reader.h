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

#ifndef TENSORFLOW_CORE_PLATFORM_GRPC_RESPONSE_READER_H_
#define TENSORFLOW_CORE_PLATFORM_GRPC_RESPONSE_READER_H_

#include "grpc++/grpc++.h"
#include "tensorflow/core/platform/platform.h"

// Include platform-dependent grpc ClientAsyncResponseReader constructors.
// TODO(b/62910646): Remove this level of indirection once this is resolved.
#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/grpc_response_reader.h"
#else
#include "tensorflow/core/platform/default/grpc_response_reader.h"
#endif

namespace tensorflow {

// Start a call and write the request out.
// The returned pointer is owned by the caller.
// See
// https://grpc.io/grpc/cpp/classgrpc_1_1_client_async_response_reader.html#ace2c5bae351f67dd7dd603fc39513e0a
// for more information.
template <class ResponseMessage, class RequestMessage>
::grpc::ClientAsyncResponseReader<ResponseMessage>*
CreateClientAsyncResponseReader(::grpc::ChannelInterface* channel,
                                ::grpc::CompletionQueue* cq,
                                const ::grpc::RpcMethod& method,
                                ::grpc::ClientContext* context,
                                const RequestMessage& request);

}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_MUTEX_H_
