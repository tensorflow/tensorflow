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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_PLATFORM_DEFAULT_GRPC_RESPONSE_READER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_PLATFORM_DEFAULT_GRPC_RESPONSE_READER_H_

#include "grpc++/grpc++.h"

namespace tensorflow {

template <class ResponseMessage, class RequestMessage>
::grpc::ClientAsyncResponseReader<ResponseMessage>*
CreateClientAsyncResponseReader(::grpc::ChannelInterface* channel,
                                ::grpc::CompletionQueue* cq,
                                const ::grpc::RpcMethod& method,
                                ::grpc::ClientContext* context,
                                const RequestMessage& request) {
  return new ::grpc::ClientAsyncResponseReader<ResponseMessage>(
      channel, cq, method, context, request);
}

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_PLATFORM_DEFAULT_GRPC_RESPONSE_READER_H_
