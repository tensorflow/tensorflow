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

#ifndef GDR_WORKER_H_
#define GDR_WORKER_H_

#include "tensorflow/contrib/gdr/gdr_memory_manager.h"

#include "tensorflow/core/distributed_runtime/recent_request_ids.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"

namespace tensorflow {

class GdrWorker : public GrpcWorker {
 public:
  GdrWorker(WorkerEnv* env, RemoteMemoryManager* remote_memory_manager);

  // Serve the RecvTensorRequest but omit the tensor content and transmit it
  // out-of-band using GPU Direct RDMA whenever possible.
  // If it's not possible, it falls back to gRPC in-band tensor transport by
  // encoding the tensor content into the grpc::ByteBuffer.
  // The RecvTensorResponse will carry the necessary information for RDMA.
  virtual void GrpcRecvTensorAsync(CallOptions* opts,
                                   const RecvTensorRequest* request,
                                   ::grpc::ByteBuffer* response,
                                   StatusCallback done) override;

 private:
  RemoteMemoryManager* remote_memory_manager_;  // Not owned
  RecentRequestIds recv_tensor_recent_request_ids_;
};

}  // namespace tensorflow

#endif  // GDR_WORKER_H_
