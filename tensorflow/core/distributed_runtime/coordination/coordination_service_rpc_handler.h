/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_RPC_HANDLER_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_RPC_HANDLER_H_

#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"

namespace tensorflow {
class CoordinationServiceAgent;

class CoordinationServiceRpcHandler {
 public:
  explicit CoordinationServiceRpcHandler() {}

  void SetAgentInstance(CoordinationServiceAgent* agent);

  void RegisterWorkerAsync(const RegisterWorkerRequest* request,
                           RegisterWorkerResponse* response,
                           StatusCallback done);

  void HeartbeatAsync(const HeartbeatRequest* request,
                      HeartbeatResponse* response, StatusCallback done);

  void WaitForAllTasksAsync(const WaitForAllTasksRequest* request,
                            WaitForAllTasksResponse* response,
                            StatusCallback done);

  void ReportErrorToAgentAsync(const ReportErrorToAgentRequest* request,
                               ReportErrorToAgentResponse* response,
                               StatusCallback done);

  void ReportErrorToServiceAsync(const ReportErrorToServiceRequest* request,
                                 ReportErrorToServiceResponse* response,
                                 StatusCallback done);

  void InsertKeyValueAsync(const InsertKeyValueRequest* request,
                           InsertKeyValueResponse* response,
                           StatusCallback done);

  void GetKeyValueAsync(const GetKeyValueRequest* request,
                        GetKeyValueResponse* response, StatusCallback done);

  void DeleteKeyValueAsync(const DeleteKeyValueRequest* request,
                           DeleteKeyValueResponse* response,
                           StatusCallback done);

 private:
  const int64_t leader_incarnation_id_ = random::New64();
  CoordinationServiceAgent* agent_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_COORDINATION_COORDINATION_SERVICE_RPC_HANDLER_H_
