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

#ifndef TENSORFLOW_CONTRIB_GDR_GDR_MEMORY_MANAGER_H_
#define TENSORFLOW_CONTRIB_GDR_GDR_MEMORY_MANAGER_H_

#include "google/protobuf/any.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class Device;
class DeviceContext;
class Tensor;

// Abstract interface that handles out-of-band tensor transport.
//
// The transport options are encoded into a protocol buffer and transmitted via
// some other communication channels like RPC.
// See RecvTensorRequest in tensorflow/core/protobuf/worker.proto
class RemoteMemoryManager {
 public:
  virtual ~RemoteMemoryManager() {}
  virtual Status Init() = 0;
  virtual void Run() = 0;
  virtual void Stop() = 0;

  // Encodes the tensor information to an arbitrary protocol buffer
  // The protocol buffer needs to be transmitted via some other channel
  virtual void TransportOptionsFromTensor(
      ::google::protobuf::Any* mutable_transport_options, const Tensor& tensor,
      Device* device, DeviceContext* device_context, bool on_host,
      StatusCallback done) = 0;

  // Retrieve the tensor from the encoded protocol buffer
  // Note that the tensor has to be allocated, but not initialized
  virtual void TensorFromTransportOptions(
      Tensor* tensor, const ::google::protobuf::Any& transport_options,
      Device* device, DeviceContext* device_context, bool on_host,
      StatusCallback done) = 0;
};

RemoteMemoryManager* CreateRemoteMemoryManager(const string& host,
                                               const string& port);

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_GDR_GDR_MEMORY_MANAGER_H_
