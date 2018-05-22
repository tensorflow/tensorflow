/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMMON_RUNTIME_NUMA_DEVICE_H_
#define TENSORFLOW_COMMON_RUNTIME_NUMA_DEVICE_H_

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"

namespace tensorflow {

// NUMA CPU device implementation.
class NumaDevice : public LocalDevice {
 public:
  NumaDevice(const SessionOptions& options, const string& name,
                   Bytes memory_limit, const DeviceLocality& locality,
                   Allocator* allocator, std::vector<int> &proc_set);
  ~NumaDevice() override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
  Allocator* GetAllocator(AllocatorAttributes attr) override;
  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  Status Sync() override { return Status::OK(); }

  typedef enum {NoTask, RunTask, Cancel} RunStatus;

  struct RunState {
    std::atomic<RunStatus>         status_;
    OpKernel         *op_kernel_;
    OpKernelContext  *context_;
    Notification     *notification_;
    std::vector<int>  proc_set_;
  };

  RunState   run_state_;

 private:
  Allocator  *allocator_;  // Not owned
  pthread_t   task_thread_;
  unsigned long memory;
  mutex       mutex_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_NUMA_DEVICE_H_
