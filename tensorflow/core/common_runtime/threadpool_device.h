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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_THREADPOOL_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_THREADPOOL_DEVICE_H_

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/node_file_writer.h"

namespace tensorflow {

// CPU device implementation.
class ThreadPoolDevice : public LocalDevice {
 public:
  ThreadPoolDevice(const SessionOptions& options, const string& name,
                   Bytes memory_limit, const DeviceLocality& locality,
                   Allocator* allocator);
  ~ThreadPoolDevice() override;

  Allocator* GetAllocator(AllocatorAttributes attr) override;
  Allocator* GetScopedAllocator(AllocatorAttributes attr,
                                int64_t step_id) override;
  ScopedAllocatorMgr* GetScopedAllocatorMgr() const override {
    return scoped_allocator_mgr_.get();
  }
  absl::Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                   const AllocatorAttributes alloc_attrs,
                                   Tensor* tensor) override;
  void CopyTensorInSameDevice(const Tensor* input_tensor, Tensor* output_tensor,
                              const DeviceContext* device_context,
                              StatusCallback done) override;

  absl::Status Sync() override { return absl::OkStatus(); }

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
  void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;

 private:
  void LogInputs(OpKernel* op_kernel, OpKernelContext* context);
  void LogOutputs(OpKernel* op_kernel, OpKernelContext* context);

  Allocator* allocator_;  // Not owned
  std::unique_ptr<ScopedAllocatorMgr> scoped_allocator_mgr_;
  NodeFileWriter* node_file_writer_ = nullptr;  // not owned
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_THREADPOOL_DEVICE_H_
