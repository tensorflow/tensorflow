/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_EVALUATION_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_EVALUATION_UTILS_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace Eigen {
class ThreadPoolInterface;
class ThreadPoolWrapper;
}  // namespace Eigen

namespace tensorflow {
namespace grappler {

class DeviceSimple : public DeviceBase {
 public:
  DeviceSimple();
  ~DeviceSimple();

  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    return cpu_allocator();
  }

 private:
  DeviceBase::CpuWorkerThreads eigen_worker_threads_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_device_;
};

Status EvaluateNode(const NodeDef& node,
                    const gtl::InlinedVector<TensorValue, 4>& inputs,
                    DeviceBase* cpu_device, ResourceMgr* resource_mgr,
                    gtl::InlinedVector<TensorValue, 4>* output);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_EVALUATION_UTILS_H_
