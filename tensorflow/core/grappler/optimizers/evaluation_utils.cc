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

#include "tensorflow/core/grappler/optimizers/evaluation_utils.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/setround.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace grappler {
using TensorVector = absl::InlinedVector<TensorValue, 4UL>;

// In order to avoid the overhead of creating a large thread pool, we set a
// small default thread count. This value should be revised should DeviceSimple
// be used to evaluate nodes with a large degree of intra-op parallelism.
const int kDeviceSimpleThreads = 2;

DeviceSimple::DeviceSimple() : DeviceBase(Env::Default()) {
  eigen_worker_threads_.num_threads = kDeviceSimpleThreads;
  eigen_worker_threads_.workers = new thread::ThreadPool(
      Env::Default(), "evaluation_utils", eigen_worker_threads_.num_threads);
  eigen_device_.reset(new Eigen::ThreadPoolDevice(
      eigen_worker_threads_.workers->AsEigenThreadPool(),
      eigen_worker_threads_.num_threads));
  set_tensorflow_cpu_worker_threads(&eigen_worker_threads_);
  set_eigen_cpu_device(eigen_device_.get());
}

DeviceSimple::~DeviceSimple() {
  eigen_device_.reset();
  delete eigen_worker_threads_.workers;
}

absl::Status DeviceSimple::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from tensor_proto.");
  }
  *tensor = parsed;
  return absl::OkStatus();
}

absl::Status EvaluateNode(const NodeDef& node, const TensorVector& inputs,
                          DeviceBase* cpu_device, ResourceMgr* resource_mgr,
                          TensorVector* output) {
  absl::Status status;
  std::unique_ptr<DeviceBase> device;
  if (cpu_device == nullptr) {
    device.reset(new DeviceSimple());
    cpu_device = device.get();
  }

  std::unique_ptr<OpKernel> op_kernel(
      CreateOpKernel(DEVICE_CPU, cpu_device, cpu_device->GetAllocator({}), node,
                     TF_GRAPH_DEF_VERSION, &status));
  TF_RETURN_IF_ERROR(status);
  OpKernelContext::Params params;
  params.device = cpu_device;
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = inputs;
  params.op_kernel = op_kernel.get();
  params.resource_manager = resource_mgr;

  absl::InlinedVector<AllocatorAttributes, 4UL> output_attrs;
  const int num_outputs = op_kernel->num_outputs();
  for (int i = 0; i < num_outputs; i++) {
    AllocatorAttributes attr;
    attr.set_on_host(true);
    output_attrs.push_back(attr);
  }
  params.output_attr_array = output_attrs.data();

  OpKernelContext op_context(&params);
  op_kernel->Compute(&op_context);
  for (int i = 0; i < num_outputs; i++) {
    output->push_back(op_context.release_output(i));
  }
  return op_context.status();
}

}  // end namespace grappler
}  // end namespace tensorflow
