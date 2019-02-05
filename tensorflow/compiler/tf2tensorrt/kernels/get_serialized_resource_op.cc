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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_KERNELS_GET_SERIALIZED_RESOURCE_OP_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_KERNELS_GET_SERIALIZED_RESOURCE_OP_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/tf2tensorrt/utils/trt_resources.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/refcount.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

class GetSerializedResourceOp : public OpKernel {
 public:
  explicit GetSerializedResourceOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  ~GetSerializedResourceOp() override {}

  void Compute(OpKernelContext* context) override {
    // TODO(laigd): it will allocate the tensor on the device and copy the
    // serialized string to that tensor, and later sess.run() will copy it back
    // to host. We need to optimize this.
    const string& container = context->input(0).scalar<string>()();
    const string& resource_name = context->input(1).scalar<string>()();

    // Get the resource.
    SerializableResourceBase* resource = nullptr;
    OP_REQUIRES_OK(context, context->resource_manager()->Lookup(
                                container, resource_name, &resource));
    ::tensorflow::core::ScopedUnref sc(resource);

    // Serialize the resource as output.
    string serialized_resource;
    OP_REQUIRES_OK(context, resource->SerializeToString(&serialized_resource));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    output->scalar<string>()() = serialized_resource;
  }
};

REGISTER_KERNEL_BUILDER(Name("GetSerializedResourceOp").Device(DEVICE_GPU),
                        GetSerializedResourceOp);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_KERNELS_GET_SERIALIZED_RESOURCE_OP_H_
