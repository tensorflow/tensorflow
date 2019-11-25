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

#include <memory>
#include <vector>

#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/refcount.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

class GetCalibrationDataOp : public OpKernel {
 public:
  explicit GetCalibrationDataOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  ~GetCalibrationDataOp() override {}

  void Compute(OpKernelContext* context) override {
    // TODO(laigd): it will allocate the tensor on the device and copy the
    // serialized string to that tensor, and later sess.run() will copy it back
    // to host. We need to optimize this.

    const string& resource_name = context->input(0).scalar<tstring>()();
    // Get the resource.
    TRTEngineCacheResource* resource = nullptr;
    OP_REQUIRES_OK(context, context->resource_manager()->Lookup(
                                std::string(kTfTrtContainerName), resource_name,
                                &resource));
    core::ScopedUnref sc(resource);

    // Serialize the resource as output.
    string serialized_resource = resource->calib_ctx_->TerminateCalibration();
    OP_REQUIRES(context, !serialized_resource.empty(),
                errors::Unknown("Calibration table is empty."));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));

    output->scalar<tstring>()() = serialized_resource;
  }
};

REGISTER_KERNEL_BUILDER(Name("GetCalibrationDataOp").Device(DEVICE_GPU),
                        GetCalibrationDataOp);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
