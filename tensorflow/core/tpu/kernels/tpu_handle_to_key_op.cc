/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_op_consts.h"
#include "tensorflow/core/tpu/tpu_configuration.h"

namespace tensorflow {

class TpuHandleToProtoKeyOp : public OpKernel {
 public:
  explicit TpuHandleToProtoKeyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~TpuHandleToProtoKeyOp() override = default;
  TpuHandleToProtoKeyOp(const TpuHandleToProtoKeyOp&) = delete;
  TpuHandleToProtoKeyOp& operator=(const TpuHandleToProtoKeyOp&) = delete;

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "TpuHandleToProtoKeyOp::Compute " << ctx->op_kernel().name()
            << " on device " << ctx->op_kernel().requested_device();
    const Tensor& uid = ctx->input(0);

    ResourceMgr* rm = GetTPUConfigResourceMgr();
    tpu::TpuCompilationCacheInterface* cache;
    OP_REQUIRES_OK(ctx, rm->Lookup<tpu::TpuCompilationCacheInterface>(
                            rm->default_container(),
                            tpu::kCompilationCacheResourceName, &cache));
    core::ScopedUnref cache_unref(cache);

    std::vector<std::string> keys;
    OP_REQUIRES_OK(ctx, cache->GetKeysFromUid(uid.scalar<int64>()(), &keys));

    TensorShape output_shape;
    output_shape.AddDim(keys.size());
    Tensor* result = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &result));
    for (int i = 0; i < keys.size(); ++i) {
      result->vec<tstring>()(i) = keys[i];
    }
  };
};

REGISTER_KERNEL_BUILDER(Name("TpuHandleToProtoKey").Device(DEVICE_CPU),
                        TpuHandleToProtoKeyOp);

}  // namespace tensorflow
