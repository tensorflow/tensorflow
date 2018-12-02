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

// RpcOp is a TensorFlow op that sends and receives arbitrary messages.
//
// See docs in ../ops/rpc_op.cc.

#include <memory>
#include <string>
#include <vector>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/rpc/call_container.h"
#include "tensorflow/core/util/rpc/rpc_factory.h"
#include "tensorflow/core/util/rpc/rpc_factory_registry.h"

namespace tensorflow {

class RpcOp : public AsyncOpKernel {
 public:
  explicit RpcOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("protocol", &protocol_));
    OP_REQUIRES(context, !protocol_.empty(),
                errors::InvalidArgument("protocol must be non-empty."));
    bool fail_fast;
    OP_REQUIRES_OK(context, context->GetAttr("fail_fast", &fail_fast));
    int64 timeout_in_ms;
    OP_REQUIRES_OK(context, context->GetAttr("timeout_in_ms", &timeout_in_ms));

    RPCFactoryRegistry::RPCFactoryFn* rpc_factory_fn =
        RPCFactoryRegistry::Global()->Get(protocol_);
    OP_REQUIRES(context, rpc_factory_fn != nullptr,
                errors::InvalidArgument("The protocol ", protocol_,
                                        " was not recognized."));

    rpc_factory_.reset((*rpc_factory_fn)(context, fail_fast, timeout_in_ms));
  }

  ~RpcOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const Tensor& address_t = ctx->input(0);
    const Tensor& method_t = ctx->input(1);
    const Tensor& request_t = ctx->input(2);

    OP_REQUIRES_ASYNC(
        ctx, address_t.dims() == 0 || address_t.dims() == 1,
        errors::InvalidArgument("address must be a scalar or vector."), done);
    OP_REQUIRES_ASYNC(
        ctx, method_t.dims() == 0 || method_t.dims() == 1,
        errors::InvalidArgument("method must be a scalar or vector."), done);
    OP_REQUIRES_ASYNC(
        ctx, request_t.dims() == 0 || request_t.dims() == 1,
        errors::InvalidArgument("request must be a scalar or vector."), done);

    TensorShape output_shape({});
    for (const Tensor& t : {address_t, method_t, request_t}) {
      if (t.dims() == 1) {
        OP_REQUIRES_ASYNC(
            ctx,
            output_shape.dims() == 0 ||
                output_shape.dim_size(0) == t.dim_size(0),
            errors::InvalidArgument(
                "Input vector shapes don't match: ", output_shape.DebugString(),
                " vs. ", t.shape().DebugString()),
            done);
        output_shape = t.shape();
      }
    }

    Tensor* response_t;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, output_shape, &response_t), done);

    const bool try_rpc = (ctx->num_outputs() > 1);

    Tensor* status_code_t = nullptr;
    Tensor* status_message_t = nullptr;
    if (try_rpc) {
      OP_REQUIRES_OK_ASYNC(
          ctx, ctx->allocate_output(1, output_shape, &status_code_t), done);
      OP_REQUIRES_OK_ASYNC(
          ctx, ctx->allocate_output(2, output_shape, &status_message_t), done);
    }

    if (request_t.NumElements() == 0) {
      // Special case, we finished early!
      done();
      return;
    }

    int64 num_elements = output_shape.num_elements();

    rpc_factory_->Call(ctx, num_elements, address_t, method_t, request_t,
                       try_rpc, response_t, status_code_t, status_message_t,
                       std::move(done));
  }

 private:
  string protocol_;
  std::unique_ptr<RPCFactory> rpc_factory_;

  TF_DISALLOW_COPY_AND_ASSIGN(RpcOp);
};

REGISTER_KERNEL_BUILDER(Name("Rpc").Device(DEVICE_CPU), RpcOp);
REGISTER_KERNEL_BUILDER(Name("TryRpc").Device(DEVICE_CPU), RpcOp);

}  // namespace tensorflow
