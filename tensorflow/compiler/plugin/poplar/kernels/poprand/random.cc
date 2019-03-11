/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/stream_executor_util.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"

#include "absl/container/flat_hash_set.h"

using namespace xla::poplarplugin;

namespace tensorflow {

class PopopsTruncatedNormalOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopopsTruncatedNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    AddRequiredAttributesToMap();
  }

  ~PopopsTruncatedNormalOp() override{};
  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::XlaBuilder* b = ctx->builder();

    xla::XlaOp output =
        xla::CustomCall(b,
                        GetPoplibsCustomOpTargetString(
                            PoplibsOp::Poprand, PoplibsOp::TruncatedNormal),
                        {}, xla_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  const absl::flat_hash_set<int64> AllocatingIndexes() override { return {}; }

  const absl::flat_hash_map<int64, int64> LayoutDependencies() override {
    return {};
  };

  const uint64 NumberOfInplaceOperands() override { return 0; }

  TF_DISALLOW_COPY_AND_ASSIGN(PopopsTruncatedNormalOp);
};

REGISTER_XLA_OP(Name("TruncatedNormal")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", DT_HALF)
                    .TypeConstraint("dtype", DT_FLOAT),
                PopopsTruncatedNormalOp);

}  // namespace tensorflow
