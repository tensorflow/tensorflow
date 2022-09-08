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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/sharding_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {

namespace {

xla::Shape GetTPUInfeedLayout(const xla::Shape& shape) {
  XLA_Shape c_shape;
  XLA_Shape c_infeed_shape;

  ApiConverter::ToC(shape, &c_shape);

  tpu::ExecutorApiFn()->TpuTransferManager_GetInfeedLayoutFn(&c_shape,
                                                             &c_infeed_shape);
  xla::Shape infeed_shape = ApiConverter::FromC(&c_infeed_shape);
  ApiConverter::Destroy(&c_shape);
  ApiConverter::Destroy(&c_infeed_shape);
  return infeed_shape;
}

// Updates the layout of the given infeed shape, optionally considering the
// sharding of the op. If the op has tile sharding, assign the layout based on
// the shard shape.
Status UpdateInfeedLayout(xla::Shape* shape,
                          absl::optional<xla::OpSharding> sharding) {
  if (sharding && sharding->type() == xla::OpSharding::OTHER) {
    TF_ASSIGN_OR_RETURN(auto hlo_sharding,
                        xla::HloSharding::FromProto(*sharding));
    for (int64_t i = 0; i < sharding->tile_assignment_devices_size(); ++i) {
      auto device = sharding->tile_assignment_devices(i);
      auto shard_shape =
          GetTPUInfeedLayout(hlo_sharding.TileShape(*shape, device));
      if (i == 0) {
        *shape->mutable_layout() = shard_shape.layout();
      }
      if (xla::ShapeUtil::ElementsIn(shard_shape) == 0) {
        // Shapes with 0 dimensions may be assigned with a different layout, but
        // it doesn't matter since we're not sending any data.
        continue;
      }
      if (!xla::LayoutUtil::Equal(shard_shape.layout(), shape->layout())) {
        return xla::Unimplemented(
            "Sharded infeed with non-uniform layouts is not supported. Try "
            "turning off the infeed layout optimization "
            "(--transpose_tpu_infeed=false) and report to XLA team.");
      }
    }
    return OkStatus();
  }
  *shape = GetTPUInfeedLayout(*shape);
  return OkStatus();
}

// TODO(pbar) Work out if we need to Infeed Tuples - if so then
// this op will need a way to provide a list of shapes
// since they can't be provided by the runtime JIT mechanism.
// (InfeedDequeue has no inputs!)
// Compare this op to tf.Queue operations which operate on N tensors.

// This TensorFlow op supports the XLA Infeed primitve.
class InfeedDequeueOp : public XlaOpKernel {
 public:
  explicit InfeedDequeueOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype_, shape_, &xla_shape_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    OP_REQUIRES_OK(ctx, UpdateInfeedLayout(&xla_shape_, b->sharding()));
    ctx->SetOutput(0, xla::Infeed(b, xla_shape_));
  }

 private:
  TensorShape shape_;
  DataType dtype_;
  xla::Shape xla_shape_;

  TF_DISALLOW_COPY_AND_ASSIGN(InfeedDequeueOp);
};

REGISTER_XLA_OP(Name("InfeedDequeue"), InfeedDequeueOp);

// This TensorFlow op supports the XLA Infeed primitive for tuple types.
class InfeedDequeueTupleOp : public XlaOpKernel {
 public:
  explicit InfeedDequeueTupleOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
    for (int i = 0; i < shapes_.size(); i++) {
      xla::Shape xla_shape;
      OP_REQUIRES_OK(ctx,
                     TensorShapeToXLAShape(dtypes_[i], shapes_[i], &xla_shape));
      xla_shapes_.push_back(xla_shape);
    }
  }

  ~InfeedDequeueTupleOp() override {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    for (int64_t i = 0; i < xla_shapes_.size(); ++i) {
      absl::optional<xla::OpSharding> sharding;
      if (b->sharding()) {
        sharding = b->sharding()->type() == xla::OpSharding::TUPLE
                       ? b->sharding()->tuple_shardings(i)
                       : b->sharding();
      }
      OP_REQUIRES_OK(ctx, UpdateInfeedLayout(&xla_shapes_[i], sharding));
    }
    tuple_shape_ = xla::ShapeUtil::MakeTupleShape(xla_shapes_);
    auto tuple = xla::Infeed(b, tuple_shape_);

    // Don't apply the infeed tuple sharding to the get-tuple-elements. They
    // need non-tuple shardings.
    xla::XlaScopedShardingAssignment clear_sharding(b, absl::nullopt);
    for (int i = 0; i < shapes_.size(); ++i) {
      ctx->SetOutput(i, xla::GetTupleElement(tuple, i));
    }
  }

 private:
  std::vector<TensorShape> shapes_;
  DataTypeVector dtypes_;
  std::vector<xla::Shape> xla_shapes_;
  xla::Shape tuple_shape_;

  TF_DISALLOW_COPY_AND_ASSIGN(InfeedDequeueTupleOp);
};

REGISTER_XLA_OP(Name("InfeedDequeueTuple"), InfeedDequeueTupleOp);

}  // anonymous namespace
}  // namespace tensorflow
