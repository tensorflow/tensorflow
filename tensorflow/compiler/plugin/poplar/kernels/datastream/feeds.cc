/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/core/framework/dataset.h"
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
#include "tensorflow/compiler/xla/shape_util.h"

#include "absl/container/flat_hash_set.h"

namespace tensorflow {

namespace {
void XlaShapesFromAttr(OpKernelConstruction* ctx,
                       std::vector<xla::Shape>& result) {
  std::vector<TensorShape> shapes;
  std::vector<tensorflow::DataType> types;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &shapes));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &types));

  for (unsigned i = 0; i < shapes.size(); ++i) {
    xla::PrimitiveType xla_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(types[i], &xla_type));
    result.emplace_back(TensorShapeToXLAShape(xla_type, shapes[i]));
  }
}
}  // namespace

class PopDatastreamInfeedDequeueOp : public XlaOpKernel {
 public:
  explicit PopDatastreamInfeedDequeueOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("infeed_id", &infeed_id_));
    XlaShapesFromAttr(ctx, xla_shapes_);
  }

  ~PopDatastreamInfeedDequeueOp() override{};

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    auto tuple_shape = xla::ShapeUtil::MakeTupleShape(xla_shapes_);
    xla::XlaOp output_tuple = xla::Infeed(b, tuple_shape, infeed_id_);
    for (int i = 0; i < ctx->num_outputs(); ++i) {
      ctx->SetOutput(i, xla::GetTupleElement(output_tuple, i));
    }
  }

 private:
  std::vector<xla::Shape> xla_shapes_;
  std::string infeed_id_;
  TF_DISALLOW_COPY_AND_ASSIGN(PopDatastreamInfeedDequeueOp);
};

REGISTER_IPU_OP("PopDatastreamInfeedDequeue", PopDatastreamInfeedDequeueOp);

class IPUConsumeDatasetOp : public OpKernel {
 public:
  explicit IPUConsumeDatasetOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), device_ordinal_(0) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("id", &id_));

    OP_REQUIRES(ctx, device_ordinal_ >= 0,
                errors::InvalidArgument("Need device_ordinal >= 0, got ",
                                        device_ordinal_));
    XlaShapesFromAttr(ctx, xla_shapes_);
  }

  ~IPUConsumeDatasetOp() override{};

  void Compute(OpKernelContext* ctx) override {
    // Create a dataset iterator.
    DatasetBase* dataset;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
    std::unique_ptr<IteratorContext> iterator_ctx =
        absl::make_unique<IteratorContext>(ctx);
    std::unique_ptr<IteratorBase> iterator;
    OP_REQUIRES_OK(ctx, dataset->MakeIterator(iterator_ctx.get(),
                                              "IPUDatasetIterator", &iterator));
    // Pass to the correct executor.
    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());
    auto* p =
        static_cast<xla::poplarplugin::PoplarPlatform*>(platform.ValueOrDie());
    auto stream_executor = p->ExecutorForDevice(device_ordinal_).ValueOrDie();
    auto* poplar_executor = static_cast<xla::poplarplugin::PoplarExecutor*>(
        stream_executor->implementation());
    poplar_executor->CreateInfeedDatasetIterator(
        id_, std::move(iterator), std::move(iterator_ctx), xla_shapes_);
  }

 private:
  int device_ordinal_;
  std::string id_;
  std::vector<xla::Shape> xla_shapes_;
  TF_DISALLOW_COPY_AND_ASSIGN(IPUConsumeDatasetOp);
};

REGISTER_KERNEL_BUILDER(Name("IPUConsumeDataset").Device(DEVICE_CPU),
                        IPUConsumeDatasetOp);

class PopDatastreamOutfeedEnqueueOp : public XlaOpKernel {
 public:
  explicit PopDatastreamOutfeedEnqueueOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), device_ordinal_(0) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("outfeed_mode", &outfeed_mode_));
  }

  ~PopDatastreamOutfeedEnqueueOp() override{};

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    const auto num_inputs = ctx->num_inputs();

    OP_REQUIRES(
        ctx, outfeed_mode_ == "all" || outfeed_mode_ == "get_last",
        errors::InvalidArgument("Unkown outfeed_mode : ", outfeed_mode_,
                                ", supported values are 'all' and 'get_last'"));

    std::vector<xla::XlaOp> inputs;
    std::vector<xla::Shape> xla_shapes;
    inputs.reserve(num_inputs);
    xla_shapes.reserve(num_inputs);

    bool is_tuple = num_inputs > 1;

    for (int i = 0; i < num_inputs; ++i) {
      inputs.push_back(ctx->Input(i));
      auto input_shape = ctx->InputShape(i);
      auto dtype = ctx->input_type(i);
      xla::Shape xla_shape;
      OP_REQUIRES_OK(ctx,
                     TensorShapeToXLAShape(dtype, input_shape, &xla_shape));

      xla_shapes.emplace_back(xla_shape);
    }

    xla::Shape outfeed_shape;
    xla::XlaOp outfeed_input;
    if (is_tuple) {
      outfeed_shape = xla::ShapeUtil::MakeTupleShape(xla_shapes);
      outfeed_input = Tuple(b, inputs);
    } else {
      outfeed_shape = xla_shapes[0];
      outfeed_input = inputs[0];
    }

    xla::XlaOp outfeed_token = CreateToken(b);
    xla::XlaOp outfeed = OutfeedWithToken(outfeed_input, outfeed_token,
                                          outfeed_shape, outfeed_mode_);
  }

 private:
  int device_ordinal_;
  std::string outfeed_mode_ = "all";
  TF_DISALLOW_COPY_AND_ASSIGN(PopDatastreamOutfeedEnqueueOp);
};

REGISTER_IPU_OP("PopDatastreamOutfeedEnqueue", PopDatastreamOutfeedEnqueueOp);

class PopDatastreamOutfeedDequeueOp : public OpKernel {
 public:
  explicit PopDatastreamOutfeedDequeueOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), device_ordinal_(0) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));

    OP_REQUIRES(ctx, device_ordinal_ >= 0,
                errors::InvalidArgument("Need device_ordinal >= 0, got ",
                                        device_ordinal_));

    std::vector<PartialTensorShape> partial_shapes;
    std::vector<tensorflow::DataType> types;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &partial_shapes));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &types));

    outfeed_all_ =
        ((partial_shapes.size() > 1) && partial_shapes[0].unknown_rank());
    int start = outfeed_all_ ? 1 : 0;
    tensor_shapes_.reserve(partial_shapes.size() - start);
    for (int i = start; i < partial_shapes.size(); ++i) {
      xla::PrimitiveType xla_type;
      TensorShape tensor_shape;
      OP_REQUIRES(ctx, partial_shapes[i].AsTensorShape(&tensor_shape),
                  errors::InvalidArgument("Unable to cast partial tensor shape "
                                          "to tensor shape for tensor : ",
                                          partial_shapes[i].DebugString()));
      OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(types[i - start], &xla_type));
      xla_shapes_.emplace_back(TensorShapeToXLAShape(xla_type, tensor_shape));
      tensor_shapes_.emplace_back(tensor_shape);
    }

    num_outputs_ = ctx->num_outputs();
    OP_REQUIRES(ctx, ctx->num_outputs() == xla_shapes_.size(),
                errors::InvalidArgument(
                    "Outfeed num_outputs() != Attribute num outputs: ",
                    ctx->num_outputs(), " != ", xla_shapes_.size()));
  }

  ~PopDatastreamOutfeedDequeueOp() override{};

  void Compute(OpKernelContext* ctx) override {
    auto platform = se::MultiPlatformManager::PlatformWithName("Poplar");
    OP_REQUIRES(ctx, platform.ok(), platform.status());
    auto* p =
        static_cast<xla::poplarplugin::PoplarPlatform*>(platform.ValueOrDie());

    auto* transfer_manager =
        xla::TransferManager::GetForPlatform(p).ValueOrDie();

    auto executor = p->ExecutorForDevice(device_ordinal_).ValueOrDie();

    auto* outfeed_queue_manager =
        xla::poplarplugin::GetXfeedManager(device_ordinal_)->outfeed();
    size_t num_available = outfeed_queue_manager->size();
    if (num_available < num_outputs_) {
      num_available = outfeed_queue_manager->WaitForBuffers(num_outputs_);
    }

    // TODO(shauryas, T7218): This is slightly tedious due to tuples being
    // enqueued as separate buffers. When we call dequeue with
    // outfeed_all_==true we may be in a situation where only some of the tuple
    // buffers have been enqueued. When the performance optimization refactoring
    // is done this will need to be rewritten to handle a dequeueing of all the
    // tuple buffers at once.
    if (outfeed_all_) {
      size_t remainder = num_available % num_outputs_;
      size_t num_dequeue = (num_available - remainder);
      size_t num_outfeed = num_dequeue / num_outputs_;
      std::vector<Tensor*> output_tensors;
      for (size_t i = 0; i < num_outputs_; ++i) {
        TensorShape tensor_shape = tensor_shapes_[i];
        tensor_shape.InsertDim(0, num_outfeed);
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_output(i, tensor_shape, &output_tensor));
        output_tensors.push_back(output_tensor);
      }

      for (size_t i = 0; i < num_outfeed; ++i) {
        for (size_t j = 0; j < num_outputs_; ++j) {
          Tensor* output_tensor = output_tensors[j];
          auto subslice = output_tensor->SubSlice(i);
          const auto& xla_shape = xla_shapes_[j];
          const char* data = subslice.tensor_data().data();
          auto result_literal = xla::MutableBorrowingLiteral(data, xla_shape);
          OP_REQUIRES_OK(ctx, transfer_manager->TransferLiteralFromOutfeed(
                                  executor, xla_shape, result_literal));
        }
      }
    } else {
      for (size_t i = 0; i < xla_shapes_.size(); ++i) {
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(
            ctx, ctx->allocate_output(i, tensor_shapes_[i], &output_tensor));
        const auto& xla_shape = xla_shapes_[i];
        const char* data = output_tensor->tensor_data().data();
        auto result_literal = xla::MutableBorrowingLiteral(data, xla_shape);
        OP_REQUIRES_OK(ctx, transfer_manager->TransferLiteralFromOutfeed(
                                executor, xla_shape, result_literal));
      }
    }
  }

 private:
  int device_ordinal_;
  std::vector<xla::Shape> xla_shapes_;
  std::vector<TensorShape> tensor_shapes_;
  bool outfeed_all_;
  size_t num_outputs_;
  TF_DISALLOW_COPY_AND_ASSIGN(PopDatastreamOutfeedDequeueOp);
};

REGISTER_KERNEL_BUILDER(Name("PopDatastreamOutfeedDequeue").Device(DEVICE_CPU),
                        PopDatastreamOutfeedDequeueOp);

}  // namespace tensorflow
