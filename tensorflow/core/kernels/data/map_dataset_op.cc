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
#include "tensorflow/core/kernels/data/map_dataset_op.h"

#include "absl/status/status.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const MapDatasetOp::kDatasetType;
/* static */ constexpr const char* const MapDatasetOp::kInputDataset;
/* static */ constexpr const char* const MapDatasetOp::kOtherArguments;
/* static */ constexpr const char* const MapDatasetOp::kFunc;
/* static */ constexpr const char* const MapDatasetOp::kTarguments;
/* static */ constexpr const char* const MapDatasetOp::kOutputTypes;
/* static */ constexpr const char* const MapDatasetOp::kOutputShapes;
/* static */ constexpr const char* const MapDatasetOp::kUseInterOpParallelism;
/* static */ constexpr const char* const MapDatasetOp::kPreserveCardinality;
/* static */ constexpr const char* const MapDatasetOp::kForceSynchronous;

class MapDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          std::unique_ptr<CapturedFunction> captured_func,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes,
          bool preserve_cardinality, bool force_synchronous)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        preserve_cardinality_(preserve_cardinality),
        force_synchronous_(force_synchronous),
        captured_func_(std::move(captured_func)),
        output_types_(output_types),
        output_shapes_(output_shapes) {
    input_->Ref();
    random_indexing_compatible_ = absl::OkStatus();
    if (input_ != nullptr) {
      random_indexing_compatible_ = input_->RandomIndexingCompatible();
    }
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    if (preserve_cardinality_) {
      return input_->Cardinality(options);
    } else {
      return kUnknownCardinality;
    }
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }

  Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    std::vector<Tensor> args;
    TF_RETURN_IF_ERROR(input_->Get(ctx, index, &args));
    if (!instantiated_captured_func_) {
      TF_RETURN_IF_ERROR(
          captured_func_->Instantiate(InstantiateCapturedFunctionParams(ctx),
                                      &instantiated_captured_func_));
    }
    return instantiated_captured_func_->RunInstantiated(args, out_tensors);
  }

  absl::Status RandomIndexingCompatible() const override {
    return random_indexing_compatible_;
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));

    // Attr: f
    AttrValue f_attr;
    b->BuildAttrValue(captured_func_->func(), &f_attr);

    // Attr: Targuments
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

    // Attr: use_inter_op_parallelism
    AttrValue use_inter_op_parallelism_attr;
    b->BuildAttrValue(captured_func_->use_inter_op_parallelism(),
                      &use_inter_op_parallelism_attr);

    // Attr: preserve_cardinality
    AttrValue preserve_cardinality_attr;
    b->BuildAttrValue(preserve_cardinality_, &preserve_cardinality_attr);

    // Attr: force_synchronous
    AttrValue force_synchronous_attr;
    b->BuildAttrValue(force_synchronous_, &force_synchronous_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {std::make_pair(0, input_graph_node)},  // Single tensor inputs.
        {std::make_pair(1, other_arguments)},         // Tensor list inputs.
        {std::make_pair(kFunc, f_attr),
         std::make_pair(kTarguments, other_arguments_types_attr),
         std::make_pair(kUseInterOpParallelism, use_inter_op_parallelism_attr),
         std::make_pair(kPreserveCardinality, preserve_cardinality_attr),
         std::make_pair(kForceSynchronous, force_synchronous_attr)},  // Attrs
        output));
    return absl::OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    bool SymbolicCheckpointCompatible() const override { return true; }

    Status Initialize(IteratorContext* ctx) override {
      TF_RETURN_IF_ERROR(
          dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
      return dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_);
    }

    // NOTE(mrry): This method is thread-safe as long as `input_impl_` and `f`
    // are thread-safe. However, if multiple threads enter this method,
    // outputs may be observed in a non-deterministic order.
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      std::vector<Tensor> args;
      TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &args, end_of_sequence));
      if (*end_of_sequence) {
        return absl::OkStatus();
      }

      Status s = instantiated_captured_func_->Run(ctx, std::move(args),
                                                  out_tensors, model_node());
      if (errors::IsOutOfRange(s)) {
        if (dataset()->preserve_cardinality_) {
          // To guarantee that the transformation preserves the cardinality of
          // the dataset, we convert `OutOfRange` to `InvalidArgument` as the
          // former may be interpreted by a caller as the end of sequence.
          return errors::InvalidArgument(
              "Function invocation produced OutOfRangeError: ", s.message());
        } else {
          // `f` may deliberately raise `errors::OutOfRange` to indicate
          // that we should terminate the iteration early.
          *end_of_sequence = true;
          return absl::OkStatus();
        }
      }
      if (!s.ok()) {
        return AddErrorContext(s);
      }
      return s;
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
          dataset()->captured_func_->CheckExternalState()));
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      return absl::OkStatus();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      return absl::OkStatus();
    }

   private:
    std::unique_ptr<IteratorBase> input_impl_;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
  };

  const DatasetBase* const input_;
  const bool preserve_cardinality_;
  const bool force_synchronous_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  // This is used for random access provided by Get().
  mutable std::unique_ptr<InstantiatedCapturedFunction>
      instantiated_captured_func_;
  absl::Status random_indexing_compatible_;
};

MapDatasetOp::MapDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  FunctionMetadata::Params params;
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kUseInterOpParallelism,
                                   &params.use_inter_op_parallelism));
  OP_REQUIRES_OK(ctx,
                 FunctionMetadata::Create(ctx, kFunc, params, &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(kPreserveCardinality, &preserve_cardinality_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kForceSynchronous, &force_synchronous_));
}

void MapDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                               DatasetBase** output) {
  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));

  *output =
      new Dataset(ctx, input, std::move(captured_func), output_types_,
                  output_shapes_, preserve_cardinality_, force_synchronous_);
}

namespace {

REGISTER_KERNEL_BUILDER(Name("MapDataset").Device(DEVICE_CPU), MapDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalMapDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("input_dataset")
                            .HostMemory("handle"),
                        MapDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("MapDataset");

}  // namespace
}  // namespace data
}  // namespace tensorflow
