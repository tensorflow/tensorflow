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
#include "tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.h"

#include <map>
#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/rewrite_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr char AssertPrevDatasetOp::kInputDataset[];
/* static */ constexpr char AssertPrevDatasetOp::kDatasetType[];
/* static */ constexpr char AssertPrevDatasetOp::kTransformations[];
/* static */ constexpr char AssertPrevDatasetOp::kOutputTypes[];
/* static */ constexpr char AssertPrevDatasetOp::kOutputShapes[];

namespace {

// Returns a `NameAttrList` of an op name and attrs, parsed from
// `transformation`.
StatusOr<NameAttrList> GetAssertions(const tstring& transformation) {
  NameAttrList assertions;
  if (!std::is_base_of<protobuf::Message, NameAttrList>()) {
    return errors::InvalidArgument(
        "Portable proto implementations are not supported.");
  }
  if (!protobuf::TextFormat::ParseFromString(
          transformation, reinterpret_cast<protobuf::Message*>(&assertions))) {
    return errors::InvalidArgument("Couldn't parse transformation '",
                                   transformation, "'.");
  }
  return assertions;
}

// Returns `dataset`'s input dataset.
StatusOr<const DatasetBase*> GetPreviousDataset(const DatasetBase& dataset) {
  std::vector<const DatasetBase*> inputs;
  TF_RETURN_IF_ERROR(dataset.InputDatasets(&inputs));
  if (inputs.empty()) {
    return errors::InvalidArgument("No previous transformation found.");
  }
  return inputs.back();
}

// Checks `dataset`'s op name against that in `assertions`.
Status CheckOpName(const DatasetBase& dataset, const NameAttrList& assertions) {
  if (!MatchesAnyVersion(assertions.name(), dataset.type_string())) {
    return errors::InvalidArgument("Asserted transformation matching '",
                                   assertions.name(), "', but found '",
                                   dataset.type_string(), "'.");
  }
  return Status::OK();
}

// Returns a NodeDef representation of `dataset`.
StatusOr<NodeDef> GetDatasetNode(const DatasetBase& dataset,
                                 absl::string_view op_name) {
  SerializationContext serialization_ctx((SerializationContext::Params()));
  GraphDefBuilder b;
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(
      AsGraphDef(&dataset, std::move(serialization_ctx), &graph_def));
  TF_ASSIGN_OR_RETURN(NodeDef node, GetDatasetNodeDef(graph_def));
  return node;
}

// Checks `dataset`'s attrs against those in `assertions`.
Status CheckAttributes(const DatasetBase& dataset,
                       const NameAttrList& assertions) {
  if (assertions.attr().empty()) return Status::OK();
  TF_ASSIGN_OR_RETURN(NodeDef node, GetDatasetNode(dataset, assertions.name()));
  std::vector<std::string> attrs_not_found;
  for (const auto& attr : assertions.attr()) {
    auto it = node.attr().find(attr.first);
    if (it != node.attr().end()) {
      if (!std::is_base_of<protobuf::Message, AttrValue>()) {
        return errors::InvalidArgument(
            "Portable proto implementations are not supported.");
      }
      if (!protobuf::util::MessageDifferencer::Equivalent(
              *reinterpret_cast<const protobuf::Message*>(&it->second),
              *reinterpret_cast<const protobuf::Message*>(&attr.second))) {
        return errors::InvalidArgument(
            "Asserted attribute '", attr.first, "' having a value of '",
            attr.second.DebugString(), "', but found value of '",
            it->second.DebugString(), "'.");
      }
    } else {
      return errors::InvalidArgument(
          "Asserted attribute '", attr.first, "' having a value of '",
          attr.second.DebugString(), "', but found no such attribute defined.");
    }
  }
  return Status::OK();
}

// Checks `dataset`'s op name and attrs against those in `transformation`.
Status CheckTransformation(const DatasetBase& dataset,
                           const tstring& transformation) {
  TF_ASSIGN_OR_RETURN(NameAttrList assertions, GetAssertions(transformation));
  TF_RETURN_IF_ERROR(CheckOpName(dataset, assertions));
  TF_RETURN_IF_ERROR(CheckAttributes(dataset, assertions));
  return Status::OK();
}

}  // namespace

class AssertPrevDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          const std::vector<tstring>& transformations,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        transformations_(transformations),
        output_types_(output_types),
        output_shapes_(output_shapes) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }
  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal() const override { return input_->Cardinality(); }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* transformations_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddVector(transformations_, &transformations_node));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node, transformations_node}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    Status Initialize(IteratorContext* ctx) override {
      const DatasetBase* current_dataset = dataset();
      for (int i = 0; i < dataset()->transformations_.size(); ++i) {
        StatusOr<const DatasetBase*> previous_dataset =
            GetPreviousDataset(*current_dataset);
        if (!previous_dataset.ok()) {
          return errors::InvalidArgument(
              "Asserted previous ", dataset()->transformations_.size(),
              " transformations but encountered only ", i, ".");
        }

        Status s = CheckTransformation(**previous_dataset,
                                       dataset()->transformations_[i]);
        if (!s.ok()) {
          return errors::InvalidArgument(
              "Failure checking transformations at offset ", i, ": ",
              s.error_message());
        }

        current_dataset = *previous_dataset;
      }
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      return Status::OK();
    }

   private:
    std::unique_ptr<IteratorBase> input_impl_;
  };

  const DatasetBase* input_;
  const std::vector<tstring> transformations_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

AssertPrevDatasetOp::AssertPrevDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void AssertPrevDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                      DatasetBase** output) {
  std::vector<tstring> transformations;
  OP_REQUIRES_OK(ctx, ParseVectorArgument<tstring>(ctx, kTransformations,
                                                   &transformations));
  *output =
      new Dataset(ctx, input, transformations, output_types_, output_shapes_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("AssertPrevDataset").Device(DEVICE_CPU),
                        AssertPrevDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
