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
#include "tensorflow/core/kernels/data/filter_dataset_op.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/stats_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const FilterDatasetOp::kDatasetType;
/* static */ constexpr const char* const FilterDatasetOp::kInputDataset;
/* static */ constexpr const char* const FilterDatasetOp::kOtherArguments;
/* static */ constexpr const char* const FilterDatasetOp::kPredicate;
/* static */ constexpr const char* const FilterDatasetOp::kTarguments;
/* static */ constexpr const char* const FilterDatasetOp::kOutputTypes;
/* static */ constexpr const char* const FilterDatasetOp::kOutputShapes;

constexpr char kInputImplEmpty[] = "input_impl_empty";
constexpr char kFilteredElements[] = "filtered_elements";
constexpr char kDroppedElements[] = "dropped_elements";

class FilterDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          std::unique_ptr<CapturedFunction> captured_func)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        captured_func_(std::move(captured_func)) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }

  Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));
    AttrValue f;
    b->BuildAttrValue(captured_func_->func(), &f);
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {{0, input_graph_node}}, {{1, other_arguments}},
        {{kPredicate, f}, {kTarguments, other_arguments_types_attr}}, output));
    return absl::OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          filtered_elements_(0),
          dropped_elements_(0) {}

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
      auto stats_aggregator = ctx->stats_aggregator();
      bool matched;
      do {
        {
          tf_shared_lock l(mu_);
          if (!input_impl_) {
            *end_of_sequence = true;
            return absl::OkStatus();
          }
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        }
        if (*end_of_sequence) {
          mutex_lock l(mu_);
          input_impl_.reset();
          return absl::OkStatus();
        }

        std::vector<Tensor> result;
        auto status = instantiated_captured_func_->RunWithBorrowedArgs(
            ctx, *out_tensors, &result, model_node());
        if (!status.ok()) {
          return AddErrorContext(status);
        }

        if (result.size() != 1 || result[0].dtype() != DT_BOOL ||
            result[0].NumElements() != 1) {
          // Clear the output tensor list since there were errors with Filter
          // prediction result.
          out_tensors->clear();
          return errors::InvalidArgument(
              "Filter predicate `f` must return a scalar bool.");
        }
        matched = result[0].scalar<bool>()();

        if (!matched) {
          // Clear the output tensor list since it didn't match.
          out_tensors->clear();
          {
            mutex_lock l(mu_);
            dropped_elements_++;
          }
          if (stats_aggregator) {
            mutex_lock l(mu_);
            stats_aggregator->AddScalar(
                stats_utils::DroppedElementsScalarName(dataset()->node_name()),
                static_cast<float>(dropped_elements_), num_elements());

            stats_aggregator->IncrementCounter(dataset()->node_name(),
                                               stats_utils::kDroppedElements,
                                               static_cast<float>(1));
          }
        }
      } while (!matched);
      // TODO(shivaniagrawal): add ratio of dropped_elements and
      // filtered_elements as a histogram.
      {
        mutex_lock l(mu_);
        filtered_elements_++;
      }
      if (stats_aggregator) {
        mutex_lock l(mu_);
        stats_aggregator->AddScalar(
            stats_utils::FilterdElementsScalarName(dataset()->node_name()),
            static_cast<float>(filtered_elements_), num_elements());

        stats_aggregator->IncrementCounter(dataset()->node_name(),
                                           stats_utils::kFilteredElements,
                                           static_cast<float>(1));
      }
      *end_of_sequence = false;
      return absl::OkStatus();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeUnknownRatioNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
          dataset()->captured_func_->CheckExternalState()));
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          prefix(), kInputImplEmpty, static_cast<int64_t>(!input_impl_)));
      if (input_impl_) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      }
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kFilteredElements, filtered_elements_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kDroppedElements, dropped_elements_));
      return absl::OkStatus();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      int64_t input_empty;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kInputImplEmpty, &input_empty));
      if (static_cast<bool>(input_empty)) {
        input_impl_.reset();
      } else {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      }
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kFilteredElements, &filtered_elements_));
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kDroppedElements, &dropped_elements_));
      return absl::OkStatus();
    }

    data::TraceMeMetadata GetTraceMeMetadata() const override {
      tf_shared_lock l(mu_);
      data::TraceMeMetadata result;
      result.push_back(std::make_pair(
          "passed",
          strings::Printf("%lld", static_cast<long long>(filtered_elements_))));
      result.push_back(std::make_pair(
          "filtered",
          strings::Printf("%lld", static_cast<long long>(dropped_elements_))));
      return result;
    }

   private:
    mutable mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    int64_t filtered_elements_ TF_GUARDED_BY(mu_);
    int64_t dropped_elements_ TF_GUARDED_BY(mu_);
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
  };

  const DatasetBase* const input_;
  const std::unique_ptr<CapturedFunction> captured_func_;
};

FilterDatasetOp::FilterDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kPredicate, /*params=*/{},
                                               &func_metadata_));
  OP_REQUIRES(ctx, func_metadata_->short_circuit_info().indices.size() <= 1,
              errors::InvalidArgument(
                  "predicate function has more than one return value."));
}

void FilterDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                  DatasetBase** output) {
  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));

  *output = new Dataset(ctx, input, std::move(captured_func));
}

namespace {

REGISTER_KERNEL_BUILDER(Name("FilterDataset").Device(DEVICE_CPU),
                        FilterDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("FilterDataset");

}  // namespace
}  // namespace data
}  // namespace tensorflow
