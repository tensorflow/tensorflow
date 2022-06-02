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
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class SleepDatasetOp : public UnaryDatasetOpKernel {
 public:
  using UnaryDatasetOpKernel::UnaryDatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64_t sleep_microseconds;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, "sleep_microseconds",
                                                     &sleep_microseconds));

    OP_REQUIRES(ctx, sleep_microseconds >= 0,
                errors::InvalidArgument("`sleep_microseconds` must be >= 0"));

    *output = new Dataset(ctx, input, sleep_microseconds);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            int64_t sleep_microseconds)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          sleep_microseconds_(sleep_microseconds) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::Sleep")});
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override { return "SleepDatasetOp::Dataset"; }

    int64_t CardinalityInternal() const override {
      return input_->Cardinality();
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      inputs->push_back(input_);
      return OkStatus();
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

      Node* sleep_microseconds = nullptr;
      TF_RETURN_IF_ERROR(
          b->AddScalar(sleep_microseconds_, &sleep_microseconds));

      return b->AddDataset(this,
                           {{0, input_graph_node},
                            {1, sleep_microseconds}},  // Single tensor inputs.
                           {},                         // Tensor list inputs.
                           {},                         // Attrs
                           output);
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      ~Iterator() override {
        {
          mutex_lock l(mu_);
          cancelled_ = true;
        }
        if (deregister_fn_) {
          deregister_fn_();
        }
      }

      Status Initialize(IteratorContext* ctx) override {
        TF_RETURN_IF_ERROR(RegisterCancellationCallback(
            ctx->cancellation_manager(),
            [this]() {
              mutex_lock l(mu_);
              cancelled_ = true;
            },
            &deregister_fn_));
        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        RecordStop(ctx);
        bool cancelled = mu_.AwaitWithDeadline(
            Condition(&cancelled_),
            EnvTime::NowNanos() +
                dataset()->sleep_microseconds_ * EnvTime::kMicrosToNanos);
        RecordStart(ctx);
        if (cancelled) {
          return errors::Cancelled("Operation was cancelled");
        }
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
        return SaveInput(ctx, writer, input_impl_);
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        return RestoreInput(ctx, reader, input_impl_);
      }

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      bool cancelled_ TF_GUARDED_BY(mu_) = false;
      std::function<void()> deregister_fn_;
    };

    const DatasetBase* const input_;
    // TODO(b/117612213): Investigate autotuning for this value.
    const int64_t sleep_microseconds_;
  };
};

REGISTER_KERNEL_BUILDER(Name("SleepDataset").Device(DEVICE_CPU),
                        SleepDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalSleepDataset").Device(DEVICE_CPU),
                        SleepDatasetOp);

REGISTER_KERNEL_BUILDER(Name("SleepDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("sleep_microseconds")
                            .HostMemory("input_dataset")
                            .HostMemory("handle"),
                        SleepDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalSleepDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("sleep_microseconds")
                            .HostMemory("input_dataset")
                            .HostMemory("handle"),
                        SleepDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
