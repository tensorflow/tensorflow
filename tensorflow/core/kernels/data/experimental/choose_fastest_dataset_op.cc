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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/histogram/histogram.h"

namespace tensorflow {
namespace data {
namespace {

static const double kPercentile = 90.0;

class ChooseFastestDatasetOp : public DatasetOpKernel {
 public:
  explicit ChooseFastestDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_experiments", &num_experiments_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    OpInputList input_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("input_datasets", &input_list));
    OP_REQUIRES(
        ctx, input_list.size() > 1,
        errors::InvalidArgument(
            "ChooseFastestDataset must have at least two input datasets."));

    std::vector<DatasetBase*> inputs;
    inputs.reserve(input_list.size());
    for (const auto& tensor : input_list) {
      DatasetBase* input;
      OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(tensor, &input));
      inputs.push_back(input);
    }

    const DataTypeVector& output_types = inputs[0]->output_dtypes();
    for (size_t i = 1, num_inputs = inputs.size(); i < num_inputs; ++i) {
      OP_REQUIRES(ctx, inputs[i]->output_dtypes() == output_types,
                  errors::InvalidArgument(
                      "All inputs to ChooseFastestDataset "
                      "must have the same output types. Input ",
                      i, " has output types: ",
                      DataTypeVectorString(inputs[i]->output_dtypes()),
                      ", while all prior inputs have types: ",
                      DataTypeVectorString(output_types), "."));
    }

    std::vector<PartialTensorShape> output_shapes = inputs[0]->output_shapes();
    // Merge the output shapes of all the input datasets, returning an
    // error if any of them are incompatible.
    for (size_t i = 1, num_inputs = inputs.size(); i < num_inputs; ++i) {
      OP_REQUIRES(
          ctx, inputs[i]->output_shapes().size() == output_shapes.size(),
          errors::InvalidArgument(
              "All inputs to ChooseFastestDataset must have compatible outputs."
              " Input ",
              i, " has ", inputs[i]->output_shapes().size(),
              " components, while all prior inputs have ", output_shapes.size(),
              " components."));
      for (size_t j = 0, num_components = output_shapes.size();
           j < num_components; ++j) {
        PartialTensorShape result;
        OP_REQUIRES(
            ctx,
            output_shapes[j]
                .MergeWith(inputs[i]->output_shapes().at(j), &result)
                .ok(),
            errors::InvalidArgument(
                "All inputs to ChooseFastestDataset must have "
                "compatible output shapes. Component ",
                j, " of input ", i, " has shape: ",
                inputs[i]->output_shapes().at(j), ", while components ", j,
                " of all prior inputs have shape: ", output_shapes[j], "."));
        output_shapes[j] = std::move(result);
      }
    }

    int64 cardinality = inputs[0]->Cardinality();
    for (size_t i = 1, num_inputs = inputs.size(); i < num_inputs; ++i) {
      if (cardinality == kUnknownCardinality) {
        cardinality = inputs[i]->Cardinality();
      } else {
        OP_REQUIRES(
            ctx,
            inputs[i]->Cardinality() == cardinality ||
                inputs[i]->Cardinality() == kUnknownCardinality,
            errors::InvalidArgument(
                "All inputs to ChooseFastestDataset must have compatible "
                "cardinalities. Input ",
                i, " has cardinality: ", inputs[i]->Cardinality(),
                ", while all prior inputs have cardinality: ", cardinality,
                "."));
      }
    }
    *output = new Dataset(ctx, std::move(inputs), std::move(output_shapes),
                          cardinality, num_experiments_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<DatasetBase*> inputs,
            std::vector<PartialTensorShape> output_shapes, int64 cardinality,
            int64 num_experiments)
        : DatasetBase(DatasetContext(ctx)),
          inputs_(std::move(inputs)),
          output_shapes_(std::move(output_shapes)),
          cardinality_(cardinality),
          num_experiments_(num_experiments) {
      for (auto input : inputs_) {
        input->Ref();
      }
    }

    ~Dataset() override {
      for (auto input : inputs_) {
        input->Unref();
      }
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<ChooseFastestIterator>(
          ChooseFastestIterator::Params{
              this, strings::StrCat(prefix, "::ChooseFastest")});
    }

    const DataTypeVector& output_dtypes() const override {
      return inputs_[0]->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "ChooseFastestDatasetOp::Dataset";
    }

    int64 Cardinality() const override { return cardinality_; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      std::vector<Node*> input_nodes;
      input_nodes.reserve(inputs_.size());
      for (const auto& input : inputs_) {
        Node* input_node;
        TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input, &input_node));
        input_nodes.push_back(input_node);
      }
      AttrValue num_experiments_attr;
      b->BuildAttrValue(num_experiments_, &num_experiments_attr);
      return b->AddDataset(
          this, {}, {std::make_pair(0, input_nodes)},
          {std::make_pair("num_experiments", std::move(num_experiments_attr))},
          output);
    }

   private:
    class ChooseFastestIterator : public DatasetIterator<Dataset> {
     public:
      explicit ChooseFastestIterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            histograms_(dataset()->inputs_.size()) {}

      Status Initialize(IteratorContext* ctx) override {
        mutex_lock l(mu_);
        input_impls_.resize(dataset()->inputs_.size());
        for (size_t i = 0, num_inputs = dataset()->inputs_.size();
             i < num_inputs; ++i) {
          TF_RETURN_IF_ERROR(dataset()->inputs_[i]->MakeIterator(
              ctx, strings::StrCat(prefix(), "_", i), &input_impls_[i]));
        }
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);

        // The first num_experiments_ iterations, we fire up a thread for
        // each input that calls its GetNext function and records the time
        // taken. We only return when all the threads have completed.
        if (experiment_counter_ < dataset()->num_experiments_) {
          experiment_counter_++;
          std::vector<ThreadInfo> threads = StartThreads(ctx);
          for (const auto& thread : threads) {
            thread.result->notification.WaitForNotification();
          }

          *out_tensors = std::move(threads[0].result->out_tensors);
          *end_of_sequence = threads[0].result->end_of_sequence;

          if (experiment_counter_ == dataset()->num_experiments_) {
            SelectFastestInputIndex();
          }
          return threads[0].result->status;
        }
        return input_impls_[fastest_index_]->GetNext(ctx, out_tensors,
                                                     end_of_sequence);
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1);
      }

      // TODO(rachelim): Save and restore histogram state as well. Currently,
      // if an iterator is saved and restored, the histograms start recording
      // from scratch.
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (input_impls_.empty()) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impls_empty"), ""));
        } else {
          for (auto& input_impl : input_impls_) {
            TF_RETURN_IF_ERROR(SaveInput(writer, input_impl));
          }
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("experiment_counter"),
                                               experiment_counter_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("fastest_index"), fastest_index_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (reader->Contains(full_name("input_impls_empty"))) {
          input_impls_.clear();
        } else {
          DCHECK_EQ(input_impls_.size(), dataset()->inputs_.size());
          for (auto& input_impl : input_impls_) {
            TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl));
          }
        }
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("experiment_counter"),
                                              &experiment_counter_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("fastest_index"), &fastest_index_));
        return Status::OK();
      }

     private:
      struct InvocationResult {
        Notification notification;
        Status status;
        bool end_of_sequence;
        std::vector<Tensor> out_tensors;
      };

      struct ThreadInfo {
        std::unique_ptr<InvocationResult> result;
        std::unique_ptr<Thread> thread;
      };

      std::vector<std::unique_ptr<IteratorBase>> input_impls_;
      // For tracking the time taken for each input's iterations.
      std::vector<histogram::Histogram> histograms_;

      mutex mu_;
      int64 experiment_counter_ GUARDED_BY(mu_) = 0;
      int64 fastest_index_ = -1;

      std::vector<ThreadInfo> StartThreads(IteratorContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        std::vector<ThreadInfo> threads(dataset()->inputs_.size());
        for (size_t i = 0, num_inputs = dataset()->inputs_.size();
             i < num_inputs; ++i) {
          threads[i].result = absl::make_unique<InvocationResult>();
          threads[i].thread.reset(ctx->env()->StartThread(
              {}, strings::StrCat("tf_data_merge_", i),
              std::bind(&ChooseFastestIterator::RunnerThread, this, ctx,
                        threads[i].result.get(), i)));
        }
        return threads;
      }

      void RunnerThread(IteratorContext* ctx, InvocationResult* result, int i) {
        int64 start = Env::Default()->NowNanos();
        Status s = input_impls_[i]->GetNext(ctx, &result->out_tensors,
                                            &result->end_of_sequence);
        histograms_[i].Add(
            static_cast<double>(Env::Default()->NowNanos() - start));

        result->status = s;
        result->notification.Notify();
      }

      // Select the fastest input to use based on the histograms of timings
      // of the completed threads. The input with the best 90th percentile
      // iteration time is selected.
      void SelectFastestInputIndex() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        fastest_index_ = 0;

        double best_percentile = histograms_[0].Percentile(kPercentile);
        for (size_t i = 1, num_inputs = histograms_.size(); i < num_inputs;
             ++i) {
          double percentile = histograms_[i].Percentile(kPercentile);
          if (percentile <= best_percentile) {
            best_percentile = percentile;
            fastest_index_ = i;
          }
        }
      }
    };  // class Iterator

    const std::vector<DatasetBase*> inputs_;
    const std::vector<PartialTensorShape> output_shapes_;
    const int64 cardinality_;
    const int64 num_experiments_;
  };  // class Dataset

  int64 num_experiments_;
};  // class ChooseFastestDatasetOp

// Register the kernel implementation for ChooseFastestDataset.
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalChooseFastestDataset").Device(DEVICE_CPU),
    ChooseFastestDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
