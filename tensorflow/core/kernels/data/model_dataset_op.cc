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

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace {

constexpr int64 kOptimizationPeriodThresholdMs = 60 * EnvTime::kSecondsToMillis;

// Default share of available RAM that can be used by model's internal buffers.
constexpr double kRamBudgetShare = 0.5;

class ModelDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ModelDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    if (ctx->HasAttr("algorithm")) {
      int64 algorithm;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("algorithm", &algorithm));
      algorithm_ = model::AutotuneAlgorithm(algorithm);
    } else {
      algorithm_ = model::AutotuneAlgorithm::HILL_CLIMB;
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cpu_budget", &cpu_budget_));
    if (cpu_budget_ == 0) {
      cpu_budget_ = port::NumSchedulableCPUs();
    }
    OP_REQUIRES(ctx, cpu_budget_ > 0,
                errors::InvalidArgument("CPU budget must be positive but is ",
                                        cpu_budget_, "."));
    ram_budget_ = kRamBudgetShare * port::AvailableRam();
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    *output = new Dataset(ctx, input, algorithm_, cpu_budget_, ram_budget_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            model::AutotuneAlgorithm algorithm, int64 cpu_budget,
            int64 ram_budget)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          algorithm_(algorithm),
          cpu_budget_(cpu_budget),
          ram_budget_(ram_budget) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::Model")});
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override { return "ModelDatasetOp::Dataset"; }

    int64 Cardinality() const override { return input_->Cardinality(); }

    Status CheckExternalState() const override {
      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
        auto remove_node_hook = [](std::shared_ptr<model::Node> node) {
          metrics::RecordTFDataElements(node->name(), node->num_elements());
        };
        model_ = std::make_shared<model::Model>(std::move(remove_node_hook));
      }

      ~Iterator() override {
        // Signal the optimize thread to terminate it. We will then join that
        // thread when we delete `this->optimize_thread_`.
        mutex_lock l(mu_);
        cancelled_ = true;
        cond_var_.notify_all();
      }

      Status Initialize(IteratorContext* ctx) override {
        IteratorContext::Params params(ctx);
        params.model = model_;
        return dataset()->input_->MakeIterator(
            IteratorContext(std::move(params)), this, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        IteratorContext::Params params(ctx);
        {
          mutex_lock l(mu_);
          TF_RETURN_IF_ERROR(EnsureOptimizeThreadStarted(ctx));
          params.model = model_;
        }
        return input_impl_->GetNext(IteratorContext(std::move(params)),
                                    out_tensors, end_of_sequence);
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        return Status::OK();
      }

     private:
      Status EnsureOptimizeThreadStarted(IteratorContext* ctx)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (!optimize_thread_) {
          std::shared_ptr<IteratorContext> new_ctx =
              std::make_shared<IteratorContext>(*ctx);
          optimize_thread_ = ctx->StartThread(
              "tf_data_model", [this, new_ctx]() { OptimizeThread(new_ctx); });
        }
        return Status::OK();
      }

      void OptimizeThread(const std::shared_ptr<IteratorContext>& ctx) {
        int64 last_optimization_ms = 0;
        int64 optimization_period_ms = 10;
        int64 current_time_ms = EnvTime::NowMicros() / EnvTime::kMillisToMicros;
        while (true) {
          {
            mutex_lock l(mu_);
            while (!cancelled_ &&
                   last_optimization_ms + optimization_period_ms >
                       current_time_ms) {
              auto wait_ms = last_optimization_ms + optimization_period_ms -
                             current_time_ms;
              VLOG(2) << "Waiting for " << wait_ms << " ms.";
              cond_var_.wait_for(l, std::chrono::milliseconds(wait_ms));
              current_time_ms = EnvTime::NowMicros() / EnvTime::kMillisToMicros;
            }
            if (cancelled_) return;
          }
          model_->Optimize(dataset()->algorithm_, dataset()->cpu_budget_,
                           dataset()->ram_budget_);
          // Exponentially increase the period of running the optimization
          // until a threshold is reached.
          if (optimization_period_ms != kOptimizationPeriodThresholdMs) {
            optimization_period_ms = std::min(optimization_period_ms << 1,
                                              kOptimizationPeriodThresholdMs);
          }
          current_time_ms = EnvTime::NowMicros() / EnvTime::kMillisToMicros;
          last_optimization_ms = current_time_ms;
        }
      }

      mutex mu_;
      condition_variable cond_var_;
      std::shared_ptr<model::Model> model_;
      std::unique_ptr<Thread> optimize_thread_ TF_GUARDED_BY(mu_);
      bool cancelled_ TF_GUARDED_BY(mu_) = false;
      std::unique_ptr<IteratorBase> input_impl_;
    };

    const DatasetBase* input_;
    const model::AutotuneAlgorithm algorithm_;
    const int64 cpu_budget_;
    const int64 ram_budget_;
  };

  model::AutotuneAlgorithm algorithm_;
  int64 cpu_budget_;
  int64 ram_budget_;
};

REGISTER_KERNEL_BUILDER(Name("ModelDataset").Device(DEVICE_CPU),
                        ModelDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
