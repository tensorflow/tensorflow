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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"

namespace tensorflow {
namespace data {
namespace {

const int kOptimizationPeriodThresholdMs = 60 * EnvTime::kSecondsToMicros;

class ModelDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ModelDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    *output = new Dataset(ctx, input);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, const DatasetBase* input)
        : DatasetBase(DatasetContext(ctx)), input_(input) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Model")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override { return "ModelDatasetOp::Dataset"; }

    int64 Cardinality() const override { return input_->Cardinality(); }

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
          : DatasetIterator<Dataset>(params),
            model_(std::make_shared<model::Model>()) {}

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
            IteratorContext(std::move(params)), prefix(), &input_impl_);
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

      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
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
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (!optimize_thread_) {
          std::shared_ptr<IteratorContext> new_ctx(new IteratorContext(*ctx));
          optimize_thread_.reset(ctx->env()->StartThread(
              {}, "tf_data_model",
              [this, new_ctx]() { OptimizeThread(new_ctx); }));
        }
        return Status::OK();
      }

      void OptimizeThread(const std::shared_ptr<IteratorContext>& ctx) {
        int64 last_optimization_ms = 0;
        int64 optimization_period_ms = 10;
        while (true) {
          {
            mutex_lock l(mu_);
            while (!cancelled_ &&
                   last_optimization_ms + optimization_period_ms >=
                       ctx->env()->NowMicros() / EnvTime::kMillisToMicros) {
              cond_var_.wait_for(
                  l, std::chrono::milliseconds(
                         last_optimization_ms + optimization_period_ms -
                         ctx->env()->NowMicros() / EnvTime::kMillisToMicros));
            }
            if (cancelled_) return;
          }
          model_->Optimize(port::NumSchedulableCPUs());
          // Exponentially increase the period of running the optimization
          // until a threshold is reached.
          if (optimization_period_ms < kOptimizationPeriodThresholdMs) {
            if (optimization_period_ms << 1 < kOptimizationPeriodThresholdMs) {
              optimization_period_ms <<= 1;
            } else {
              optimization_period_ms = kOptimizationPeriodThresholdMs;
            }
          }
          last_optimization_ms =
              ctx->env()->NowMicros() / EnvTime::kMillisToMicros;
        }
      }

      mutex mu_;
      condition_variable cond_var_;
      std::shared_ptr<model::Model> model_;
      std::unique_ptr<Thread> optimize_thread_ GUARDED_BY(mu_);
      bool cancelled_ GUARDED_BY(mu_) = false;
      std::unique_ptr<IteratorBase> input_impl_;
    };

    const DatasetBase* input_;
  };
};

REGISTER_KERNEL_BUILDER(Name("ModelDataset").Device(DEVICE_CPU),
                        ModelDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
