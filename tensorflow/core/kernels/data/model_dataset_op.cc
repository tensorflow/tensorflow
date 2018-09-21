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

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset.h"
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

      Status Initialize(IteratorContext* ctx) override {
        IteratorContext ctx_with_model(CreateParams(ctx));
        return dataset()->input_->MakeIterator(&ctx_with_model, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        int64 now = ctx->env()->NowMicros() / EnvTime::kMillisToMicros;
        if (last_optimization_ms_ + optimization_period_ms_ < now) {
          model_->Optimize(port::NumSchedulableCPUs());
          // Exponentially increase the period of running the optimization until
          // a threshold is reached.
          if (optimization_period_ms_ < kOptimizationPeriodThresholdMs) {
            if (optimization_period_ms_ << 1 < kOptimizationPeriodThresholdMs) {
              optimization_period_ms_ <<= 1;
            } else {
              optimization_period_ms_ = kOptimizationPeriodThresholdMs;
            }
          }
          last_optimization_ms_ =
              ctx->env()->NowMicros() / EnvTime::kMillisToMicros;
        }
        IteratorContext ctx_with_model(CreateParams(ctx));
        return input_impl_->GetNext(&ctx_with_model, out_tensors,
                                    end_of_sequence);
      }

     protected:
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

      IteratorContext::Params CreateParams(IteratorContext* ctx) {
        IteratorContext::Params params = ctx->params();
        params.model = model_;
        return params;
      }

     private:
      mutex mu_;
      std::shared_ptr<model::Model> model_;
      int64 last_optimization_ms_ GUARDED_BY(mu_) = 0;
      int64 optimization_period_ms_ GUARDED_BY(mu_) = 10;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    const DatasetBase* input_;
  };
};

REGISTER_KERNEL_BUILDER(Name("ModelDataset").Device(DEVICE_CPU),
                        ModelDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
