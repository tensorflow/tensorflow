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
#include "tensorflow/core/kernels/data/model_dataset_op.h"

#include "tensorflow/core/framework/cancellation.h"

// On mobile we do not provide model dataset op because not all of its
// dependencies are available there. The op is replaced with a no-op.
#if !defined(IS_MOBILE_PLATFORM)
#include "absl/memory/memory.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace {

// Default share of available RAM that can be used by model's internal buffers.
constexpr double kRamBudgetShare = 0.5;

}  // namespace

/* static */ constexpr const char* const ModelDatasetOp::kDatasetType;
/* static */ constexpr const char* const ModelDatasetOp::kDatasetOp;
/* static */ constexpr const char* const ModelDatasetOp::kAlgorithm;
/* static */ constexpr const char* const ModelDatasetOp::kCpuBudget;
/* static */ constexpr const char* const ModelDatasetOp::kRamBudget;

class ModelDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          model::AutotuneAlgorithm algorithm, int64 cpu_budget,
          int64 ram_budget)
      : Dataset(DatasetContext(ctx), input, algorithm, cpu_budget, ram_budget) {
  }

  Dataset(DatasetContext&& ctx, const DatasetBase* input,
          model::AutotuneAlgorithm algorithm, int64 cpu_budget,
          int64 ram_budget)
      : DatasetBase(std::move(ctx)),
        input_(input),
        algorithm_(algorithm),
        cpu_budget_(cpu_budget),
        ram_budget_(ram_budget),
        traceme_metadata_(
            {{"algorithm", algorithm == model::AutotuneAlgorithm::HILL_CLIMB
                               ? "hill climb"
                               : "gradient descent"},
             {"cpu_budget",
              strings::Printf("%lld", static_cast<long long>(cpu_budget))},
             {"ram_budget",
              strings::Printf("%lldB", static_cast<long long>(ram_budget))}}) {
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
    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, output));
    AttrValue algorithm_attr;
    b->BuildAttrValue(static_cast<int64>(algorithm_), &algorithm_attr);
    AttrValue cpu_budget_attr;
    b->BuildAttrValue(cpu_budget_, &cpu_budget_attr);
    AttrValue ram_budget_attr;
    b->BuildAttrValue(ram_budget_, &ram_budget_attr);

    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node},
                      {std::make_pair(kAlgorithm, algorithm_attr),
                       std::make_pair(kCpuBudget, cpu_budget_attr),
                       std::make_pair(kRamBudget, ram_budget_attr)},
                      output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          cpu_budget_(dataset()->cpu_budget_ == 0 ? port::NumSchedulableCPUs()
                                                  : dataset()->cpu_budget_),
          ram_budget_(dataset()->ram_budget_ == 0
                          ? kRamBudgetShare * port::AvailableRam()
                          : dataset()->ram_budget_) {
      cancellation_manager_ = absl::make_unique<CancellationManager>();
      model_ = std::make_shared<model::Model>();
    }

    ~Iterator() override { cancellation_manager_->StartCancel(); }

    Status Initialize(IteratorContext* ctx) override {
      return dataset()->input_->MakeIterator(IteratorContext(CreateParams(ctx)),
                                             this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      if (!ctx->model()) {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(EnsureOptimizationLoopThreadStarted(ctx));
      }
      return input_impl_->GetNext(IteratorContext(CreateParams(ctx)),
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
      return SaveInput(ctx, writer, input_impl_);
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return RestoreInput(IteratorContext(CreateParams(ctx)), reader,
                          input_impl_);
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
      return dataset()->traceme_metadata_;
    }

   private:
    IteratorContext::Params CreateParams(IteratorContext* ctx) {
      IteratorContext::Params params(ctx);
      if (!ctx->model()) {
        params.model = model_;
      }
      return params;
    }

    Status EnsureOptimizationLoopThreadStarted(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!model_thread_) {
        model_thread_ = ctx->StartThread("tf_data_model", [this]() {
          Status status =
              model_->OptimizeLoop(dataset()->algorithm_, cpu_budget_,
                                   ram_budget_, cancellation_manager_.get());
          if (!status.ok()) {
            LOG(WARNING) << "Optimization loop failed: " << status.ToString();
          }
        });
      }
      return Status::OK();
    }

    mutex mu_;
    std::shared_ptr<model::Model> model_;
    // Controls cancellation of `model_thread_`. Must be ordered before
    // `model_thread_` so that `model_thread_` is destroyed first.
    std::unique_ptr<CancellationManager> cancellation_manager_;
    std::unique_ptr<Thread> model_thread_ TF_GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> input_impl_;
    const int64 cpu_budget_;
    const int64 ram_budget_;
  };

  const DatasetBase* input_;
  const model::AutotuneAlgorithm algorithm_;
  const int64 cpu_budget_;
  const int64 ram_budget_;
  const TraceMeMetadata traceme_metadata_;
};

// static
void ModelDatasetOp::MakeDatasetFromOptions(OpKernelContext* ctx,
                                            DatasetBase* input,
                                            model::AutotuneAlgorithm algorithm,
                                            bool cpu_budget, bool ram_budget,
                                            DatasetBase** output) {
  *output = new ModelDatasetOp::Dataset(
      DatasetContext(DatasetContext::Params(
          {ModelDatasetOp::kDatasetType, ModelDatasetOp::kDatasetOp})),
      input, algorithm, cpu_budget, ram_budget);
}

ModelDatasetOp::ModelDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  if (ctx->HasAttr(kAlgorithm)) {
    int64 algorithm;
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kAlgorithm, &algorithm));
    algorithm_ = model::AutotuneAlgorithm(algorithm);
  } else {
    algorithm_ = model::AutotuneAlgorithm::HILL_CLIMB;
  }
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kCpuBudget, &cpu_budget_));
  OP_REQUIRES(ctx, cpu_budget_ >= 0,
              errors::InvalidArgument("CPU budget must be positive but is ",
                                      cpu_budget_, "."));
  if (ctx->HasAttr(kRamBudget)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kRamBudget, &ram_budget_));
  } else {
    ram_budget_ = 0;
  }
  OP_REQUIRES(ctx, ram_budget_ >= 0,
              errors::InvalidArgument("RAM budget must be positive but is ",
                                      ram_budget_, "."));
}

void ModelDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                 DatasetBase** output) {
  *output = new ModelDatasetOp::Dataset(ctx, input, algorithm_, cpu_budget_,
                                        ram_budget_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ModelDataset").Device(DEVICE_CPU),
                        ModelDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
#else   // !IS_MOBILE_PLATFORM
namespace tensorflow {
namespace data {
// static
void ModelDatasetOp::MakeDatasetFromOptions(OpKernelContext* ctx,
                                            DatasetBase* input,
                                            model::AutotuneAlgorithm algorithm,
                                            bool cpu_budget, bool ram_budget,
                                            DatasetBase** output) {
  input->Ref();
  *output = input;
}

ModelDatasetOp::ModelDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}

void ModelDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                 DatasetBase** output) {
  input->Ref();
  *output = input;
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ModelDataset").Device(DEVICE_CPU),
                        ModelDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
#endif  // !IS_MOBILE_PLATFORM
