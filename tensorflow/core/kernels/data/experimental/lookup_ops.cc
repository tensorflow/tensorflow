/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/root_dataset.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/initializable_lookup_table.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/refcount.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

using InitializerSerializer =
    ::tensorflow::lookup::InitializableLookupTable::InitializerSerializer;

class DatasetIterator
    : public lookup::InitializableLookupTable::InitTableIterator {
 public:
  explicit DatasetIterator(data::DatasetBase* dataset) : dataset_(dataset) {}

  ~DatasetIterator() override {}

  Status Init(OpKernelContext* ctx) {
    data::IteratorContext::Params params(ctx);
    function_handle_cache_ = std::make_unique<FunctionHandleCache>(params.flr);
    params.function_handle_cache = function_handle_cache_.get();
    params.resource_mgr = &resource_mgr_;
    cancellation_manager_ =
        std::make_unique<CancellationManager>(ctx->cancellation_manager());
    params.cancellation_manager = cancellation_manager_.get();
    iterator_ctx_ = std::make_unique<data::IteratorContext>(std::move(params));

    DatasetBase* finalized_dataset;
    TF_RETURN_IF_ERROR(
        data::FinalizeDataset(ctx, dataset_, &finalized_dataset));
    TF_RETURN_IF_ERROR(finalized_dataset->MakeIterator(
        iterator_ctx_.get(), nullptr, "LookupTable", &iterator_));
    core::ScopedUnref unref(finalized_dataset);
    Next();
    return OkStatus();
  }

  void Next() override {
    bool end_of_input;
    tensors_.clear();
    status_ = iterator_->GetNext(iterator_ctx_.get(), &tensors_, &end_of_input);
    if (status_.ok() && end_of_input) {
      status_ = errors::OutOfRange("end of iterator");
    }
  }

  bool Valid() const override { return status_.ok(); }

  const Tensor& keys() const override { return tensors_[0]; }

  const Tensor& values() const override { return tensors_[1]; }

  Status status() const override { return status_; }

  int64_t total_size() const override {
    int64_t size = dataset_->Cardinality();
    if (size < 0) {
      return 0;
    }
    return size;
  }

 private:
  data::DatasetBase* dataset_;  // owned.
  std::unique_ptr<data::IteratorContext> iterator_ctx_;
  std::unique_ptr<FunctionHandleCache> function_handle_cache_;
  ResourceMgr resource_mgr_;
  std::unique_ptr<CancellationManager> cancellation_manager_;
  std::unique_ptr<data::IteratorBase> iterator_;
  std::vector<Tensor> tensors_;
  Status status_;
};

std::unique_ptr<InitializerSerializer> MakeDatasetInitializerSerializer(
    OpKernelContext* ctx, data::DatasetBase* dataset) {
  dataset->Ref();
  auto unref_dataset = [dataset] { dataset->Unref(); };
  return std::make_unique<InitializerSerializer>(
      [dataset, resource_manager = ctx->resource_manager(),
       device_name = ctx->device()->attributes().name()](
          GraphDefBuilder* builder, Node* table, Node** out) {
        data::DatasetBase::DatasetGraphDefBuilder db(builder);
        data::SerializationContext::Params params;
        params.resource_mgr = resource_manager;
        params.device_name = device_name;
        data::SerializationContext serialization_ctx(params);
        Node* dataset_node;
        TF_RETURN_IF_ERROR(
            db.AddInputDataset(&serialization_ctx, dataset, &dataset_node));
        *out = ops::BinaryOp("InitializeTableFromDataset", table, dataset_node,
                             builder->opts());
        if (*out == nullptr) {
          return errors::Internal(
              "Failed to create InitializeTableFromDataset op: ",
              builder->opts().StatusToString());
        }
        return OkStatus();
      },
      /*cleanup=*/std::move(unref_dataset));
}

void InitializeTableFromDataset(OpKernelContext* ctx,
                                data::DatasetBase* dataset,
                                lookup::InitializableLookupTable* table,
                                AsyncOpKernel::DoneCallback done) {
  // Construct the cleanup before `iter` below so that `iter` is destroyed
  // before calling `done`.
  auto cleanup = gtl::MakeCleanup([done = std::move(done)]() { done(); });
  // Assert that the dataset types match up to that expected in the table.
  const auto& dataset_types = dataset->output_dtypes();
  OP_REQUIRES(
      ctx, dataset_types.size() == 2,
      errors::InvalidArgument("Dataset should have two output types only"));
  OP_REQUIRES(ctx, dataset_types[0] == table->key_dtype(),
              errors::InvalidArgument(
                  "Key dtype expected: ", table->key_dtype(),
                  " but obtained: ", dataset_types[0], " from the dataset"));
  OP_REQUIRES(ctx, dataset_types[1] == table->value_dtype(),
              errors::InvalidArgument(
                  "Value dtype expected: ", table->value_dtype(),
                  " but obtained: ", dataset_types[1], " from the dataset"));
  // Assert that the dataset output shapes are scalars.
  const auto& dataset_shapes = dataset->output_shapes();
  OP_REQUIRES(
      ctx, dataset_shapes.size() == 2,
      errors::InvalidArgument("Dataset should have two output shapes only"));
  OP_REQUIRES(ctx, dataset_shapes[0].IsCompatibleWith(PartialTensorShape({})),
              errors::InvalidArgument("Expected scalar for key. Obtained: ",
                                      dataset_shapes[0].DebugString()));
  OP_REQUIRES(ctx, dataset_shapes[1].IsCompatibleWith(PartialTensorShape({})),
              errors::InvalidArgument("Expected scalar for key. Obtained: ",
                                      dataset_shapes[1].DebugString()));
  DatasetIterator iter(dataset);
  OP_REQUIRES_OK(ctx, iter.Init(ctx));
  Status s =
      table->Initialize(iter, MakeDatasetInitializerSerializer(ctx, dataset));
  if (errors::IsFailedPrecondition(s) && table->is_initialized()) {
    LOG(INFO) << "Table already initialized from dataset.";
    return;
  }
  ctx->SetStatus(s);
}

class InitializeTableFromDatasetOp : public AsyncOpKernel {
 public:
  explicit InitializeTableFromDatasetOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(), "initialize_table_from_dataset") {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    lookup::InitializableLookupTable* table;
    OP_REQUIRES_OK_ASYNC(
        ctx, GetInitializableLookupTable("table_handle", ctx, &table), done);
    core::ScopedUnref unref_me(table);
    data::DatasetBase* dataset;
    OP_REQUIRES_OK_ASYNC(
        ctx, GetDatasetFromVariantTensor(ctx->input(1), &dataset), done);
    background_worker_.Schedule([ctx, dataset, table, done]() {
      InitializeTableFromDataset(ctx, dataset, table, done);
    });
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(InitializeTableFromDatasetOp);

  data::BackgroundWorker background_worker_;
};

REGISTER_KERNEL_BUILDER(Name("InitializeTableFromDataset").Device(DEVICE_CPU),
                        InitializeTableFromDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
