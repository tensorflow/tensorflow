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

#include "tensorflow/core/data/standalone.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/root_dataset.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/data/tf_data_memory_logger.h"
#include "tensorflow/core/data/tfdataz_metrics.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/version.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/refcount.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace data {
namespace standalone {

namespace {

OpKernelContext::Params CreateParams(
    ProcessFunctionLibraryRuntime* pflr, DeviceMgr* device_mgr,
    std::function<void(std::function<void()>)>* runner) {
  OpKernelContext::Params params;
  params.function_library = pflr->GetFLR("/device:CPU:0");
  params.device = device_mgr->ListDevices()[0];
  params.runner = runner;
  return params;
}

}  // namespace

Iterator::Iterator(IteratorBase* iterator, IteratorContext* ctx,
                   SerializationContext* serialization_ctx)
    : iterator_(iterator), ctx_(ctx), serialization_ctx_(serialization_ctx) {
  if (DatasetBaseIterator* dataset_iterator =
          dynamic_cast<DatasetBaseIterator*>(iterator_.get())) {
    tf_dataz_metrics_collector_ = std::make_shared<TfDatazMetricsCollector>(
        *Env::Default(), dataset_iterator, ctx_->model());
    TfDatazMetricsRegistry::Register(tf_dataz_metrics_collector_);
    EnsureIteratorMemoryLoggerStarted();
  }
}

Iterator::~Iterator() {
  if (tf_dataz_metrics_collector_) {
    TfDatazMetricsRegistry::Deregister(tf_dataz_metrics_collector_);
  }
}

Status Iterator::GetNext(std::vector<Tensor>* outputs, bool* end_of_input) {
  return iterator_->GetNext(ctx_.get(), outputs, end_of_input);
}

absl::StatusOr<std::vector<Tensor>> Iterator::Save() {
  VariantTensorDataWriter writer;
  TF_RETURN_IF_ERROR(iterator_->Save(serialization_ctx_.get(), &writer));
  std::vector<std::unique_ptr<VariantTensorData>> data;
  writer.ReleaseData(&data);

  std::vector<Tensor> serialized;
  for (size_t i = 0; i < data.size(); ++i) {
    Tensor tensor(DT_VARIANT, TensorShape({1}));
    IteratorStateVariant variant;
    TF_RETURN_IF_ERROR(variant.InitializeFromVariantData(std::move(data[i])));
    tensor.vec<Variant>()(0) = std::move(variant);
    serialized.push_back(std::move(tensor));
  }
  return serialized;
}

Status Iterator::Restore(const std::vector<Tensor>& saved_iterator) {
  std::vector<const VariantTensorData*> data;
  data.reserve(saved_iterator.size());
  for (int i = 0; i < saved_iterator.size(); ++i) {
    auto saved_vec = saved_iterator[i].vec<Variant>();
    auto* variant = saved_vec(0).get<IteratorStateVariant>();
    if (!variant) {
      return errors::Internal(
          "Cannot initialize an iterator from tensor ",
          saved_vec(0).DebugString(),
          ". Expected a variant tensor of type IteratorStateVariant.");
    }
    data.push_back(variant->GetData());
  }
  VariantTensorDataReader reader(data);
  return iterator_->Restore(ctx_.get(), &reader);
}

std::shared_ptr<model::Model> Iterator::model() const { return ctx_->model(); }

Status Dataset::FromGraph(Params params, const GraphDef& graph_def,
                          std::unique_ptr<Dataset>* result) {
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ImportGraphDef({}, graph_def, &graph, nullptr));

  // Instantiate enough of the TF runtime to run `graph` on a single CPU device.
  auto device_mgr = std::make_unique<StaticDeviceMgr>(DeviceFactory::NewDevice(
      "CPU", params.session_options, "/job:localhost/replica:0/task:0"));
  Device* device = device_mgr->ListDevices()[0];
  // Create a copy of the `FunctionLibraryDefinition` to extend lifetime beyond
  // the lifetime of `graph`.
  auto flib_def = std::make_unique<FunctionLibraryDefinition>(
      OpRegistry::Global(), graph_def.library());
  auto pflr = std::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr.get(), Env::Default(), /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, flib_def.get(), OptimizerOptions{},
      /*thread_pool=*/nullptr, /*parent=*/nullptr,
      /*session_metadata=*/nullptr,
      Rendezvous::Factory{[](const int64_t, const DeviceMgr* device_mgr,
                             tsl::core::RefCountPtr<Rendezvous>* r) {
        *r = tsl::core::RefCountPtr<Rendezvous>(
            new IntraProcessRendezvous(device_mgr));
        return absl::OkStatus();
      }});

  string fetch_node = "";
  for (const auto& node : graph_def.node()) {
    if (node.op() == "_Retval") {
      fetch_node = node.input(0);
    }
  }
  if (fetch_node.empty()) {
    return errors::NotFound("Failed to find a _Retval op in the given dataset");
  }

  // Run graph up to `output_node` and extract the `DatasetBase` stored in the
  // DT_VARIANT output tensor.
  std::vector<Tensor> outputs;
  GraphRunner graph_runner(device);
  TF_RETURN_IF_ERROR(graph_runner.Run(&graph, pflr->GetFLR("/device:CPU:0"), {},
                                      {fetch_node}, &outputs));
  data::DatasetBase* dataset;
  TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(outputs[0], &dataset));

  data::DatasetBase* finalized_dataset;
  std::unique_ptr<thread::ThreadPool> pool(
      NewThreadPoolFromSessionOptions(params.session_options));
  std::function<void(std::function<void()>)> runner =
      [&pool](std::function<void()> c) { pool->Schedule(std::move(c)); };
  OpKernelContext::Params op_params =
      CreateParams(pflr.get(), device_mgr.get(), &runner);
  OpKernelContext ctx(&op_params, /*num_outputs=*/0);
  TF_RETURN_IF_ERROR(data::FinalizeDataset(&ctx, dataset, &finalized_dataset));
  core::ScopedUnref unref(finalized_dataset);
  *result = absl::WrapUnique(new Dataset(
      finalized_dataset, dataset, device_mgr.release(), pflr.release(),
      flib_def.release(), pool.release(), std::move(runner)));
  return absl::OkStatus();
}  // static

Status Dataset::MakeIterator(
    std::vector<std::unique_ptr<SplitProvider>> split_providers,
    std::unique_ptr<Iterator>* result) {
  // Create an `IteratorContext`, which bundles together the necessary runtime
  // support to create and get elements from an iterator.
  std::unique_ptr<IteratorContext> ctx;
  // NOTE(mrry): In the current API, an `IteratorContext` is always initially
  // created from an `OpKernelContext*`, so we need to create `OpKernelContext`
  // with a valid subset of parameters.
  OpKernelContext::Params op_params =
      CreateParams(pflr_.get(), device_mgr_.get(), &runner_);
  OpKernelContext op_ctx(&op_params, /*num_outputs=*/0);
  IteratorContext::Params params(&op_ctx);
  params.cancellation_manager = &cancellation_manager_;
  params.function_handle_cache = function_handle_cache_.get();
  params.resource_mgr = &resource_mgr_;
  std::move(split_providers.begin(), split_providers.end(),
            std::back_inserter(params.split_providers));
  params.thread_factory = unbounded_thread_pool_.get_thread_factory();
  params.thread_pool = &unbounded_thread_pool_;
  // The model should only be created if autotuning is on.
  if (ShouldUseAutotuning(finalized_dataset_->options())) {
    params.model = std::make_shared<model::Model>();
  }
  params.run_mode = RunMode::STANDALONE;
  ctx = std::make_unique<IteratorContext>(std::move(params));
  SerializationContext::Params serialization_params(&op_ctx);
  auto serialization_ctx =
      std::make_unique<SerializationContext>(std::move(serialization_params));

  // Create the iterator from the dataset.
  std::unique_ptr<IteratorBase> iterator;
  TF_RETURN_IF_ERROR(finalized_dataset_->MakeIterator(
      ctx.get(), /*parent=*/nullptr, "Iterator", &iterator));
  *result = absl::WrapUnique(new Iterator(iterator.release(), ctx.release(),
                                          serialization_ctx.release()));
  return absl::OkStatus();
}

Status Dataset::MakeIterator(std::unique_ptr<Iterator>* result) {
  return MakeIterator(/*split_providers=*/{}, result);
}

Status Dataset::MakeSplitProviders(
    std::vector<std::unique_ptr<SplitProvider>>* result) {
  return finalized_dataset_->MakeSplitProviders(result);
}

const DatasetBase* Dataset::Get() const { return finalized_dataset_; }

Dataset::Dataset(DatasetBase* finalized_dataset, DatasetBase* original_dataset,
                 DeviceMgr* device_mgr, ProcessFunctionLibraryRuntime* pflr,
                 FunctionLibraryDefinition* flib_def, thread::ThreadPool* pool,
                 std::function<void(std::function<void()>)> runner)
    : finalized_dataset_(finalized_dataset),
      original_dataset_(original_dataset),
      device_mgr_(device_mgr),
      flib_def_(flib_def),
      pflr_(pflr),
      interop_threadpool_(pool),
      runner_(std::move(runner)),
      unbounded_thread_pool_(Env::Default(), "tf_data_standalone") {
  finalized_dataset_->Ref();
  original_dataset_->Ref();
  function_handle_cache_ =
      std::make_unique<FunctionHandleCache>(pflr_->GetFLR("/device:CPU:0"));
}

Dataset::~Dataset() {
  finalized_dataset_->Unref();
  original_dataset_->Unref();
}

}  // namespace standalone
}  // namespace data
}  // namespace tensorflow
