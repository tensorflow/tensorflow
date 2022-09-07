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
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/data/root_dataset.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/ptr_util.h"

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

Status Iterator::GetNext(std::vector<Tensor>* outputs, bool* end_of_input) {
  return iterator_->GetNext(ctx_.get(), outputs, end_of_input);
}

Iterator::Iterator(IteratorBase* iterator, IteratorContext* ctx)
    : iterator_(iterator), ctx_(ctx) {}

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
      Rendezvous::Factory{
          [](const int64_t, const DeviceMgr* device_mgr, Rendezvous** r) {
            *r = new IntraProcessRendezvous(device_mgr);
            return OkStatus();
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
  *result = WrapUnique(new Dataset(
      finalized_dataset, dataset, device_mgr.release(), pflr.release(),
      flib_def.release(), pool.release(), std::move(runner)));
  return OkStatus();
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
  ctx = std::make_unique<IteratorContext>(std::move(params));

  // Create the iterator from the dataset.
  std::unique_ptr<IteratorBase> iterator;
  TF_RETURN_IF_ERROR(finalized_dataset_->MakeIterator(
      ctx.get(), /*parent=*/nullptr, "Iterator", &iterator));
  *result = WrapUnique(new Iterator(iterator.release(), ctx.release()));

  return OkStatus();
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
