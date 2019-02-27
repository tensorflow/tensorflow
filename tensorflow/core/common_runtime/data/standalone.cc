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

#include "tensorflow/core/common_runtime/data/standalone.h"

#include <memory>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace standalone {

Status Iterator::GetNext(std::vector<Tensor>* outputs, bool* end_of_input) {
  return iterator_->GetNext(ctx_.get(), outputs, end_of_input);
}

Iterator::Iterator(IteratorBase* iterator, IteratorContext* ctx)
    : iterator_(iterator), ctx_(ctx) {}

Status Dataset::FromGraph(Params params, const GraphDef& graph_def,
                          const string& fetch_node,
                          std::unique_ptr<Dataset>* result) {
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ImportGraphDef({}, graph_def, &graph, nullptr));

  // Instantiate enough of the TensorFlow runtime to run `graph` on a single CPU
  // device.
  std::unique_ptr<DeviceMgr> device_mgr = MakeUnique<DeviceMgr>(
      DeviceFactory::NewDevice("CPU", params.session_options, ""));
  Device* device = device_mgr->ListDevices()[0];
  // Clone the `FunctionLibraryDefinition` to extend its lifetime extends beyond
  // the lifetime of `graph`.
  std::unique_ptr<FunctionLibraryDefinition> flib_def =
      MakeUnique<FunctionLibraryDefinition>(graph.flib_def());
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr =
      MakeUnique<ProcessFunctionLibraryRuntime>(
          device_mgr.get(), Env::Default(), TF_GRAPH_DEF_VERSION,
          flib_def.get(), OptimizerOptions{}, nullptr /* parent */);

  // Run graph up to `output_node` and extract the `DatasetBase` stored in the
  // DT_VARIANT output tensor.
  data::DatasetBase* dataset;
  {
    std::vector<Tensor> outputs;
    GraphRunner graph_runner(device);
    TF_RETURN_IF_ERROR(graph_runner.Run(&graph, pflr->GetFLR("/device:CPU:0"),
                                        {}, {fetch_node}, &outputs));
    TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(outputs[0], &dataset));
    // NOTE(mrry): The dataset is currently owned by `outputs[0]`, so acquire an
    // additional reference.
    dataset->Ref();
  }

  std::unique_ptr<thread::ThreadPool> pool(
      NewThreadPoolFromSessionOptions(params.session_options));
  *result =
      WrapUnique(new Dataset(dataset, device_mgr.release(), pflr.release(),
                             flib_def.release(), pool.release()));
  return Status::OK();
}  // static

Status Dataset::MakeIterator(std::unique_ptr<Iterator>* result) {
  // Create an `IteratorContext`, which bundles together the necessary runtime
  // support to create and get elements from an iterator.
  std::unique_ptr<IteratorContext> ctx;
  {
    // NOTE(mrry): In the current API, an `IteratorContext` is always initially
    // created from an `OpKernelContext*`, so we need to create a fake
    // `OpKernelContext` with the appropriate subset of parameters.
    OpKernelContext::Params op_params;
    op_params.function_library = pflr_->GetFLR("/device:CPU:0");
    op_params.device = device_mgr_->ListDevices()[0];
    op_params.runner = &runner_;
    OpKernelContext op_ctx(&op_params, 0);
    IteratorContext::Params params(&op_ctx);
    params.function_handle_cache = function_handle_cache_.get();
    ctx = MakeUnique<IteratorContext>(std::move(params));
  }

  // Create the iterator from the dataset.
  std::unique_ptr<IteratorBase> iterator;
  TF_RETURN_IF_ERROR(dataset_->MakeIterator(ctx.get(), "iterator", &iterator));

  *result = WrapUnique(new Iterator(iterator.release(), ctx.release()));

  return Status::OK();
}

Dataset::Dataset(DatasetBase* dataset, DeviceMgr* device_mgr,
                 ProcessFunctionLibraryRuntime* pflr,
                 FunctionLibraryDefinition* flib_def, thread::ThreadPool* pool)
    : dataset_(dataset),
      device_mgr_(device_mgr),
      flib_def_(flib_def),
      pflr_(pflr),
      pool_(pool) {
  runner_ = [this](std::function<void()> c) { pool_->Schedule(std::move(c)); };
  function_handle_cache_ =
      MakeUnique<FunctionHandleCache>(pflr_->GetFLR("/device:CPU:0"));
}

Dataset::~Dataset() { dataset_->Unref(); }

}  // namespace standalone
}  // namespace data
}  // namespace tensorflow
