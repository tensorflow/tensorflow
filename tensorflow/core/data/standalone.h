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
#ifndef TENSORFLOW_CORE_DATA_STANDALONE_H_
#define TENSORFLOW_CORE_DATA_STANDALONE_H_

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/data/tfdataz_metrics.h"
#include "tensorflow/core/data/unbounded_thread_pool.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/public/session_options.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace data {
namespace standalone {

// The purpose of the API in this file is to facilitate standalone execution of
// a tf.data input pipeline graph.
//
// The API exposes two abstractions -- a `Dataset` and an `Iterator` -- which
// encapsulate TensorFlow runtime.
//
// The `Dataset` abstraction represents an input pipeline as a collection
// of data sources and a logical plan of transformations that operate over the
// data.
//
// The `Iterator` abstraction represents an execution of an input pipeline that
// can be used to enumerate its elements.
//
// Example usage:
//
//   // Create a `Dataset` by running the `graph_def` graph.
//   tensorflow::data:standalone::Dataset::Params params;
//   std::unique_ptr<tensorflow::data::standalone::Dataset> dataset;
//   Status s = tensorflow::data::standalone::Dataset::FromGraph(
//      params, graph_def, &dataset);
//   if (!s.ok()) { /* error handling */ }
//
//   std::unique_ptr<tensorflow::data::standalone::Iterator> iterator;
//   s = dataset->MakeIterator(&iterator);
//   if (!s.ok()) { /* error handling */ }
//
//   bool end_of_input = false;
//   while (!end_of_input) {
//     std::vector<tensorflow::Tensor> outputs;
//     s = iterator->GetNext(&outputs, &end_of_input);
//     if (!s.ok()) { /* error handling */ }
//     if (!end_of_input) { /* output handling */ }
//   }

class Dataset;

// Represents an execution of an input pipeline that can be used to enumerate
// its elements.
class Iterator {
 public:
  virtual ~Iterator();

  // Returns the next element of the input pipeline (if there is one) and an
  // indication of whether the end of the input pipeline has been reached.
  absl::Status GetNext(std::vector<Tensor>* outputs, bool* end_of_input);

  // Saves a checkpoint of the iterator. Returns Tensors that can be called with
  // `Restore()`.
  absl::StatusOr<std::vector<Tensor>> Save();

  // Restores the iterator from a checkpoint. `saved_iterator` is the serialized
  // iterator saved by calling `Save()`.
  absl::Status Restore(const std::vector<Tensor>& saved_iterator);

  // Returns the dataset model for performance analysis.
  std::shared_ptr<model::Model> model() const;

 private:
  friend class Dataset;

  Iterator(IteratorBase* iterator, IteratorContext* ctx,
           SerializationContext* serialization_ctx);

  std::unique_ptr<IteratorBase> iterator_;
  std::unique_ptr<IteratorContext> ctx_;
  std::unique_ptr<SerializationContext> serialization_ctx_;
  std::shared_ptr<TfDatazMetricsCollector> tf_dataz_metrics_collector_;
};

// Represents an input pipeline as a collection of data sources and a logical
// plan of transformations that operate over the data.
class Dataset {
 public:
  // Parameters for `Dataset` creation (e.g. TensorFlow runtime configuration).
  struct Params {
    SessionOptions session_options;
  };

  // Creates a new `Dataset` instance by running the given dataset graph.
  static absl::Status FromGraph(Params params, const GraphDef& graph_def,
                                std::unique_ptr<Dataset>* result);

  ~Dataset();

  // Creates an iterator for this dataset.
  absl::Status MakeIterator(std::unique_ptr<Iterator>* result);
  // Creates an iterator, optionally with a split provider.
  absl::Status MakeIterator(
      std::vector<std::unique_ptr<SplitProvider>> split_providers,
      std::unique_ptr<Iterator>* result);

  // Creates split providers for this dataset.
  absl::Status MakeSplitProviders(
      std::vector<std::unique_ptr<SplitProvider>>* result);
  // Returns a pointer to the underlying dataset.
  const DatasetBase* Get() const;

 private:
  Dataset(DatasetBase* finalized_dataset, DatasetBase* original_dataset,
          DeviceMgr* device_mgr, ProcessFunctionLibraryRuntime* pflr,
          FunctionLibraryDefinition* flib_def, thread::ThreadPool* pool,
          std::function<void(std::function<void()>)> runner);

  DatasetBase* finalized_dataset_;  // owned
  DatasetBase* original_dataset_;   // owned
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  std::unique_ptr<thread::ThreadPool> interop_threadpool_;
  std::unique_ptr<FunctionHandleCache> function_handle_cache_;
  std::function<void(std::function<void()>)> runner_;
  ResourceMgr resource_mgr_;
  CancellationManager cancellation_manager_;
  UnboundedThreadPool unbounded_thread_pool_;
};

}  // namespace standalone
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_STANDALONE_H_
