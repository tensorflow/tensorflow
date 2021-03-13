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

#include <memory>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/public/session_options.h"

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
  // Returns the next element of the input pipeline (if there is one) and an
  // indication of whether the end of the input pipeline has been reached.
  Status GetNext(std::vector<Tensor>* outputs, bool* end_of_input);

 private:
  friend class Dataset;

  Iterator(IteratorBase* iterator, IteratorContext* ctx);

  std::unique_ptr<IteratorBase> iterator_;
  std::unique_ptr<IteratorContext> ctx_;
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
  static Status FromGraph(Params params, const GraphDef& graph_def,
                          std::unique_ptr<Dataset>* result);

  ~Dataset();

  // Creates an iterator for this dataset.
  Status MakeIterator(std::unique_ptr<Iterator>* result);
  // Creates an iterator, optionally with a split provider.
  Status MakeIterator(std::unique_ptr<SplitProvider> split_provider,
                      std::unique_ptr<Iterator>* result);

  // Creates a split provider for this dataset.
  Status MakeSplitProvider(std::unique_ptr<SplitProvider>* result);
  // Returns a pointer to the underlying dataset.
  const DatasetBase* Get() const;

 private:
  Dataset(DatasetBase* dataset, DeviceMgr* device_mgr,
          ProcessFunctionLibraryRuntime* pflr,
          FunctionLibraryDefinition* flib_def, thread::ThreadPool* pool);

  DatasetBase* dataset_;  // owned
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  std::unique_ptr<thread::ThreadPool> pool_;
  std::function<void(std::function<void()>)> runner_;
  ResourceMgr resource_mgr_;
  CancellationManager cancellation_manager_;
};

}  // namespace standalone
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_STANDALONE_H_
