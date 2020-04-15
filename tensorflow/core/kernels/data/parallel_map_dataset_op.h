/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_PARALLEL_MAP_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_PARALLEL_MAP_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"

namespace tensorflow {
namespace data {

class ParallelMapDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "ParallelMap";
  static constexpr const char* const kInputDataset = "input_dataset";
  static constexpr const char* const kOtherArguments = "other_arguments";
  static constexpr const char* const kNumParallelCalls = "num_parallel_calls";
  static constexpr const char* const kFunc = "f";
  static constexpr const char* const kTarguments = "Targuments";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kUseInterOpParallelism =
      "use_inter_op_parallelism";
  static constexpr const char* const kDeterministic = "deterministic";
  static constexpr const char* const kSloppy = "sloppy";
  static constexpr const char* const kPreserveCardinality =
      "preserve_cardinality";

  explicit ParallelMapDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  const int op_version_;
  std::shared_ptr<FunctionMetadata> func_metadata_ = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  bool sloppy_;
  bool preserve_cardinality_;
  DeterminismPolicy deterministic_;
};

class ParallelMapFunctor {
 public:
  virtual ~ParallelMapFunctor() {}

  // A function that runs when the Iterator is initialized. It enables the user
  // to specify error checking logic that can fail early.
  virtual Status InitFunc(IteratorContext* ctx) { return Status::OK(); }

  // Indicates whether the functor depends on any external state.
  // If so, the method returns `errors::FailedPrecondition` with
  // a message that identifies the external state. Otherwise, the method returns
  // `Status::OK()`.
  virtual Status CheckExternalState() = 0;

  // A function that transforms elements of one dataset into another
  // asynchronously. The arguments are:
  // 1. An `IteratorContext*` for the context in which the function should
  // execute.
  // 2. A `std::vector<Tensor>` containing the input element.
  // 3. A `std::vector<Tensor>*` to which the function will write the result.
  // 4. A `StatusCallback` that should be invoked when the function is complete.
  virtual void MapFunc(IteratorContext* ctx, const string& prefix,
                       std::vector<Tensor> input, std::vector<Tensor>* output,
                       StatusCallback callback) = 0;
};

// Returns a new iterator that uses `parallel_map_functor` to apply `MapFunc`
// to the elements of `input_dataset` using the given degree of parallelism.
std::unique_ptr<IteratorBase> NewParallelMapIterator(
    const DatasetBaseIterator::BaseParams& params,
    const DatasetBase* input_dataset,
    std::unique_ptr<ParallelMapFunctor> parallel_map_functor,
    int64 num_parallel_calls, bool deterministic, bool preserve_cardinality);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_PARALLEL_MAP_DATASET_OP_H_
