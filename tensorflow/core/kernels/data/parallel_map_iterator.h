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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_PARALLEL_MAP_ITERATOR_H_
#define TENSORFLOW_CORE_KERNELS_DATA_PARALLEL_MAP_ITERATOR_H_

#include <memory>

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

// A function that transforms elements of one dataset into another
// asynchronously. The arguments are:
// 1. An `IteratorContext*` for the context in which the function should
// execute.
// 2. A `std::vector<Tensor>` containing the input element.
// 3. A `std::vector<Tensor>*` to which the function will write the result.
// 4. A `StatusCallback` that should be invoked when the function is complete.
using ParallelMapIteratorFunction =
    std::function<void(IteratorContext*, std::vector<Tensor>,
                       std::vector<Tensor>*, StatusCallback)>;

// Returns a new iterator that applies `map_func` to the elements of
// `input_dataset` using the given degree of parallelism. `init_func` (if
// specified) will be executed when the iterator is initialized (see
// `IteratorBase::Initialize()`) and enables the user to specify error checking
// logic that can fail early.
std::unique_ptr<IteratorBase> NewParallelMapIterator(
    const DatasetBaseIterator::BaseParams& params,
    const DatasetBase* input_dataset,
    std::function<Status(IteratorContext*)> init_func,
    ParallelMapIteratorFunction map_func, int32 num_parallel_calls);
std::unique_ptr<IteratorBase> NewParallelMapIterator(
    const DatasetBaseIterator::BaseParams& params,
    const DatasetBase* input_dataset, ParallelMapIteratorFunction map_func,
    int32 num_parallel_calls);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_PARALLEL_MAP_ITERATOR_H_
