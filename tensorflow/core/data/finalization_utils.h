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

#ifndef TENSORFLOW_CORE_DATA_FINALIZATION_UTILS_H_
#define TENSORFLOW_CORE_DATA_FINALIZATION_UTILS_H_

#include <functional>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace data {

// Returns the finalized version of the dataset. The returned DatasetBase is
// unowned and lives for as long as this dataset.
StatusOr<DatasetBase*> GetFinalizedDataset(OpKernelContext* ctx,
                                           const DatasetBase* dataset);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_FINALIZATION_UTILS_H_
