/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_UTILS_H_

#include <string>
#include <vector>

#include "absl/flags/declare.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow::serving {
enum class BatchPaddingPolicy;  // Forward-declaring for the ABSL_DECLARE_FLAG.
}  // namespace tensorflow::serving

// Exposed for testing only.
ABSL_DECLARE_FLAG(tensorflow::serving::BatchPaddingPolicy,
                  tensorflow_batch_padding_policy);

namespace tensorflow {
namespace serving {

// Returns the next allowed batch size, which is the smallest allowed batch size
// greater than or equal to the given batch size. If allowed_batch_sizes,
// returns batch_size as is.
int GetNextAllowedBatchSize(int batch_size,
                            const std::vector<int32>& allowed_batch_sizes,
                            bool disable_padding);

// Returns the largest allowed batch size that is smaller than or equal to
// batch_size. Returns batch_size if no such size exists.
int GetPrevAllowedBatchSize(int batch_size,
                            const std::vector<int32>& allowed_batch_sizes,
                            bool disable_padding);

// See the description of the --tensorflow_batch_padding_policy flag (in the
// .cc file) for the documentation.
enum class BatchPaddingPolicy {
  kPadUp,
  kBatchDown,
  kMinimizeTpuCostPerRequest,
};

bool AbslParseFlag(absl::string_view text, BatchPaddingPolicy* out,
                   std::string* error);
std::string AbslUnparseFlag(BatchPaddingPolicy in);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_UTILS_H_
