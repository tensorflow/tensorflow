/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DTENSOR_CC_PARALLEL_EXECUTOR_H_
#define TENSORFLOW_DTENSOR_CC_PARALLEL_EXECUTOR_H_

#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace dtensor {

// ParallelExecutor Interface
class ParallelExecutor {
 public:
  virtual ~ParallelExecutor() = default;
  // Note: The API is under development and subject to change.
  virtual tsl::Status Execute() const = 0;
};

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_CC_PARALLEL_EXECUTOR_H_
