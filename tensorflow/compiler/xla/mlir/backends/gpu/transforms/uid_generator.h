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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_GPU_TRANSFORMS_UID_GENERATOR_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_GPU_TRANSFORMS_UID_GENERATOR_H_

#include <atomic>

namespace xla {
namespace gpu {

// Every stateful operation in the module gets assigned a unique id, that is
// passed to the custom call handler. This id is used for caching resources
// between the different invocations of the same custom call (e.g. cache
// convolution descriptors).
//
// TODO(b/255600288): Improve stateful custom calls in Xla runtime.
class UidGenerator {
 public:
  UidGenerator() : uid_(0) {}
  int64_t uid() { return uid_.fetch_add(1); }

 private:
  std::atomic<int64_t> uid_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_GPU_TRANSFORMS_UID_GENERATOR_H_
