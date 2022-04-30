/*
 * Copyright 2022 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_JIT_DEFAULT_TF_JITRT_QUERY_OF_DEATH_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_JIT_DEFAULT_TF_JITRT_QUERY_OF_DEATH_H_

#include "absl/strings/string_view.h"

namespace tensorflow {

// Note: this class is just a no-op interface. An actual implementation would
// track "live" JitRt queries through the lifetime of this object.
// On a crash, this can help find the originating "query of death".
class TfJitRtQueryOfDeathLogger {
 public:
  TfJitRtQueryOfDeathLogger(absl::string_view kernel_name,
                            absl::string_view kernel_serialized_operation,
                            absl::string_view operands) {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_JIT_DEFAULT_TF_JITRT_QUERY_OF_DEATH_H_
