/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static void BM_DisabledVlog(::testing::benchmark::State& state) {
  for (auto s : state) {
    VLOG(1) << "Testing VLOG(1)!";
  }
}
BENCHMARK(BM_DisabledVlog);

}  // namespace tensorflow
