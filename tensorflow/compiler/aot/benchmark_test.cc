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

#include "tensorflow/compiler/aot/benchmark.h"

#include "tensorflow/compiler/aot/test_graph_tfadd.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace tfcompile {
namespace benchmark {
namespace {

// There isn't much we can verify in a stable fashion, so we just run the
// benchmark with max_iters, and ensure we end up with that many iter stats.
TEST(Benchmark, Benchmark) {
  AddComp add;

  Options options;
  options.max_iters = 1;
  Stats stats1;
  Benchmark(options, [&] { add.Run(); }, &stats1);
  EXPECT_EQ(stats1.per_iter_us.size(), 1);

  options.max_iters = 5;
  Stats stats5;
  Benchmark(options, [&] { add.Run(); }, &stats5);
  EXPECT_EQ(stats5.per_iter_us.size(), 5);
}

}  // namespace
}  // namespace benchmark
}  // namespace tfcompile
}  // namespace tensorflow
