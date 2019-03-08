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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_BENCHMARK_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_BENCHMARK_UTILS_H_

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
namespace grappler {

GraphDef CreateRandomGraph(int size) {
  random::PhiloxRandom philox(0x12345);
  random::SimplePhilox rnd(&philox);

  string prefix = "long_node_name_prefix_to_measure_string_copy_overhead";

  GraphDef graph;
  for (int i = 0; i < size; ++i) {
    const string name = absl::StrCat(prefix, i);
    const uint32 num_inputs = rnd.Uniform(std::min(i, 5));

    NodeDef node;
    node.set_name(name);
    for (int n = 0; n < num_inputs; ++n) {
      const uint32 input_node = rnd.Uniform(i);
      node.add_input(absl::StrCat(prefix, input_node));
    }

    *graph.add_node() = std::move(node);
  }

  return graph;
}

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_BENCHMARK_UTILS_H_
