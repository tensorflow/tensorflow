/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass_internal.h"

#include <limits>

#include "absl/random/random.h"

namespace tensorflow {
namespace {

static int64_t overridden_node_id = -1;

}  // namespace

namespace internal {

void OverrideNodeIdForTesting(const int64_t node_id) {
  overridden_node_id = node_id;
}

uint64 GetNodeId() {
  static absl::BitGen bitgen;
  if (overridden_node_id > -1) {
    return overridden_node_id;
  } else {
    return absl::Uniform(bitgen, uint64{0}, std::numeric_limits<uint64>::max());
  }
}

}  // namespace internal
}  // namespace tensorflow
