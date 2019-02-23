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

#include "tensorflow/core/platform/numa.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace internal {

TEST(Numa, NumNodes) {
  if (port::NUMAEnabled()) {
    EXPECT_GE(port::NUMANumNodes(), 1);
  }
}

TEST(Numa, Malloc) {
  if (port::NUMAEnabled()) {
    int num_nodes = port::NUMANumNodes();
    for (int request_node = 0; request_node < num_nodes; ++request_node) {
      void* ptr = port::NUMAMalloc(request_node, 8, 0);
      EXPECT_NE(ptr, nullptr);
      // Affinity cannot be tested until page is touched, so save a value.
      *(reinterpret_cast<int*>(ptr)) = 0;
      int affinity_node = port::NUMAGetMemAffinity(ptr);
      EXPECT_EQ(affinity_node, request_node);
      port::NUMAFree(ptr, 8);
    }
  }
}

TEST(Numa, SetNodeAffinity) {
  // NOTE(tucker): This test is not reliable when executed under tap because
  // the virtual machine may not have access to all of the available NUMA
  // nodes.  Not sure what to do about that.
  EXPECT_EQ(-1, port::NUMAGetThreadNodeAffinity());
  if (port::NUMAEnabled()) {
    int num_nodes = port::NUMANumNodes();
    for (int request_node = 0; request_node < num_nodes; ++request_node) {
      port::NUMASetThreadNodeAffinity(request_node);
      int affinity_node = port::NUMAGetThreadNodeAffinity();
      EXPECT_EQ(affinity_node, request_node);
    }
  }
}

}  // namespace internal
}  // namespace tensorflow
