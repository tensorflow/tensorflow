/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_BUILD_GRAPH_OPTIONS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_BUILD_GRAPH_OPTIONS_H_

#include <vector>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {

struct BuildGraphOptions {
  std::vector<string> feed_endpoints;
  std::vector<string> fetch_endpoints;

  // TODO(vrv): Remove this when we unify target_nodes and fetch_endpoint,
  // the former via "ref" fetch_endpoints.
  std::vector<string> target_nodes;

  string DebugString() const;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_BUILD_GRAPH_OPTIONS_H_
