/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/build_graph_options.h"

#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

string BuildGraphOptions::DebugString() const {
  string rv = "Feed endpoints: ";
  for (auto& s : feed_endpoints) {
    strings::StrAppend(&rv, s, ", ");
  }
  strings::StrAppend(&rv, "\nFetch endpoints: ");
  for (auto& s : fetch_endpoints) {
    strings::StrAppend(&rv, s, ", ");
  }
  strings::StrAppend(&rv, "\nTarget nodes: ");
  for (auto& s : target_nodes) {
    strings::StrAppend(&rv, s, ", ");
  }
  return rv;
}

}  // namespace tensorflow
