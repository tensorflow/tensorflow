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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_MAP_TOOLS_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_MAP_TOOLS_H_

#include <functional>

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

// Helpers for building maps of pointers.

template <typename Ptr>
struct LessAtPtr : std::function<bool(Ptr, Ptr)> {
  bool operator()(const Ptr& x, const Ptr& y) const { return *x < *y; }
};

template <typename Ptr>
struct EqAtPtr : std::function<bool(Ptr, Ptr)> {
  bool operator()(const Ptr& x, const Ptr& y) const { return *x == *y; }
};

template <typename Ptr>
struct HashAtPtr : std::function<size_t(Ptr)> {
  size_t operator()(const Ptr& x) const { return x->Hash(); }
};

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_MAP_TOOLS_H_
