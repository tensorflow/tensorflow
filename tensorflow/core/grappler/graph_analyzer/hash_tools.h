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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_HASH_TOOLS_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_HASH_TOOLS_H_

#include <cstddef>

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

// Unfortunately, std::hash provides no way to combine hashes, so everyone
// is copying boost::hash_combine. This is a version that follows Google's
// guidelines on the arguments, and contains only the combination, without
// hashing.
inline void CombineHash(size_t from, size_t* to) {
  *to ^= from + 0x9e3779b9 + (*to << 6) + (*to >> 2);
}

// Combine two hashes in such a way that the order of combination doesn't matter
// (so it's really both commutative and associative). The result is not a very
// high-quality hash but can be used in case if the order of sub-elements must
// not matter in the following comparison. An alternative would be to sort the
// hashes of the sub-elements and then combine them normally in the sorted
// order.
inline void CombineHashCommutative(size_t from, size_t* to) {
  *to = *to + from + 0x9e3779b9;
}

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_HASH_TOOLS_H_
