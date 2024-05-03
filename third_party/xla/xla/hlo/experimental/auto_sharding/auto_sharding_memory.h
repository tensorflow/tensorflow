/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_MEMORY_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_MEMORY_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace spmd {

// Reduces the # of terms in a liveness matrix by collapsing co-occurring terms:
//
//  |                      |  444466666555
//  |      333333333  ==>  |      ........3  Groups:
//  |      22222222   ==>  |      ........     m[4] = m[0] + m[1]
//  |  111111111      ==>  |  .........        m[5] = m[2] + m[3]
//  | 0000000000      ==>  | 0.........        m[6] = m[0] + m[1] + m[2] + m[3]
//  +-------------->  ==>  +-------------->
//       (time)                 (time)
//
// In the above example, we have four overlapping primitives (0, 1, 2, and 3)
// that are alive for up to ten time units each.  To enforce all memory
// constraints, the encoding on the left requires thirty-six terms in the Mixed
// ILP.  The encoding on the right requires only fourteen terms, plus eight more
// to model some new groups (4, 5, and 6) formed from the others.
class MemoryTermReducer {
 public:
  int64_t Reduce(
      int64_t num_lives, int64_t num_primitives,
      std::function<tsl::protobuf::RepeatedField<int64_t>(int64_t)>  // NOLINT
          live);
  const std::vector<std::vector<int64_t>>& GetReducedLive() const;
  const std::vector<absl::flat_hash_set<int64_t>>& GetReducedGroups() const;

 private:
  std::vector<std::vector<int64_t>> reduced_live_;
  std::vector<absl::flat_hash_set<int64_t>> reduced_groups_;
};

}  // namespace spmd
}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_MEMORY_H_
