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
#include <limits>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
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
  // Returns the number of memory terms before and after the reduction.
  std::pair<int64_t, int64_t> Reduce(
      int64_t num_lives, int64_t num_primitives,
      const std::function<
          tsl::protobuf::RepeatedField<int64_t>(int64_t)>&  // NOLINT
          live,
      int64_t max_iterations = std::numeric_limits<int64_t>::max());

  // An alternate interface that consumes primitive intervals instead of a
  // liveness matrix.
  std::pair<int64_t, int64_t> Reduce(
      int64_t num_lives, int64_t num_primitives,
      const std::function<std::pair<int64_t, int64_t>(int64_t)>& intervals,
      int64_t max_iterations = std::numeric_limits<int64_t>::max());

  const std::vector<std::vector<int64_t>>& GetReducedLive() const;
  const std::vector<std::pair<int64_t, int64_t>>& GetReducedIntervals() const;
  const std::vector<absl::btree_set<int64_t>>& GetReducedGroups() const;

  // Retrieves a reduced subset of time points along the liveness profile that
  // are sufficient to establish memory constraints.
  absl::flat_hash_set<int64_t> GetReducedTimes(int64_t num_primitives);

  // A static version of the above method (in case we're using a precomputed
  // memory term reduction).
  static absl::flat_hash_set<int64_t> GetReducedTimes(
      int64_t num_primitives,
      const std::vector<std::pair<int64_t, int64_t>>& reduced_intervals,
      const std::vector<absl::btree_set<int64_t>>& reduced_groups);

 private:
  // The internal implementation, agnostic to whether the client uses a liveness
  // matrix or primitive intervals.
  void Reduce(int64_t num_lives, int64_t num_primitives,
              int64_t max_iterations);

  std::vector<std::vector<int64_t>> reduced_live_;
  std::vector<std::pair<int64_t, int64_t>> reduced_intervals_;
  std::vector<absl::btree_set<int64_t>> reduced_groups_;
};

}  // namespace spmd
}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_MEMORY_H_
