/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SOURCE_TARGET_PAIRS_H_
#define XLA_SERVICE_SOURCE_TARGET_PAIRS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

//  Encapsulation of source-target pairs used in collective permute.
namespace xla {

struct SourceTargetPair {
  int64_t source;
  int64_t target;
  SourceTargetPair() = default;
  SourceTargetPair(int64_t s, int64_t t) : source(s), target(t) {}
  bool operator==(const SourceTargetPair& other) const {
    return source == other.source && target == other.target;
  }
};

// SourceTargetPairs represents a list of (source, target)
// pairs used in a collective permute instruction
// e.g. {{0,1},{1,2},{2,3},{3,0}}.
class SourceTargetPairs {
  static constexpr int64_t kInlineFactor = 8;

 public:
  using STVector = absl::InlinedVector<SourceTargetPair, kInlineFactor>;

  SourceTargetPairs() = default;

  explicit SourceTargetPairs(
      const std::vector<std::pair<int64_t, int64_t>>& pairs) {
    for (const auto& pair : pairs) {
      emplace_back(pair.first, pair.second);
    }
  }

  static absl::StatusOr<SourceTargetPairs> FromString(absl::string_view str) {
    // reusing replica groups parsing.
    TF_ASSIGN_OR_RETURN(std::vector<ReplicaGroup> groups,
                        // absl::StatusOr<std::vector<ReplicaGroup>> groups =
                        ParseReplicaGroupsOnly(str));
    SourceTargetPairs res;
    for (const ReplicaGroup& group : groups) {
      if (group.replica_ids_size() != 2) {
        return Internal("Incorrect element size : %s", str);
      }
      res.emplace_back(group.replica_ids(0), group.replica_ids(1));
    }
    return res;
  }

  // Returns a cannoical string such as {{0,1},{1,2},{2,3},{3,0}}.
  std::string ToString() const {
    auto formatter = [](std::string* out, const SourceTargetPair& pair) {
      absl::StrAppend(out, "{", pair.source, ",", pair.target, "}");
    };
    const std::string pairs_str = absl::StrJoin(pairs_, ",", formatter);
    return absl::StrCat("{", pairs_str, "}");
  }

  STVector& data() { return pairs_; }
  const STVector& data() const { return pairs_; }

  SourceTargetPair& operator[](int64_t i) {
    CHECK_LT(i, pairs_.size())
        << "Index out of bounds. Size: " << pairs_.size() << " Index: " << i;
    return pairs_[i];
  }
  const SourceTargetPair& operator[](int64_t i) const {
    CHECK_LT(i, pairs_.size())
        << "Index out of bounds. Size: " << pairs_.size() << " Index: " << i;
    return pairs_[i];
  }

  bool operator==(const SourceTargetPairs& other) const {
    return pairs_ == other.pairs_;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SourceTargetPairs& pairs) {
    absl::Format(&sink, "%s", pairs.ToString());
  }

  size_t size() const { return pairs_.size(); }
  void emplace_back(int64_t source, int64_t target) {
    pairs_.emplace_back(source, target);
  }
  void push_back(const SourceTargetPair& pair) { pairs_.push_back(pair); }

  // Converts to a vector of pairs of ints.
  std::vector<std::pair<int64_t, int64_t>> expand() const {
    std::vector<std::pair<int64_t, int64_t>> data;
    for (const auto& pair : pairs_) {
      data.push_back({pair.source, pair.target});
    }
    return data;
  }

  int64_t GetMaxDeviceNum() const {
    int64_t max_device_num = -1;
    for (auto [source, target] : pairs_) {
      max_device_num = std::max(std::max(source, target), max_device_num);
    }
    return max_device_num;
  }

 private:
  STVector pairs_;
};

}  // namespace xla
#endif  // XLA_SERVICE_SOURCE_TARGET_PAIRS_H_
