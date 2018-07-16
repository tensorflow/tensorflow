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

#ifndef TENSORFLOW_CONTRIB_BIGTABLE_KERNELS_BIGTABLE_RANGE_HELPERS_H_
#define TENSORFLOW_CONTRIB_BIGTABLE_KERNELS_BIGTABLE_RANGE_HELPERS_H_

#include <string>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Represents a continuous range of keys defined by either a prefix or a range.
//
// Ranges are represented as "half-open", where the beginning key is included
// in the range, and the end_key is the first excluded key after the range.
//
// The range of keys can be specified either by a key prefix, or by an explicit
// begin key and end key. All methods on this class are valid no matter which
// way the range was specified.
//
// Example:
//   MultiModeKeyRange range = MultiModeKeyRange::FromPrefix("myPrefix");
//   if (range.contains_key("myPrefixedKey")) {
//     LOG(INFO) << "range from " << range.begin_key() << " to "
//               << range.end_key() << "contains \"myPrefixedKey\"";
//   }
//   if (!range.contains_key("randomKey")) {
//     LOG(INFO) << "range does not contain \"randomKey\"";
//   }
//   range = MultiModeKeyRange::FromRange("a_start_key", "z_end_key");
class MultiModeKeyRange {
 public:
  static MultiModeKeyRange FromPrefix(string prefix);
  static MultiModeKeyRange FromRange(string begin, string end);

  // The first valid key in the range.
  const string& begin_key() const;
  // The first invalid key after the valid range.
  const string& end_key() const;
  // Returns true if the provided key is a part of the range, false otherwise.
  bool contains_key(StringPiece key) const;

 private:
  MultiModeKeyRange(string begin, string end)
      : begin_(std::move(begin)), end_(std::move(end)) {}

  const string begin_;
  const string end_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_BIGTABLE_KERNELS_BIGTABLE_RANGE_HELPERS_H_
