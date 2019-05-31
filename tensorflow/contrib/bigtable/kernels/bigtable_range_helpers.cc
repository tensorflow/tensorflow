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

#include "tensorflow/contrib/bigtable/kernels/bigtable_range_helpers.h"

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {

string MakePrefixEndKey(const string& prefix) {
  string end = prefix;
  while (true) {
    if (end.empty()) {
      return end;
    }
    ++end[end.size() - 1];
    if (end[end.size() - 1] == 0) {
      // Handle wraparound case.
      end = end.substr(0, end.size() - 1);
    } else {
      return end;
    }
  }
}

}  // namespace

/* static */ MultiModeKeyRange MultiModeKeyRange::FromPrefix(string prefix) {
  string end = MakePrefixEndKey(prefix);
  VLOG(1) << "Creating MultiModeKeyRange from Prefix: " << prefix
          << ", with end key: " << end;
  return MultiModeKeyRange(std::move(prefix), std::move(end));
}

/* static */ MultiModeKeyRange MultiModeKeyRange::FromRange(string begin,
                                                            string end) {
  return MultiModeKeyRange(std::move(begin), std::move(end));
}

const string& MultiModeKeyRange::begin_key() const { return begin_; }

const string& MultiModeKeyRange::end_key() const { return end_; }

bool MultiModeKeyRange::contains_key(StringPiece key) const {
  if (StringPiece(begin_) > key) {
    return false;
  }
  if (StringPiece(end_) <= key && !end_.empty()) {
    return false;
  }
  return true;
}

}  // namespace tensorflow
