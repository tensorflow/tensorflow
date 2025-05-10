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

#ifndef XLA_COMPACT_STRING_H_
#define XLA_COMPACT_STRING_H_

#include <string>

#include "absl/strings/string_view.h"

namespace xla {

// Like std::string, except uses only 8 bytes inline, and only allows
// full assignment, rather than editing of sub-pieces.
class CompactString {
 public:
  CompactString() : rep_(nullptr) {}

  CompactString(const CompactString& s) : rep_(nullptr) { set(s.view()); }
  CompactString& operator=(const CompactString& s) {
    set(s.view());
    return *this;
  }

  // Convenience conversions for absl::string_view and std::string
  // to make this more of a drop-in replacement for std::string
  CompactString(absl::string_view s) : rep_(nullptr) { set(s); }
  CompactString& operator=(const absl::string_view& s) {
    set(s);
    return *this;
  }

  CompactString(const std::string& s) : rep_(nullptr) { set(s); }
  CompactString& operator=(const std::string& s) {
    set(s);
    return *this;
  }

  // This is a minimal interface mimicking a small part of the std::string
  // interface. Further additions could emulate more of that interface if
  // needed.

  // Set contents of this to "s"
  void set(absl::string_view s);

  size_t size() const;

  // Return the current string_view.  Remains valid while this object
  // is live and no subsequent calls to set have been made.
  absl::string_view view() const;

  // Automatic conversion so CompactString can be passed where an
  // absl::string_view is expected.  Remains valid while this object
  // is live and no subsequent calls to set have been made.
  operator absl::string_view() const { return view(); }

 private:
  // Points to a new char[] array of exactly enough space to hold the
  // rep, which starts with the length encoded as a varint value,
  // followed by the actual bytes of the string.
  std::unique_ptr<char[]> rep_;
};

}  // namespace xla

#endif  // XLA_COMPACT_STRING_H_
