// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_OUTSTREAM_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_OUTSTREAM_H_

#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"

namespace litert::tools {

using OutStream = std::reference_wrapper<std::ostream>;
using OutStreamPtr = std::unique_ptr<std::ostream>;

// Out stream configured by a user by flag.
class UserStream {
 public:
  // Parse the flag and get a configured stream.
  static UserStream MakeFromFlag(absl::string_view flag) {
    if (flag == kCerr) {
      LITERT_LOG(LITERT_INFO, "Setup cerr stream\n", "");
      return UserStream(std::cerr);
    } else if (flag == kCout) {
      LITERT_LOG(LITERT_INFO, "Setup cout stream\n", "");
      return UserStream(std::cout);
    } else if (flag == kNone) {
      LITERT_LOG(LITERT_INFO, "Setup null stream\n", "");
      return UserStream();
    } else {
      // File stream.
      LITERT_LOG(LITERT_INFO, "Setup file stream\n", "");
      auto ofstream = std::make_unique<std::ofstream>();
      ofstream->open(flag.data());
      return UserStream(std::move(ofstream));
    }
  }

  // Get the actual stream to write to.
  OutStream Get() { return used_; }

  // Silent stream.
  UserStream()
      : stored_(std::make_unique<std::ostream>(nullptr)), used_(*stored_) {}
  // From reference to external stream (cerr, cout)
  explicit UserStream(OutStream ostream) : stored_(nullptr), used_(ostream) {}
  // From stream to internalize.
  explicit UserStream(OutStreamPtr ostream)
      : stored_(std::move(ostream)), used_(*stored_) {}

  UserStream(UserStream&&) = default;
  UserStream& operator=(UserStream&&) = default;

 private:
  // These are used in the various CLI's flags that configure output streams.
  static constexpr absl::string_view kCerr = "--";
  static constexpr absl::string_view kCout = "-";
  static constexpr absl::string_view kNone = "none";

  OutStreamPtr stored_;
  OutStream used_;
};

}  // namespace litert::tools

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_OUTSTREAM_H_
