/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <cstdlib>

#include "absl/strings/match.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/stringpiece.h"

// This is a fuzzer for tensorflow::io::ParseURI.

namespace {

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  std::string uri(reinterpret_cast<const char *>(data), size);
  tensorflow::StringPiece scheme, host, path;
  tensorflow::io::ParseURI(uri, &scheme, &host, &path);

  // If a path is invalid.
  if (path == uri) {
    assert(host == "");
    assert(scheme == "");
  } else {
    assert(absl::StrContains(uri, host));
    assert(absl::StrContains(uri, scheme));
    assert(absl::StrContains(uri, path));
    assert(absl::StrContains(uri, "://"));
  }

  return 0;
}

}  // namespace
