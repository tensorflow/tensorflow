/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/regularization/util.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow::graph_regularization {

uint64 ComputeHash(const GraphDef& graph_def) {
  std::string graph_def_string;
  SerializeToStringDeterministic(graph_def, &graph_def_string);
  return tensorflow::Fingerprint64(graph_def_string);
}

absl::StatusOr<int64_t> GetSuffixUID(absl::string_view function_name) {
  std::vector<absl::string_view> v = absl::StrSplit(function_name, '_');

  int64_t uid;
  if (!strings::safe_strto64(v.back(), &uid)) {
    return errors::InvalidArgument(absl::StrCat(
        "Function name: `", function_name, "` does not end in an integer."));
  }

  return uid;
}
}  // namespace tensorflow::graph_regularization
