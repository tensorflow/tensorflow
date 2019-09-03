/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/internal/parse_annotation.h"

#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {
namespace {

std::vector<absl::string_view> SplitNameAndMetadata(
    absl::string_view annotation) {
  std::vector<absl::string_view> parts;
  if (annotation.empty() || annotation.back() != '#') {
    parts.emplace_back(annotation);
  } else {
    annotation.remove_suffix(1);
    parts = absl::StrSplit(annotation, '#', absl::SkipEmpty());
    if (parts.size() > 2) {
      parts.resize(2);
    }
  }
  while (parts.size() < 2) {
    parts.emplace_back();
  }
  return parts;
}

std::vector<std::pair<absl::string_view, absl::string_view>> ParseMetadata(
    absl::string_view metadata) {
  std::vector<std::pair<absl::string_view, absl::string_view>> key_values;
  for (absl::string_view pair : absl::StrSplit(metadata, ',')) {
    std::vector<absl::string_view> parts = absl::StrSplit(pair, '=');
    if (parts.size() == 2 && !parts[0].empty() && !parts[1].empty()) {
      key_values.push_back(std::make_pair(parts[0], parts[1]));
    }
  }
  return key_values;
}

}  // namespace

Annotation ParseAnnotation(absl::string_view annotation) {
  Annotation result;
  std::vector<absl::string_view> parts = SplitNameAndMetadata(annotation);
  if (!parts.empty()) {
    result.name = parts[0];
    for (const auto& key_value : ParseMetadata(parts[1])) {
      result.metadata.push_back({key_value.first, key_value.second});
    }
  }
  return result;
}

std::vector<Annotation> ParseAnnotationStack(
    absl::string_view annotation_stack) {
  std::vector<Annotation> annotations;
  const std::string kAnnotationDelimiter = "::";
  for (absl::string_view annotation : absl::StrSplit(
           annotation_stack, kAnnotationDelimiter, absl::SkipEmpty())) {
    annotations.emplace_back(ParseAnnotation(annotation));
  }
  return annotations;
}

}  // namespace profiler
}  // namespace tensorflow
