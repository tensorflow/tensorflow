/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/scoped_command_buffer_annotation.h"

#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace stream_executor {
namespace {

std::vector<std::string>& GetThreadLocalAnnotations() {
  thread_local std::vector<std::string> annotations;
  return annotations;
}

}  // namespace

ScopedCommandBufferAnnotation::ScopedCommandBufferAnnotation(
    absl::string_view annotation) {
  GetThreadLocalAnnotations().push_back(std::string(annotation));
}

ScopedCommandBufferAnnotation::~ScopedCommandBufferAnnotation() {
  auto& annotations = GetThreadLocalAnnotations();
  if (!annotations.empty()) {
    annotations.pop_back();
  }
}

absl::string_view ScopedCommandBufferAnnotation::GetCurrentAnnotation() {
  const auto& annotations = GetThreadLocalAnnotations();
  if (annotations.empty()) {
    return "";
  }
  return annotations.back();
}

}  // namespace stream_executor
