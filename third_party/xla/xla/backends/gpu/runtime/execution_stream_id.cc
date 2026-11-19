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

#include "xla/backends/gpu/runtime/execution_stream_id.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xla/side_effect_util.h"

namespace xla::gpu {

absl::StatusOr<LocalityDomainStreamId> ParseLocalityDomainStreamAnnotation(
    absl::string_view annotation) {
  absl::string_view domain = annotation;
  if (!absl::ConsumePrefix(&domain, kXlaLocalityDomainStreamAnnotationPrefix) ||
      domain.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid locality-domain stream annotation: expected "
                     "\"locality_domain:<d>\", got \"",
                     annotation, "\""));
  }
  for (char c : domain) {
    if (!absl::ascii_isdigit(static_cast<unsigned char>(c))) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid locality-domain stream annotation: domain id must be an "
          "unsigned decimal integer, got \"",
          domain, "\""));
    }
  }
  uint64_t domain_id;
  if (!absl::SimpleAtoi(domain, &domain_id)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid locality-domain stream annotation: domain id is out of "
        "range, got \"",
        domain, "\""));
  }
  return LocalityDomainStreamId(domain_id);
}

}  // namespace xla::gpu
