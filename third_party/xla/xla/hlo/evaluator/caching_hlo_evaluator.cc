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

#include "xla/hlo/evaluator/caching_hlo_evaluator.h"

#include <array>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/literal.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {
absl::StatusOr<std::string> MakeCachingHloEvaluatorCacheKey(
    const HloComputation& computation, absl::Span<const Literal* const> args) {
  tsl::Fprint128 fingerprint =
      tsl::Fingerprint128(computation.ToString(HloPrintOptions::Default()));
  for (const Literal* arg : args) {
    TF_ASSIGN_OR_RETURN(std::string serialized, arg->SerializeAsString());
    fingerprint =
        tsl::FingerprintCat128(fingerprint, tsl::Fingerprint128(serialized));
  }
  const std::array<char, 16> fingerprint_bytes =
      tsl::Fprint128ToBytes(fingerprint);
  const absl::string_view fingerprint_bytes_view(fingerprint_bytes.data(),
                                                 fingerprint_bytes.size());
  return absl::BytesToHexString(fingerprint_bytes_view);
}
}  // namespace

absl::StatusOr<Literal> CachingHloEvaluator::Evaluate(
    const HloComputation& computation, absl::Span<const Literal* const> args) {
  TF_ASSIGN_OR_RETURN(const std::string cache_key,
                      MakeCachingHloEvaluatorCacheKey(computation, args));
  const std::string filename =
      tsl::io::JoinPath(cache_dir_, absl::StrCat(cache_key, ".hloeval"));

  switch (mode_) {
    case Mode::kRead:
    case Mode::kReadAndEvaluateIfCacheMiss: {
      std::string serialized_literal;
      if (const absl::Status status = tsl::ReadFileToString(
              tsl::Env::Default(), filename, &serialized_literal);
          !status.ok()) {
        if (mode_ != kReadAndEvaluateIfCacheMiss) {
          return absl::NotFoundError(absl::StrCat(
              "Failed to read serialized result. ", status.message()));
        }
        LOG(INFO)
            << "Failed to read serialized result. Running wrapped evaluator. "
            << status;
        return wrapped_->Evaluate(computation, args);
      }
      return Literal::DeserializeFromString(serialized_literal);
    }
    case Mode::kWrite: {
      TF_ASSIGN_OR_RETURN(Literal literal,
                          wrapped_->Evaluate(computation, args));
      TF_ASSIGN_OR_RETURN(const std::string serialized_literal,
                          literal.SerializeAsString());
      TF_RETURN_IF_ERROR(tsl::WriteStringToFile(tsl::Env::Default(), filename,
                                                serialized_literal));
      return std::move(literal);
    }
  }
  LOG(FATAL) << "Unknown mode: " << mode_
             << ". Exhaustive switch should not reach here.";
}
}  // namespace xla
