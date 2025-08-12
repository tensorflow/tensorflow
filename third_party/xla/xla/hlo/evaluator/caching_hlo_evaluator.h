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

#ifndef XLA_HLO_EVALUATOR_CACHING_HLO_EVALUATOR_H_
#define XLA_HLO_EVALUATOR_CACHING_HLO_EVALUATOR_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/evaluator/hlo_evaluator_interface.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/literal.h"
#include "xla/service/dynamic_dimension_inference.h"

namespace xla {

// A decorator class that implements the HloEvaluatorInterface.
//
// This class is used to cache the evaluation results of an HLO computation
// given a specific set of arguments. Files are written to disk in the specified
// cache directory.
//
// There are two modes of operation: kWrite mode will evaluate the wrapped
// evaluator implementation and persist the result to disk. It returns the
// computed result as-is. kRead mode will read a previously-computed result from
// disk and return it. If the result cannot be found, it will return an error.
// The kReadAndEvaluateIfCacheMiss variant is similar but will re-compute the
// result if no cached result is found.
//
// This class is not thread-safe.
class CachingHloEvaluator : public HloEvaluatorInterface {
 public:
  enum Mode {
    // Evaluate with the wrapped evaluator and persist the result to disk.
    kWrite,
    // Read a previously evaluated result from disk. Error if not found.
    kRead,
    // Same as kRead, but fall back to wrapped evaluator in cache miss case.
    kReadAndEvaluateIfCacheMiss
  };

  CachingHloEvaluator(std::unique_ptr<HloEvaluatorInterface> wrapped,
                      std::string cache_dir, Mode mode)
      : wrapped_(std::move(wrapped)), cache_dir_(cache_dir), mode_(mode) {}

  absl::StatusOr<Literal> Evaluate(
      const HloComputation& computation,
      absl::Span<const Literal* const> args) override;

  void ResetVisitStates() override { wrapped_->ResetVisitStates(); }

  void set_dynamic_dimension_inference(
      DynamicDimensionInference* dynamic_dimension_inference) override {
    wrapped_->set_dynamic_dimension_inference(dynamic_dimension_inference);
  }

  void set_use_fast_path(bool value) override {
    wrapped_->set_use_fast_path(value);
  }

  void set_custom_call_handler(CustomCallHandler handler) override {
    wrapped_->set_custom_call_handler(std::move(handler));
  }

 private:
  std::unique_ptr<HloEvaluatorInterface> wrapped_;
  std::string cache_dir_;
  Mode mode_;
};
}  // namespace xla

#endif  // XLA_HLO_EVALUATOR_CACHING_HLO_EVALUATOR_H_
