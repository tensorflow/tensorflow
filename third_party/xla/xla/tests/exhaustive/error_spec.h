/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_TESTS_EXHAUSTIVE_ERROR_SPEC_H_
#define XLA_TESTS_EXHAUSTIVE_ERROR_SPEC_H_

#include <cstdint>
#include <utility>

#include "xla/xla_data.pb.h"

namespace xla {
namespace exhaustive_op_test {

class ErrorSpecBuilder;

// Specify error tolerances for comparison between a value produced on device
// and its expectation.
//
// N.B.: It is our intention to eventually merge this with the one at
// tensorflow/compiler/xla/error_spec.h. All functionality available here will
// remain, but the majority of the error tolerances will be taken from the other
// one. This type will become either a typealias of or a child class inheriting
// from the other.
struct ErrorSpec {
  using Builder = ErrorSpecBuilder;

  double abs_err = 0.0;
  double rel_err = 0.0;
  // The acceptable amount of floating point values between the expected and
  // actual (also calling floating point distance).
  //
  // This is similar to absolute error, but the same distance_err can have
  // different floating point values as the exponent changes. In some way, it is
  // a hybrid of absolute and relative error, as it allows a fixed binary
  // difference (like abs_err), but that has a varied floating point value based
  // on the number (like rel_err).
  int64_t distance_err = 0;
  // If true, will consider -0 not near to +0 and vice versa.  Note that
  // +epsilon may still be considered close to -0, depending on the error
  // spec; this only covers the case when both `expected` and `actual` are
  // equal to 0.
  bool strict_signed_zeros = false;
  // If true, this will skip comparing the output of the test to the expected
  // value. This should be used only as a last resort, since it is effectively
  // turning off the test for a specific input value set.
  bool skip_comparison = false;
};

// Builder pattern to construct an ErrorSpec without a proliferation of
// constructors or requiring extensive argument name comments.
//
// This was created mostly to avoid using designated initializers since that is
// not compliant with all intended compilers (MSVC). This offers about the same
// functionality with slightly more keystrokes.
//
// You can use an lvalue or rvalue to call the setter functions, but you can
// only build (explicitly or implicitly) using an rvalue from std::move.
class ErrorSpecBuilder {
 public:
  ErrorSpecBuilder() : spec_() {}

  ErrorSpecBuilder& abs_err(double abs_err) & {
    spec_.abs_err = abs_err;
    return *this;
  }
  ErrorSpecBuilder&& abs_err(double abs_err) && {
    spec_.abs_err = abs_err;
    return std::move(*this);
  }

  ErrorSpecBuilder& rel_err(double rel_err) & {
    spec_.rel_err = rel_err;
    return *this;
  }
  ErrorSpecBuilder&& rel_err(double rel_err) && {
    spec_.rel_err = rel_err;
    return std::move(*this);
  }

  ErrorSpecBuilder& distance_err(int64_t distance_err) & {
    spec_.distance_err = distance_err;
    return *this;
  }
  ErrorSpecBuilder&& distance_err(int64_t distance_err) && {
    spec_.distance_err = distance_err;
    return std::move(*this);
  }

  ErrorSpecBuilder& strict_signed_zeros(bool strict_signed_zeros = true) & {
    spec_.strict_signed_zeros = strict_signed_zeros;
    return *this;
  }
  ErrorSpecBuilder&& strict_signed_zeros(bool strict_signed_zeros = true) && {
    spec_.strict_signed_zeros = strict_signed_zeros;
    return std::move(*this);
  }

  ErrorSpecBuilder& skip_comparison(bool skip_comparison = true) & {
    spec_.skip_comparison = skip_comparison;
    return *this;
  }
  ErrorSpecBuilder&& skip_comparison(bool skip_comparison = true) && {
    spec_.skip_comparison = skip_comparison;
    return std::move(*this);
  }

  ErrorSpec build() && { return spec_; }

  explicit operator ErrorSpec() && { return std::move(*this).build(); }

 private:
  ErrorSpec spec_;
};

}  // namespace exhaustive_op_test
}  // namespace xla

#endif  // XLA_TESTS_EXHAUSTIVE_ERROR_SPEC_H_
