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

#ifndef XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_UNARY_TEST_DEFINITIONS_H_
#define XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_UNARY_TEST_DEFINITIONS_H_

#include <array>    // IWYU pragma: keep, exhaustive_unary_test_definitions.inc
#include <cstdint>  // IWYU pragma: keep, exhaustive_unary_test_definitions.inc
#include <ios>      // IWYU pragma: keep, exhaustive_unary_test_definitions.inc
#include <utility>  // IWYU pragma: keep, exhaustive_unary_test_definitions.inc

#include "absl/log/check.h"  // IWYU pragma: keep, exhaustive_unary_test_definitions.inc
#include "absl/log/log.h"  // IWYU pragma: keep, exhaustive_unary_test_definitions.inc
#include "absl/types/span.h"  // IWYU pragma: keep, exhaustive_unary_test_definitions.inc
#include "xla/literal.h"  // IWYU pragma: keep, exhaustive_unary_test_definitions.inc
#include "xla/tests/exhaustive/exhaustive_op_test.h"  // IWYU pragma: keep, exhaustive_unary_test_definitions.inc
#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"  // IWYU pragma: keep, exhaustive_unary_test_definitions.inc
#include "tsl/platform/test.h"  // IWYU pragma: keep, exhaustive_unary_test_definitions.inc

namespace xla {
namespace exhaustive_op_test {

#include "xla/tests/exhaustive/exhaustive_unary_test_definitions.inc"

}  // namespace exhaustive_op_test
}  // namespace xla

#endif  // XLA_TESTS_EXHAUSTIVE_EXHAUSTIVE_UNARY_TEST_DEFINITIONS_H_
