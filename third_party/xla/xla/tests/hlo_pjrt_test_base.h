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

#ifndef XLA_TESTS_HLO_PJRT_TEST_BASE_H_
#define XLA_TESTS_HLO_PJRT_TEST_BASE_H_

#include "absl/base/macros.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {

using HloPjRtTestBaseOptions
    [[deprecated("HloPjRtTestBaseOptions is a deprecated alias for "
                 "HloTestBaseOptions.")]] ABSL_REFACTOR_INLINE =
        HloTestBaseOptions;

using HloPjRtTestBase
    [[deprecated("HloPjRtTestBase is a deprecated alias for "
                 "HloTestBase.")]] ABSL_REFACTOR_INLINE = HloTestBase;

}  // namespace xla

#endif  // XLA_TESTS_HLO_PJRT_TEST_BASE_H_
