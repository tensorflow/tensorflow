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

#ifndef XLA_TESTS_HLO_PJRT_INTERPRETER_REFERENCE_MIXIN_H_
#define XLA_TESTS_HLO_PJRT_INTERPRETER_REFERENCE_MIXIN_H_

#include "absl/base/macros.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"

namespace xla {

template <typename T>
using HloPjRtInterpreterReferenceMixin
    [[deprecated("HloPjRtInterpreterReferenceMixin is a deprecated alias for "
                 "HloInterpreterReferenceMixin.")]] ABSL_REFACTOR_INLINE =
        HloInterpreterReferenceMixin<T>;

}  // namespace xla

#endif  // XLA_TESTS_HLO_PJRT_INTERPRETER_REFERENCE_MIXIN_H_
