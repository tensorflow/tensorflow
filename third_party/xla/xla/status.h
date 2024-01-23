/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_STATUS_H_
#define XLA_STATUS_H_

#include "absl/log/check.h"  // IWYU pragma: export
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xla {
// NOLINTBEGIN(misc-unused-using-decls)
using absl::OkStatus;
using absl::Status;
using absl::StatusOr;
// NOLINTEND(misc-unused-using-decls)
}  // namespace xla

#endif  // XLA_STATUS_H_
