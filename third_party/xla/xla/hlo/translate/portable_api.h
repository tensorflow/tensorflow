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

#ifndef XLA_HLO_TRANSLATE_PORTABLE_API_H_
#define XLA_HLO_TRANSLATE_PORTABLE_API_H_

#include <string>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"

// This file is a portable version of the HLO API.
// Is offers a string API passthrough for MLIR datatypes and is intended
// to offer a safe means of using StableHLO opaquely in non-MLIR code.

namespace xla {

absl::StatusOr<std::string> ConvertHloToStablehlo(
    xla::HloModule const& hlo_module, bool emit_bytecode = false);

}

#endif  // XLA_HLO_TRANSLATE_PORTABLE_API_H_
