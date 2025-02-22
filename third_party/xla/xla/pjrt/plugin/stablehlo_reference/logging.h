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

#ifndef XLA_PJRT_PLUGIN_STABLEHLO_REFERENCE_LOGGING_H_
#define XLA_PJRT_PLUGIN_STABLEHLO_REFERENCE_LOGGING_H_

// This file has some joint logging to allow LOG and VLOG to play well with
// MLIR data structures

#include "absl/log/log.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#define LOG_UNIMPLEMENTED(name) \
  LOG(ERROR) << "MlirPjrtBuffer::" #name " is not implemented"

#define TRACE_ME LOG(INFO) << __func__;

#define TRACE_ME_MEMBER LOG(INFO) << __func__ << "(" << (void*)this << ")\n";

namespace mlir::stablehlo {
std::string ToString(mlir::Attribute attr);
std::string ToString(llvm::SmallVector<mlir::DenseElementsAttr> attrs);
std::string ToString(Operation* op);

// Looks for `PJRT_LOG_LEVEL = INFO|WARNING|ERROR` in env variables.
void SetupLogLevelFromEnv();
}  // namespace mlir::stablehlo

#endif  // XLA_PJRT_PLUGIN_STABLEHLO_REFERENCE_LOGGING_H_
