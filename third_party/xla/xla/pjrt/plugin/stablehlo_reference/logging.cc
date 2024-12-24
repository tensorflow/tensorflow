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

#include "xla/pjrt/plugin/stablehlo_reference/logging.h"

#include <cstdlib>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

namespace mlir::stablehlo {

std::string ToString(mlir::Attribute attr) {
  std::string out;
  llvm::raw_string_ostream os(out);
  attr.print(os);
  return out;
}

std::string ToString(SmallVector<DenseElementsAttr> attrs) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "[";
  bool first = true;
  for (auto attr : attrs) {
    if (!first) os << ", ";
    first = false;
    attr.print(os);
  }
  os << "]";
  return out;
}

std::string ToString(Operation* op) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << *op;
  return out;
}

void SetupLogLevelFromEnv() {
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
  const char* log_env = std::getenv("PJRT_LOG_LEVEL");
  if (!log_env) return;
  if (strcmp(log_env, "INFO") == 0) {
    absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
  } else if (strcmp(log_env, "WARNING") == 0) {
    absl::SetMinLogLevel(absl::LogSeverityAtLeast::kWarning);
  } else if (strcmp(log_env, "ERROR") == 0) {
    absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
  } else {
    LOG(ERROR) << "Invalid PJRT_LOG_LEVEL: " << log_env;
  }
}

}  // namespace mlir::stablehlo
