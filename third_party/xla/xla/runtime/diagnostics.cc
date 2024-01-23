/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/runtime/diagnostics.h"

#include <utility>

namespace xla {
namespace runtime {

const DiagnosticEngine* DiagnosticEngine::DefaultDiagnosticEngine() {
  static auto* diagnostic_engine = new DiagnosticEngine();
  return diagnostic_engine;
}

void InFlightDiagnostic::Report() {
  if (IsInFlight()) {
    engine_->Emit(std::move(*diagnostic_));
    engine_ = nullptr;
  }
  diagnostic_.reset();
}

void InFlightDiagnostic::Abandon() { engine_ = nullptr; }

}  // namespace runtime
}  // namespace xla
