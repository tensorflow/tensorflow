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

#include "xla/backends/cpu/codegen/execution_engine.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "xla/backends/cpu/codegen/contiguous_section_memory_manager.h"
#include "xla/service/cpu/orc_jit_memory_mapper.h"

namespace xla::cpu {

static std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer>
CreateObjectLinkingLayer(llvm::orc::ExecutionSession& execution_session) {
  return std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
      execution_session, [](const llvm::MemoryBuffer&) {
        return std::make_unique<ContiguousSectionMemoryManager>(
            orc_jit_memory_mapper::GetInstance());
      });
}

ExecutionEngine::ExecutionEngine(
    std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
    const llvm::DataLayout& data_layout,
    DefinitionGenerator definition_generator)
    : execution_session_(std::move(execution_session)),
      object_layer_(CreateObjectLinkingLayer(*execution_session_)),
      data_layout_(data_layout),
      definition_generator_(definition_generator) {
  execution_session_->setErrorReporter([](llvm::Error err) {
    LOG(ERROR) << "LLVM compilation error: " << llvm::toString(std::move(err));
  });
}

void ExecutionEngine::RegisterJITEventListeners() {
  gdb_listener_ = llvm::JITEventListener::createGDBRegistrationListener();
  perf_listener_ = llvm::JITEventListener::createPerfJITEventListener();

  // Register GDB and perf event listeners with the object linking layer.
  if (gdb_listener_) object_layer_->registerJITEventListener(*gdb_listener_);
  if (perf_listener_) object_layer_->registerJITEventListener(*perf_listener_);
}

void ExecutionEngine::AllocateDylibs(size_t num_dylibs) {
  dylibs_.resize(std::max<size_t>(1, num_dylibs));
  for (size_t i = 0; i < dylibs_.size(); ++i) {
    dylibs_[i] = &execution_session()->createBareJITDylib(
        absl::StrCat("<xla_jit_dylib_", i, ">"));

    if (definition_generator_) {
      dylibs_[i]->addGenerator(definition_generator_(data_layout_));
    }
  }
}

ExecutionEngine::~ExecutionEngine() {
  if (execution_session_) {
    if (auto err = execution_session_->endSession()) {
      execution_session_->reportError(std::move(err));
    }
  }

  // Unregister GDB and perf event listeners with the object linking layer.
  if (gdb_listener_) object_layer_->unregisterJITEventListener(*gdb_listener_);
  if (perf_listener_)
    object_layer_->unregisterJITEventListener(*perf_listener_);
}

void ExecutionEngine::SetObjectLayerFlags() {
  // TODO(basioli) can we invoke this from the constructor?
  object_layer_->setOverrideObjectFlagsWithResponsibilityFlags(true);
  object_layer_->setAutoClaimResponsibilityForObjectSymbols(true);
}

}  // namespace xla::cpu
