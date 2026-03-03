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

#include "xla/python/ifrt/ir/program_memory_tracer.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/ir/compiled_ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"

namespace xla {
namespace ifrt {

absl::StatusOr<std::unique_ptr<ProgramMemoryTracer>>
ProgramMemoryTracer::Create(std::shared_ptr<CompiledIfrtIrProgram> program,
                            Client* client, DeviceListRef devices) {
  return absl::UnimplementedError("Create is not implemented.");
}

absl::StatusOr<int64_t> ProgramMemoryTracer::PerDeviceByteSize(
    IfrtArrayType array) {
  return absl::UnimplementedError("PerDeviceByteSize is not implemented.");
}

absl::StatusOr<int64_t> ProgramMemoryTracer::PerHostByteSize(
    IfrtArrayType array) {
  return absl::UnimplementedError("PerHostByteSize is not implemented.");
}

absl::Status ProgramMemoryTracer::AllocateArray(mlir::Value value,
                                                absl::string_view created_by) {
  return absl::UnimplementedError("AllocateArray is not implemented.");
}

absl::Status ProgramMemoryTracer::FreeArray(mlir::Value value) {
  return absl::UnimplementedError("FreeArray is not implemented.");
}

absl::Status ProgramMemoryTracer::GenerateEvents() {
  return absl::UnimplementedError("GenerateEvents is not implemented.");
}

absl::StatusOr<IfrtIrProgramMemoryStats> ProgramMemoryTracer::GetMemoryStats() {
  return absl::UnimplementedError("GetMemoryStats is not implemented.");
}

absl::StatusOr<std::string> ProgramMemoryTracer::GetXprofUrl() {
  return absl::UnimplementedError("GetXprofUrl is not implemented.");
}

absl::Status ProgramMemoryTracer::GenerateEvents(
    CallLoadedExecutableOp call_loaded_op) {
  return absl::UnimplementedError("GenerateEvents is not implemented.");
}

absl::Status ProgramMemoryTracer::GenerateEvents(CopyArraysOp copy_arrays_op) {
  return absl::UnimplementedError("GenerateEvents is not implemented.");
}

absl::Status ProgramMemoryTracer::GenerateEvents(RemapArraysOp remap_op) {
  return absl::UnimplementedError("GenerateEvents is not implemented.");
}

absl::Status ProgramMemoryTracer::GenerateEvents(
    mlir::func::ReturnOp return_op) {
  return absl::UnimplementedError("GenerateEvents is not implemented.");
}

}  // namespace ifrt
}  // namespace xla
