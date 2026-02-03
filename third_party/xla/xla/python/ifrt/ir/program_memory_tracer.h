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

#ifndef XLA_PYTHON_IFRT_IR_PROGRAM_MEMORY_TRACER_H_
#define XLA_PYTHON_IFRT_IR_PROGRAM_MEMORY_TRACER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/ir/compiled_ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/program_memory_trace.pb.h"
#include "xla/service/hlo.pb.h"

namespace xla {
namespace ifrt {

// Stats about memory usage of an IFRT IR program.
struct IfrtIrProgramMemoryStats {
  // Device default memory (e.g., HBM for GPU/TPU) usage stats.
  int64_t argument_size_in_bytes = 0;
  int64_t output_size_in_bytes = 0;
  // Mappings between device id and peak bytes used and minimum bytes available.
  absl::flat_hash_map<int64_t, int64_t> device_to_peak_bytes_used;
  absl::flat_hash_map<int64_t, int64_t> device_to_min_memory_bytes_available;

  // Host memory usage stats.
  int64_t host_argument_size_in_bytes = 0;
  int64_t host_output_size_in_bytes = 0;

  template <class Sink>
  friend void AbslStringify(Sink& sink, const IfrtIrProgramMemoryStats& stats) {
    absl::Cord device_to_peak_bytes_used_str;
    absl::Cord device_to_min_memory_bytes_available_str;
    for (const auto& [device_id, peak_bytes_used] :
         stats.device_to_peak_bytes_used) {
      device_to_peak_bytes_used_str.Append(
          absl::StrCat("devices", device_id, "=", peak_bytes_used, "; "));
    }
    for (const auto& [device_id, min_memory_bytes_available] :
         stats.device_to_min_memory_bytes_available) {
      device_to_min_memory_bytes_available_str.Append(absl::StrCat(
          "device", device_id, "=", min_memory_bytes_available, "; "));
    }
    absl::Format(&sink,
                 "IfrtIrProgramMemoryStats(argument_size_in_bytes=%d; "
                 "output_size_in_bytes=%d; host_argument_size_in_bytes=%d; "
                 "host_output_size_in_bytes=%d; device_to_peak_bytes_used=%s; "
                 "device_to_min_memory_bytes_available=%s)",
                 stats.argument_size_in_bytes, stats.output_size_in_bytes,
                 stats.host_argument_size_in_bytes,
                 stats.host_output_size_in_bytes, device_to_peak_bytes_used_str,
                 device_to_min_memory_bytes_available_str);
  }
};

// Generates a xprof memory profile of a MPMD program. It does not require
// running the program, and can be generated as part of cross compilation.
class ProgramMemoryTracer {
 public:
  static absl::StatusOr<std::unique_ptr<ProgramMemoryTracer>> Create(
      std::shared_ptr<CompiledIfrtIrProgram> program, xla::ifrt::Client* client,
      xla::ifrt::DeviceListRef devices);

  // Gets the predicted memory states of the given IFRT IR program.
  absl::StatusOr<IfrtIrProgramMemoryStats> GetMemoryStats();

  // Returns a URL to a generated xprof of the predicted memory profile.
  // This profile differs from profiles captured at runtime in the following
  // ways:
  // 1. array allocations are added to "reserved" memory since this appears
  // at the bottom of the xprof UI, and so it makes it clear what memory is
  // being retained by live arrays.
  // 2. executable allocations are added to "allocated" memory since this
  // appears on top of the "reserved" memory.
  absl::StatusOr<std::string> GetXprofUrl();

 private:
  ProgramMemoryTracer(std::shared_ptr<CompiledIfrtIrProgram> program,
                      xla::ifrt::Client* client,
                      xla::ifrt::DeviceListRef devices, std::string dump_dir,
                      mlir::Liveness liveness)
      : program_(std::move(program)),
        client_(client),
        devices_(std::move(devices)),
        dump_dir_(std::move(dump_dir)),
        liveness_(std::move(liveness)) {}

  absl::Status GenerateEvents();
  absl::Status GenerateEvents(xla::ifrt::CallLoadedExecutableOp call_loaded_op);
  absl::Status GenerateEvents(xla::ifrt::RemapArraysOp remap_op);
  absl::Status GenerateEvents(xla::ifrt::CopyArraysOp copy_arrays_op);
  absl::Status GenerateEvents(mlir::func::ReturnOp return_op);

  // Gets the per-device size of the array in bytes.
  // This function assumes that the size per device does not differ. This is a
  // fine assumption for now since all IFRT IR shardings have this limitation.
  absl::StatusOr<int64_t> PerDeviceByteSize(xla::ifrt::IfrtArrayType array);
  absl::StatusOr<int64_t> PerHostByteSize(xla::ifrt::IfrtArrayType array);

  absl::Status AllocateArray(mlir::Value value, absl::string_view created_by);
  absl::Status FreeArray(mlir::Value value);

  mlir::SymbolTableCollection symbol_table_;
  // The program for which the memory trace is being generated.
  std::shared_ptr<CompiledIfrtIrProgram> program_;
  xla::ifrt::Client* client_;
  xla::ifrt::DeviceListRef devices_;

  // Directory where to dump the memory trace for the traced devices.
  std::string dump_dir_;

  // Cached liveness analysis of the IFRT IR program.
  mlir::Liveness liveness_;

  // Proto message to which the memory trace events will be added to. Lazily,
  // populate when `GetMemoryStats` or `GetXprofUrl` is first called.
  std::optional<ProgramMemoryTrace> memory_trace_;

  // Mapping between array and unique id and name given at array allocation
  // time. This mapping is used to associate a free event with an allocation
  // event.
  llvm::DenseMap<mlir::Value, std::pair<int64_t, std::string>>
      array_to_allocation_id_and_name_;

  // Monotonically increating counter for generating unique allocation ids.
  int64_t current_allocation_id_ = 0;

  // Set of logical device ids for which to generate a trace. The set does not
  // contain all the device ids in order to avoid generating large traces, but
  // it includes at least a device from each execution or transfer. It is
  // constructed by picking the device with the smallest logical id from each
  // `ifrt.CallLoadedExecutableOp`, and the smallest logical device ids
  // from `ifrt.CopyArraysOp` inputs, respectively outputs.
  absl::flat_hash_set<int> device_ids_to_trace_;

  // Mapping between logical device id and topology description. This mapping is
  // used to avoid getting the devices and constructing a topology description
  // for each array.
  absl::flat_hash_map<int, std::shared_ptr<const xla::PjRtTopologyDescription>>
      logical_device_id_to_topology_desc_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_PROGRAM_MEMORY_TRACER_H_
