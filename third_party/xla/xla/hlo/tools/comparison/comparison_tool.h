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

#ifndef XLA_HLO_TOOLS_COMPARISON_COMPARISON_TOOL_H_
#define XLA_HLO_TOOLS_COMPARISON_COMPARISON_TOOL_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "re2/re2.h"
#include "xla/hlo/tools/comparison/comparison_options.pb.h"
#include "xla/hlo/tools/comparison/comparison_service.pb.h"
#include "xla/literal.h"
#include "xla/service/hlo.pb.h"
#include "xla/tools/debug_event.pb.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::numerics::comparison {

using ::xla::LogData;

// Abstract base class defining the public API for the HLO comparison tool.
// Users interact with an instance of this class (obtained via
// `ComparisonToolSingleton::Get` in `singleton.h`) to record information about
// HLO module executions and intermediate tensors for comparison purposes.
// Concrete implementations handle the specifics of either sending data (client)
// or receiving and processing data (server).
class ComparisonTool {
 public:
  explicit ComparisonTool(const ComparisonOptions& options,
                          tsl::thread::ThreadPool* async_queue = nullptr);
  virtual ~ComparisonTool() = default;

  // Registers the original HLO module proto before compiler optimization.
  // This should be called once per module being compared.
  //
  // Args:
  //   `module`: The HLO module proto to register.
  virtual absl::Status RegisterOriginalHloModule(const HloModuleProto& module);

  // Registers the start of a specific execution run for a given HLO module.
  // Associates a unique `run_id` with the `hlo_module_name`.
  //
  // Args:
  //   `logical_device_id`: The logical device id of the execution run.
  //   `run_id`: A unique identifier for this execution run.
  //   `hlo_module_name`: The name of the HLO module this run belongs to.
  virtual absl::Status RegisterRun(int32_t logical_device_id, uint64_t run_id,
                                   absl::string_view hlo_module_name) = 0;

  // Marks the completion of a specific execution run identified by `run_id`.
  // This triggers the `FinishHloModuleRun` call for the associated module
  // if this was the last active run for that module.
  //
  // Args:
  //   `logical_device_id`: The logical device id of the execution run.
  //   `run_id`: The unique identifier of the execution run to complete.
  //   `hlo_module_name`: The name of the HLO module this run belongs to.
  virtual absl::Status FinishRun(int32_t logical_device_id, uint64_t run_id,
                                 absl::string_view hlo_module_name) = 0;

  // Records an intermediate tensor value encountered during an execution run.
  // Users should call this method for each tensor value encountered during an
  // execution run.
  //
  // Args:
  //   `log_record`: Metadata associated with the tensor, such as its name and
  //      originating HLO instruction.
  //   `literal`: The actual data of the tensor. A pointer is used here
  //      because the literal can be big so we do not want to copy it. A shared
  //      pointer is used because it is created as a shared pointer in the
  //      logging handler and shared among all logging handlers.
  void RecordTensor(const LogData& log_record,
                    const std::shared_ptr<const xla::Literal>& literal);

 protected:
  // See `RegisterOriginalHloModule` for details.
  virtual absl::Status RegisterOriginalHloModuleImpl(
      const HloModuleProto& module) = 0;

  // Processes the summary of a recorded tensor. Implementations decide how to
  // handle this summary (e.g., send it via gRPC, store it for comparison, etc).
  //
  // Args:
  //   `hlo_module_name`: The name of the HLO module this tensor summary belongs
  //      to.
  //   `summary`: The TensorSummary proto to process.
  virtual absl::Status ProcessTensorSummary(absl::string_view hlo_module_name,
                                            const TensorSummary& summary) = 0;

  // Helper method to create a TensorSummary proto from the log record metadata
  // and the tensor literal data. May include hashing or other summarization.
  //
  // Args:
  //   `log_record`: Metadata associated with the tensor.
  //   `literal`: The actual data of the tensor.
  //
  // Returns:
  //   A TensorSummary proto populated with information from the inputs.
  TensorSummary CreateTensorSummary(const LogData& log_record,
                                    const xla::Literal& literal);

  // Static helper to check if the log record indicates an original tensor value
  // (as opposed to a potentially transformed one).
  //
  // Args:
  //   `log_record`: The log record to inspect.
  //
  // Returns:
  //   True if the log record's HLO output metadata has an original value,
  //   false otherwise.
  static bool HasOriginalValue(const LogData& log_record) {
    return log_record.hlo_output_metadata().has_original_value();
  }

  // Configuration options for the comparison process.
  const ComparisonOptions options_;
  const RE2 hlo_module_name_regex_;
  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, ModuleStats> module_stats_map_
      ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_set<std::string> registered_hlo_module_names_
      ABSL_GUARDED_BY(mutex_);

 private:
  // Queue for asynchronous processing of tensor summaries.
  std::unique_ptr<tsl::thread::ThreadPool> owned_async_queue_;
  tsl::thread::ThreadPool* async_queue_;
};
}  // namespace xla::numerics::comparison

#endif  // XLA_HLO_TOOLS_COMPARISON_COMPARISON_TOOL_H_
