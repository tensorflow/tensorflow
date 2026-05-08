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

#include "xla/service/multi_module_driver.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/hlo_module_splitter.h"
#include "xla/hlo/transforms/hlo_module_stitcher.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/blocking_counter.h"

namespace xla {

std::atomic<int> MultiModuleDriver::compile_count_ = 0;
int MultiModuleDriver::GetCompileCount() { return compile_count_.load(); }
void MultiModuleDriver::ResetCompileCount() { compile_count_ = 0; }

bool MultiModuleDriver::ShouldProcess(const HloModule& module) {
  for (const HloComputation* computation : module.MakeComputationPostOrder()) {
    for (const auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCall) {
        auto it =
            instruction->frontend_attributes().map().find("compilation_unit");
        if (it != instruction->frontend_attributes().map().end()) {
          return true;
        }
      }
    }
  }
  return false;
}

absl::StatusOr<std::unique_ptr<HloModule>> MultiModuleDriver::Compile(
    std::unique_ptr<HloModule> module,
    const std::vector<se::StreamExecutor*>& /*stream_execs*/,
    const Compiler::CompileOptions& options) const {
  compile_count_++;
  xla::HloModuleSplitter splitter;
  ASSIGN_OR_RETURN(bool changed, splitter.Run(module.get()));
  if (!changed) {
    return module;
  }

  std::vector<std::unique_ptr<HloModule>> submodules =
      std::move(splitter.submodules());

  tsl::thread::ThreadPool* thread_pool =
      options.thread_pool ? options.thread_pool : thread_pool_;

  std::vector<std::unique_ptr<HloModule>> all_modules;
  all_modules.reserve(submodules.size() + 1);
  all_modules.push_back(std::move(module));
  for (auto& submodule : submodules) {
    all_modules.push_back(std::move(submodule));
  }

  std::vector<absl::StatusOr<std::unique_ptr<HloModule>>> results(
      all_modules.size());

  if (thread_pool == nullptr) {
    // Sequential compilation.
    for (size_t i = 0; i < all_modules.size(); ++i) {
      results[i] = compile_fn_(std::move(all_modules[i]), options);
    }
  } else {
    // Parallel compilation.
    tsl::BlockingCounter counter(all_modules.size());
    for (size_t i = 0; i < all_modules.size(); ++i) {
      thread_pool->Schedule(
          [this, &all_modules, &options, &results, &counter, i]() {
            results[i] = compile_fn_(std::move(all_modules[i]), options);
            counter.DecrementCount();
          });
    }
    counter.Wait();
  }

  ASSIGN_OR_RETURN(module, std::move(results[0]));
  std::vector<std::unique_ptr<HloModule>> optimized_submodules;
  optimized_submodules.reserve(results.size() - 1);
  absl::flat_hash_map<std::string, const HloModule*> optimized_modules_map;
  optimized_modules_map.reserve(results.size() - 1);
  for (size_t i = 1; i < results.size(); ++i) {
    ASSIGN_OR_RETURN(auto opt_submod, std::move(results[i]));
    optimized_submodules.push_back(std::move(opt_submod));
    optimized_modules_map[optimized_submodules.back()->name()] =
        optimized_submodules.back().get();
  }

  xla::HloModuleStitcher stitcher(optimized_modules_map);
  RETURN_IF_ERROR(stitcher.Run(module.get()).status());

  return module;
}

}  // namespace xla
