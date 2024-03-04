/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_ANNOTATION_H_
#define XLA_SERVICE_GPU_RUNTIME_ANNOTATION_H_

#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "tsl/profiler/lib/nvtx_utils.h"

namespace xla::gpu {

// Prepared information for the top level NVTX/profiler range covering an
// HloModule
struct ModuleAnnotation {
  explicit ModuleAnnotation(std::string_view module_name);
  explicit ModuleAnnotation(const HloModule& mod);

  std::string_view longest_op_name_prefix() const { return longest_prefix; }
  explicit operator std::string_view() const { return title_str; }

 private:
  friend void RangePush(nvtxDomainHandle_t domain,
                        const ModuleAnnotation& annotation) {
    tsl::profiler::RangePush(domain, annotation.title);
  }

  std::string longest_prefix;
  std::string title_str;
  nvtxStringHandle_t title;
};

// Prepared information for a kernel/thunk/fusion/... within an HloModule
struct KernelAnnotation {
  KernelAnnotation(const ModuleAnnotation& module_annotation,
                   const HloInstruction& inst);

  explicit operator std::string_view() const { return title_str; }

 private:
  friend void RangePush(nvtxDomainHandle_t domain,
                        const KernelAnnotation& annotation) {
    tsl::profiler::RangePush(domain, annotation.title);
  }

  std::string title_str;
  nvtxStringHandle_t title;
};

// Parsed/prepared information for an HloModule that gets propagated to NVTX
// ranges/profilers/... at execution time.
struct ModuleAnnotations {
  explicit ModuleAnnotations(std::string_view module_name);
  explicit ModuleAnnotations(const HloModule&);

  ModuleAnnotation top_level;
  absl::flat_hash_map<std::string_view, KernelAnnotation> kernels;
};

//===----------------------------------------------------------------------===//
// Scoped RAII helper to set and restore thread local module annotations
//===----------------------------------------------------------------------===//

class ScopedModuleAnnotations {
 public:
  explicit ScopedModuleAnnotations(const ModuleAnnotations* annotations);
  ~ScopedModuleAnnotations();

 private:
  const ModuleAnnotations* restore_;
};

const ModuleAnnotations* GetCurrentModuleAnnotations();

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_ANNOTATION_H_
