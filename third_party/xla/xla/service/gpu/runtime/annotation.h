/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_module.h"
#include "tsl/profiler/lib/nvtx_utils.h"

namespace xla::gpu {
// Prepared information for the top level NVTX/profiler range covering an
// HloModule
struct ModuleAnnotation {
  ModuleAnnotation(std::string module_name, int module_id);
  ModuleAnnotation(const HloModule& mod);
  std::string_view longest_op_name_prefix() const;
  nvtxStringHandle_t NvtxRegisteredTitle() const;
  std::string_view Title() const;

 private:
  std::string longest_prefix;
  std::string title_str;
  nvtxStringHandle_t title{};
};

// Prepared information for a kernel/thunk/fusion/... within an HloModule
struct KernelAnnotation {
  KernelAnnotation(const ModuleAnnotation& module_annotaion,
                   const HloInstruction& inst);
  nvtxStringHandle_t NvtxRegisteredTitle() const;
  std::string_view Title() const;

 private:
  std::string title_str;
  nvtxStringHandle_t title{};
};
// Parsed/prepared information for an HloModule that gets propagated to NVTX
// ranges/profilers/... at execution time.
struct ModuleAnnotations {
  ModuleAnnotations(const HloModule&);
  ModuleAnnotation top_level;
  absl::flat_hash_map<std::string_view, KernelAnnotation> kernels{};
};
}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_ANNOTATION_H_
