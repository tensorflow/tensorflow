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

#ifndef XLA_SERVICE_CPU_EXPORT_HLO_H_
#define XLA_SERVICE_CPU_EXPORT_HLO_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/cpu/executable.pb.h"

namespace xla {
namespace cpu {

class SymbolUploader {
 public:
  virtual ~SymbolUploader() = default;

  virtual std::optional<std::string> MaybeUploadUnoptimizedHloModule(
      HloModule* module,
      const TargetMachineOptionsProto& target_machine_options) = 0;

  virtual std::optional<std::string> MaybeUploadOptimizedHloModule(
      HloModule* module,
      const TargetMachineOptionsProto& target_machine_options) = 0;

  virtual void MaybeUploadSymbolMapping(
      absl::string_view unoptimized_fingerprint,
      absl::string_view optimized_fingerprint) = 0;

  virtual void WaitForUploads() = 0;
};

class SymbolUploaderRegistry {
 public:
  SymbolUploaderRegistry() : xsymbol_uploader_(nullptr) {}
  void Register(std::unique_ptr<SymbolUploader> xsymbol_uploader) {
    xsymbol_uploader_ = std::move(xsymbol_uploader);
  }

  SymbolUploader* uploader() const { return xsymbol_uploader_.get(); }

 private:
  std::unique_ptr<SymbolUploader> xsymbol_uploader_;
};

inline SymbolUploaderRegistry& GetGlobalSymbolUploaderRegistry() {
  static auto* const registry = new SymbolUploaderRegistry;
  return *registry;
}

inline std::optional<std::string> MaybeUploadUnoptimizedCpuSymbols(
    HloModule* module,
    const TargetMachineOptionsProto& target_machine_options) {
  if (SymbolUploader* uploader = GetGlobalSymbolUploaderRegistry().uploader();
      uploader != nullptr) {
    return uploader->MaybeUploadUnoptimizedHloModule(module,
                                                     target_machine_options);
  }

  return std::nullopt;
}

inline std::optional<std::string> MaybeUploadOptimizedCpuSymbols(
    HloModule* module,
    const TargetMachineOptionsProto& target_machine_options) {
  if (SymbolUploader* uploader = GetGlobalSymbolUploaderRegistry().uploader();
      uploader != nullptr) {
    return uploader->MaybeUploadOptimizedHloModule(module,
                                                   target_machine_options);
  }

  return std::nullopt;
}

inline void MaybeUploadCpuSymbolMapping(
    absl::string_view unoptimized_fingerprint,
    absl::string_view optimized_fingerprint) {
  if (SymbolUploader* uploader = GetGlobalSymbolUploaderRegistry().uploader();
      uploader != nullptr) {
    return uploader->MaybeUploadSymbolMapping(unoptimized_fingerprint,
                                              optimized_fingerprint);
  }
}

inline void MaybeWaitForUploads() {
  if (SymbolUploader* uploader = GetGlobalSymbolUploaderRegistry().uploader();
      uploader != nullptr) {
    uploader->WaitForUploads();
  }
}

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_EXPORT_HLO_H_
