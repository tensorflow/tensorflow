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

#ifndef XLA_SERVICE_GPU_EXPORT_HLO_H_
#define XLA_SERVICE_GPU_EXPORT_HLO_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/stream_executor/device_description.pb.h"

// Functionality to enable HLO uploads from XLA to HLO repositories. Unoptimized
// HLO means the HLO given to XLA, while optimized HLO refers to HLO that has
// been successfully compiled. Errors in upload should not block compilation.

namespace xla::gpu {

// Uploads HLO to a repository. The only non-dummy implementation is
// Google-internal as of 2023-10.
class SymbolUploader {
 public:
  virtual ~SymbolUploader() = default;

  // Returns a string identifying the uploaded HLO, or empty if the upload did
  // not complete. We use optional rather than absl::StatusOr because an upload
  // error is not a compiler error.
  virtual std::optional<std::string> MaybeUploadUnoptimizedHloModule(
      HloModule* module,
      const stream_executor::GpuTargetConfigProto& gpu_target_config) = 0;

  virtual std::optional<std::string> MaybeUploadOptimizedHloModule(
      const HloProto& hlo_proto) = 0;

  virtual void MaybeUploadSymbolMapping(
      absl::string_view unoptimized_fingerprint,
      absl::string_view optimized_fingerprint) = 0;

  virtual void WaitForUploads() = 0;
};

// Registers a single process-wide XSymbolUploader to use. The registry is used
// to provide a hook for internal infrastructure and ensure that only one
// background thread is uploading.
class SymbolUploaderRegistry {
 public:
  SymbolUploaderRegistry() : xsymbol_uploader_(nullptr) {}
  std::unique_ptr<SymbolUploader> Register(
      std::unique_ptr<SymbolUploader> xsymbol_uploader) {
    return std::exchange(xsymbol_uploader_, std::move(xsymbol_uploader));
  }

  SymbolUploader* uploader() const { return xsymbol_uploader_.get(); }

 private:
  std::unique_ptr<SymbolUploader> xsymbol_uploader_;
};

inline SymbolUploaderRegistry& GetGlobalSymbolUploaderRegistry() {
  static auto* const registry = new SymbolUploaderRegistry;
  return *registry;
}

// The actual entry points from XLA start here.
inline void MaybeUploadUnoptimizedGpuSymbols(
    HloModule* module,
    const stream_executor::GpuTargetConfigProto& gpu_target_config) {
  SymbolUploader* uploader = GetGlobalSymbolUploaderRegistry().uploader();
  if (uploader == nullptr) {
    return;
  }
  std::optional<std::string> unoptimized_fingerprint =
      uploader->MaybeUploadUnoptimizedHloModule(module, gpu_target_config);
  if (!unoptimized_fingerprint.has_value()) {
    return;
  }
  module->set_unoptimized_fingerprint(*unoptimized_fingerprint);
}

inline void MaybeUploadOptimizedGpuSymbols(
    const HloModule& module, const BufferAssignmentProto& buffer_assignment) {
  SymbolUploader* uploader = GetGlobalSymbolUploaderRegistry().uploader();
  if (uploader == nullptr) {
    return;
  }
  HloProto hlo_proto = MakeHloProto(module);
  *hlo_proto.mutable_buffer_assignment() = buffer_assignment;
  std::optional<std::string> optimized_fingerprint =
      uploader->MaybeUploadOptimizedHloModule(hlo_proto);
  if (module.unoptimized_fingerprint().empty() ||
      !optimized_fingerprint.has_value()) {
    return;
  }
  uploader->MaybeUploadSymbolMapping(module.unoptimized_fingerprint(),
                                     *optimized_fingerprint);
}

inline void MaybeWaitForUploads() {
  if (SymbolUploader* uploader = GetGlobalSymbolUploaderRegistry().uploader();
      uploader != nullptr) {
    uploader->WaitForUploads();
  }
}

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_EXPORT_HLO_H_
