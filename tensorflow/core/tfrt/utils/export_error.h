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
#ifndef TENSORFLOW_CORE_TFRT_UTILS_EXPORT_ERROR_H_
#define TENSORFLOW_CORE_TFRT_UTILS_EXPORT_ERROR_H_

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"

namespace tensorflow {
namespace tfrt_stub {

class ErrorExporter {
 public:
  virtual ~ErrorExporter() = default;

  virtual void MaybeExportError(absl::string_view message,
                                absl::string_view subcomponent) {}
};

class ErrorExporterRegistry {
 public:
  ErrorExporterRegistry()
      : error_exporter_(std::make_unique<ErrorExporter>()) {}

  void Register(std::unique_ptr<ErrorExporter> error_exporter) {
    error_exporter_ = std::move(error_exporter);
  }

  ErrorExporter &Get() const { return *error_exporter_; }

 private:
  std::unique_ptr<ErrorExporter> error_exporter_;
};

inline ErrorExporterRegistry &GetGlobalErrorExporterRegistry() {
  static auto *const registry = new ErrorExporterRegistry;
  return *registry;
}

// Builds ErrorLog proto for TFRT:{subcomponent} and sends ML stack error corp
// log to Sawmill via ULS (if enabled in `error_logging`).
inline void MaybeExportError(absl::string_view message,
                             absl::string_view subcomponent) {
  return GetGlobalErrorExporterRegistry().Get().MaybeExportError(message,
                                                                 subcomponent);
}

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_UTILS_EXPORT_ERROR_H_
