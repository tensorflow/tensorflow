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
#ifndef TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_EXPORT_MLIR_H_
#define TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_EXPORT_MLIR_H_

#include <memory>
#include <string>
#include <utility>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace tensorflow {
namespace tfrt_stub {

class XsymbolUploader {
 public:
  virtual ~XsymbolUploader() = default;

  virtual std::string MaybeUploadMlirToXsymbol(mlir::ModuleOp module) {
    return "";
  }
};

class XsymbolUploaderRegistry {
 public:
  XsymbolUploaderRegistry()
      : xsymbol_uploader_(std::make_unique<XsymbolUploader>()) {}

  void Register(std::unique_ptr<XsymbolUploader> xsymbol_uploader) {
    xsymbol_uploader_ = std::move(xsymbol_uploader);
  }

  XsymbolUploader &Get() const { return *xsymbol_uploader_; }

 private:
  std::unique_ptr<XsymbolUploader> xsymbol_uploader_;
};

inline XsymbolUploaderRegistry &GetGlobalXsymbolUploaderRegistry() {
  static auto *const registry = new XsymbolUploaderRegistry;
  return *registry;
}

inline std::string MaybeUploadMlirToXsymbol(mlir::ModuleOp module) {
  return GetGlobalXsymbolUploaderRegistry().Get().MaybeUploadMlirToXsymbol(
      module);
}

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_EXPORT_MLIR_H_
