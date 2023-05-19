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

#include "tensorflow/compiler/xla/python/pjrt_ifrt/xla_compiler.h"

#include <memory>

namespace xla {
namespace ifrt {

char XlaCompileOptions::ID = 0;
char XlaDeserializeOptions::ID = 0;

StatusOr<std::unique_ptr<XlaCompileOptions>> GetXlaCompileOptions(
    std::unique_ptr<CompileOptions> options) {
  if (!llvm::isa<XlaCompileOptions>(options.get())) {
    return xla::InvalidArgument("options must be XlaCompileOptions");
  }
  return std::unique_ptr<XlaCompileOptions>(
      static_cast<XlaCompileOptions*>(options.release()));
}

StatusOr<std::unique_ptr<XlaDeserializeOptions>> GetXlaDeserializeOptions(
    std::unique_ptr<DeserializeOptions> options) {
  if (!llvm::isa<XlaDeserializeOptions>(options.get())) {
    return xla::InvalidArgument("options must be XlaDeserializeOptions");
  }
  return std::unique_ptr<XlaDeserializeOptions>(
      static_cast<XlaDeserializeOptions*>(options.release()));
}

}  // namespace ifrt
}  // namespace xla
