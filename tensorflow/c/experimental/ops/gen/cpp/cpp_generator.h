/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_CPP_GENERATOR_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_CPP_GENERATOR_H_

#include "tensorflow/c/experimental/ops/gen/common/controller.h"
#include "tensorflow/c/experimental/ops/gen/common/path_config.h"
#include "tensorflow/c/experimental/ops/gen/common/source_code.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/cpp_config.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer_context.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {

class CppGenerator {
 public:
  explicit CppGenerator(cpp::CppConfig cpp_config, PathConfig path_config);
  SourceCode HeaderFileContents() const;
  SourceCode SourceFileContents() const;
  string HeaderFileName() const;
  string SourceFileName() const;
  void WriteHeaderFile() const;
  void WriteSourceFile() const;

 private:
  SourceCode GenerateOneFile(cpp::RendererContext::Mode mode) const;

  Controller controller_;
  cpp::CppConfig cpp_config_;
  PathConfig path_config_;
};

}  // namespace generator
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_CPP_GENERATOR_H_
