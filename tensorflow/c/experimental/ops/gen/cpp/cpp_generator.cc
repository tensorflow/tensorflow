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
#include "tensorflow/c/experimental/ops/gen/cpp/cpp_generator.h"

#include <vector>

#include "tensorflow/c/experimental/ops/gen/common/path_config.h"
#include "tensorflow/c/experimental/ops/gen/common/source_code.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/cpp_config.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/cpp_file_renderer.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer_context.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/op_view.h"
#include "tensorflow/c/experimental/ops/gen/model/op_spec.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {

CppGenerator::CppGenerator(cpp::CppConfig cpp_config, PathConfig path_config)
    : controller_(path_config),
      cpp_config_(cpp_config),
      path_config_(path_config) {}

SourceCode CppGenerator::GenerateOneFile(
    cpp::RendererContext::Mode mode) const {
  SourceCode generated_code;
  const std::vector<OpSpec> ops(controller_.GetModelOps());
  std::vector<cpp::OpView> views(ops.begin(), ops.end());
  cpp::RendererContext context{mode, generated_code, cpp_config_, path_config_};
  cpp::CppFileRenderer(context, views).Render();
  return generated_code;
}

SourceCode CppGenerator::HeaderFileContents() const {
  return GenerateOneFile(cpp::RendererContext::kHeader);
}

SourceCode CppGenerator::SourceFileContents() const {
  return GenerateOneFile(cpp::RendererContext::kSource);
}

string CppGenerator::HeaderFileName() const {
  return io::JoinPath(path_config_.output_path, cpp_config_.unit + "_ops.h");
}

string CppGenerator::SourceFileName() const {
  return io::JoinPath(path_config_.output_path, cpp_config_.unit + "_ops.cc");
}

void CppGenerator::WriteHeaderFile() const {
  controller_.WriteFile(HeaderFileName(), HeaderFileContents());
}

void CppGenerator::WriteSourceFile() const {
  controller_.WriteFile(SourceFileName(), SourceFileContents());
}

}  // namespace generator
}  // namespace tensorflow
