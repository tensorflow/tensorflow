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
#include "tensorflow/c/experimental/ops/gen/cpp/cpp_controller.h"

#include "tensorflow/c/experimental/ops/gen/cpp/renderers/cpp_renderer.h"
#include "tensorflow/core/lib/io/path.h"

namespace tensorflow {
namespace generator {

CppController::CppController(cpp::CppConfig cpp_config,
                             PathConfig controller_config)
    : Controller(controller_config), cpp_config_(cpp_config) {}

SourceCode CppController::GenerateOneFile(cpp::RendererContext::Mode mode) {
  SourceCode generated_code;
  std::vector<cpp::OpView> op_views(operators_.begin(), operators_.end());
  cpp::RendererContext context{mode, generated_code, cpp_config_,
                               controller_config_};
  cpp::CppRenderer(context, op_views).Render();
  return generated_code;
}
SourceCode CppController::HeaderFileContents() {
  return GenerateOneFile(cpp::RendererContext::kHeader);
}
SourceCode CppController::SourceFileContents() {
  return GenerateOneFile(cpp::RendererContext::kSource);
}
string CppController::HeaderFileName() {
  return io::JoinPath(controller_config_.output_path,
                      cpp_config_.unit + "_ops.h");
}
string CppController::SourceFileName() {
  return io::JoinPath(controller_config_.output_path,
                      cpp_config_.unit + "_ops.cc");
}

}  // namespace generator
}  // namespace tensorflow
