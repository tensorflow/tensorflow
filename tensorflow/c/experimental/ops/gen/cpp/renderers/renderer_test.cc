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
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer.h"

#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer_context.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace generator {
namespace cpp {
namespace {

TEST(Renderer, typical_usage) {
  class TestRenderer : Renderer {
   public:
    explicit TestRenderer(SourceCode& code)
        : Renderer(
              {RendererContext::kSource, code, CppConfig(), PathConfig()}) {}

    void Render() {
      CommentLine("File level comment.");
      CodeLine("#include \"header.h\"");
      BlankLine();
      BlockOpen("void TestFunction()");
      {
        Statement("int i = 1");
        BlankLine();
        BlockOpen("while (i == 1)");
        {
          CommentLine("Do nothing, really....");
          CodeLine("#if 0");
          Statement("call()");
          CodeLine("#endif");
          BlockClose();
        }
        BlockClose("  // comment ending TestFunction");
      }
    }
  };

  SourceCode code;
  TestRenderer(code).Render();

  string expected = R"(// File level comment.
#include "header.h"

void TestFunction() {
   int i = 1;

   while (i == 1) {
      // Do nothing, really....
#if 0
      call();
#endif
   }
}  // comment ending TestFunction
)";

  code.SetSpacesPerIndent(3);
  EXPECT_EQ(expected, code.Render());
}

}  // namespace
}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
