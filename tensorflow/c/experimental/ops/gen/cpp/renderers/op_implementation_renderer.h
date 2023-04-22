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
#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_RENDERERS_OP_IMPLEMENTATION_RENDERER_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_RENDERERS_OP_IMPLEMENTATION_RENDERER_H_

#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/op_view.h"

namespace tensorflow {
namespace generator {
namespace cpp {

class OpImplementationRenderer : public Renderer {
 public:
  explicit OpImplementationRenderer(RendererContext context, OpView op);
  void Render();

 private:
  void RenderInitialization();
  void RenderExecutionListOp();
  void RenderExecutionMultipleOutputs();
  void RenderExecutionZeroOutputs();
  void RenderExecutionSingleOutput();

  OpView op_;
};

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_RENDERERS_OP_IMPLEMENTATION_RENDERER_H_
