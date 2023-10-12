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
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/op_implementation_renderer.h"

#include "tensorflow/c/experimental/ops/gen/common/view_util.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/arg_view.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/attr_view.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/op_view.h"

namespace tensorflow {
namespace generator {
namespace cpp {

OpImplementationRenderer::OpImplementationRenderer(RendererContext context,
                                                   OpView op)
    : Renderer(context), op_(op) {}

void OpImplementationRenderer::Render() {
  RenderInitialization();

  if (op_.IsListOp()) {
    RenderExecutionListOp();
  } else if (op_.NumOutputs() == 0) {
    RenderExecutionZeroOutputs();
  } else if (op_.NumOutputs() == 1) {
    RenderExecutionSingleOutput();
  } else {
    RenderExecutionMultipleOutputs();
  }
}

void OpImplementationRenderer::RenderInitialization() {
  // Create Op variable and initialize it
  Statement("AbstractOperationPtr $0(ctx->CreateOperation())",
            op_.VariableName());
  TFStatement(Call(op_.VariableName(), "Reset",
                   {op_.OpNameString(), "raw_device_name"}));
  TFStatement(Call("MaybeSetOpName", {op_.VariableName() + ".get()", "name"}));
  // Set each input
  for (const ArgView& ar : op_.Inputs()) {
    TFStatement(Call(op_.VariableName(), ar.SetterMethod(), ar.SetterArgs()));
  }
  // Set each attribute
  for (const AttrView& ar : op_.Attributes()) {
    TFStatement(Call(op_.VariableName(), ar.SetterMethod(), ar.SetterArgs()));
  }
}

void OpImplementationRenderer::RenderExecutionListOp() {
  ArgView output_arg = op_.OnlyOutput();
  Statement("int num_retvals = $0.size()", output_arg.VariableName());
  Statement("return " + Call(op_.VariableName(), "Execute",
                             {output_arg.VariableName(), "&num_retvals"}));
}

void OpImplementationRenderer::RenderExecutionSingleOutput() {
  ArgView output_arg = op_.OnlyOutput();
  Statement("int num_retvals = 1");
  Statement("return $0->Execute(absl::MakeSpan($1, 1), &num_retvals)",
            op_.VariableName(), output_arg.VariableName());
}

void OpImplementationRenderer::RenderExecutionMultipleOutputs() {
  Statement("int num_retvals = $0", op_.NumOutputs());
  Statement("AbstractTensorHandle* temp_outputs[$0]", op_.NumOutputs());
  Statement("Status status = $0->Execute(temp_outputs, &num_retvals)",
            op_.VariableName());

  for (const ArgView& arg : op_.Outputs()) {
    Statement("*$0 = temp_outputs[$1]", arg.VariableName(), arg.Position());
  }

  Statement("return status");
}

void OpImplementationRenderer::RenderExecutionZeroOutputs() {
  Statement("int num_retvals = 0");
  Statement("std::vector<AbstractTensorHandle*> dummy_outputs");
  Statement("return $0->Execute(absl::MakeSpan(dummy_outputs), &num_retvals)",
            op_.VariableName());
}

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
