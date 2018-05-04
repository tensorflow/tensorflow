/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace {

constexpr auto kRNNModeAttrs =
    "rnn_mode: {'rnn_relu', 'rnn_tanh', 'lstm', 'gru'} = 'lstm'";

constexpr auto kRNNInputModeAttrs =
    "input_mode: {'linear_input', 'skip_input', 'auto_select'} = "
    "'linear_input'";

constexpr auto kRNNDirectionAttrs =
    "direction: {'unidirectional', 'bidirectional'} = 'unidirectional'";

}  // namespace

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;


REGISTER_OP("CudnnRNNParamsSize")
    .Input("num_layers: int32")
    .Input("num_units: int32")
    .Input("input_size: int32")
    .Attr("T: {float16, float32, float64}")
    .Attr("S: {int32, int64}")
    .Attr(kRNNModeAttrs)
    .Attr(kRNNInputModeAttrs)
    .Attr(kRNNDirectionAttrs)
    .Attr("dropout: float = 0.0")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Output("params_size: S")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(1));
      return Status::OK();
    });


REGISTER_OP("CudnnRNN")
    .Input("input: T")
    .Input("input_h: T")
    .Input("input_c: T")
    .Input("params: T")
    .SetIsStateful()
    .Output("output: T")
    .Output("output_h: T")
    .Output("output_c: T")
    .Output("reserve_space: T")
    .Attr("T: {float16, float32, float64}")
    .Attr(kRNNModeAttrs)
    .Attr(kRNNInputModeAttrs)
    .Attr(kRNNDirectionAttrs)
    .Attr("dropout: float = 0.0")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("is_training: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      auto input_shape = c->input(0);
      auto input_h_shape = c->input(1);
      auto seq_length = c->Dim(input_shape, 0);
      auto batch_size = c->Dim(input_shape, 1);
      auto num_units = c->Dim(input_h_shape, 2);
      string direction;
      TF_RETURN_IF_ERROR(c->GetAttr("direction", &direction));
      string rnn_mode;
      TF_RETURN_IF_ERROR(c->GetAttr("rnn_mode", &rnn_mode));
      int dir_count = (direction == "bidirectional") ? 2 : 1;
      DimensionHandle output_size;
      TF_RETURN_IF_ERROR(c->Multiply(num_units, dir_count, &output_size));
      auto output_shape = c->MakeShape({seq_length, batch_size, output_size});
      auto output_h_shape = input_h_shape;
      auto output_c_shape TF_ATTRIBUTE_UNUSED =
          (rnn_mode == "lstm") ? output_h_shape : c->MakeShape({});
      c->set_output(0, output_shape);
      c->set_output(1, output_h_shape);
      c->set_output(2, output_c_shape);
      c->set_output(3, c->UnknownShape());
      return Status::OK();
    });


REGISTER_OP("CudnnRNNBackprop")
    .Input("input: T")
    .Input("input_h: T")
    .Input("input_c: T")
    .Input("params: T")
    .Input("output: T")
    .Input("output_h: T")
    .Input("output_c: T")
    .Input("output_backprop: T")
    .Input("output_h_backprop: T")
    .Input("output_c_backprop: T")
    .Input("reserve_space: T")
    .SetIsStateful()
    .Output("input_backprop: T")
    .Output("input_h_backprop: T")
    .Output("input_c_backprop: T")
    .Output("params_backprop: T")
    .Attr("T: {float16, float32, float64}")
    .Attr(kRNNModeAttrs)
    .Attr(kRNNInputModeAttrs)
    .Attr(kRNNDirectionAttrs)
    .Attr("dropout: float = 0.0")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn([](InferenceContext* c) {
      auto input_shape = c->input(0);
      auto input_h_shape = c->input(1);
      auto input_c_shape = c->input(2);
      auto params_shape = c->input(3);
      c->set_output(0, input_shape);
      c->set_output(1, input_h_shape);
      c->set_output(2, input_c_shape);
      c->set_output(3, params_shape);
      return Status::OK();
    });


REGISTER_OP("CudnnRNNParamsToCanonical")
    .Input("num_layers: int32")
    .Input("num_units: int32")
    .Input("input_size: int32")
    .Input("params: T")
    .Output("weights: num_params * T")
    .Output("biases: num_params * T")
    .Attr("T: {float16, float32, float64}")
    .Attr("num_params: int")
    .Attr(kRNNModeAttrs)
    .Attr(kRNNInputModeAttrs)
    .Attr(kRNNDirectionAttrs)
    .Attr("dropout: float = 0.0")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &unused));
      int num_params;
      TF_RETURN_IF_ERROR(c->GetAttr("num_params", &num_params));
      // Set shape for weight matrices
      for (int i = 0; i < num_params; i++) {
        c->set_output(i, c->Matrix(InferenceContext::kUnknownDim,
                                   InferenceContext::kUnknownDim));
      }
      // Set shape for bias vectors
      for (int i = 0; i < num_params; i++) {
        c->set_output(num_params + i, c->Vector(InferenceContext::kUnknownDim));
      }
      return Status::OK();
    });


REGISTER_OP("CudnnRNNCanonicalToParams")
    .Input("num_layers: int32")
    .Input("num_units: int32")
    .Input("input_size: int32")
    .Input("weights: num_params * T")
    .Input("biases: num_params * T")
    .Output("params: T")
    .Attr("T: {float16, float32, float64}")
    .Attr("num_params: int")
    .Attr(kRNNModeAttrs)
    .Attr(kRNNInputModeAttrs)
    .Attr(kRNNDirectionAttrs)
    .Attr("dropout: float = 0.0")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    });

}  // namespace tensorflow
