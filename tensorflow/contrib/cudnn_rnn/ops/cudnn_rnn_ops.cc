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

constexpr auto kCudnnRNNCommonAttrs = R"doc(
rnn_mode: Indicates the type of the RNN model.
input_mode: Indicate whether there is a linear projection between the input and
    The actual computation before the first layer. 'skip_input' is only allowed
    when input_size == num_units; 'auto_select' implies 'skip_input' when
    input_size == num_units; otherwise, it implies 'linear_input'.
direction: Indicates whether a bidirectional model will be used.
    dir = (direction == bidirectional) ? 2 : 1
)doc";

constexpr auto kCudnnRNNParamsBuffer = R"doc(
Note that the params buffer may not be compatible across different GPUs. So any
save and restoration should be converted to and from the canonical weights and
biases.
)doc";

constexpr auto kRNNModeAttrs =
    "rnn_mode: {'rnn_relu', 'rnn_tanh', 'lstm', 'gru'} = 'lstm'";

constexpr auto kRNNInputModeAttrs =
    "input_mode: {'linear_input', 'skip_input', 'auto_select'} = "
    "'auto_select'";

constexpr auto kRNNDirectionAttrs =
    "direction: {'unidirectional', 'bidirectional'} = 'unidirectional'";

constexpr auto kCudnnRNNCanonicalParams = R"doc(
canonical_weights: the canonical form of weights that can be used for saving
    and restoration. They are more likely to be compatible across different
    generations.
canonical_biases: the canonical form of biases that can be used for saving and
    restoration. They are more likely to be compatible across different
    generations.
)doc";

}  // namespace

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("CudnnRNNParamsSize")
    .Input("num_layers: int32")
    .Input("num_units: int32")
    .Input("input_size: int32")
    .Attr("T: {float}")
    .Attr("S: {int32, int64}")
    .Attr(kRNNModeAttrs)
    .Attr(kRNNInputModeAttrs)
    .Attr(kRNNDirectionAttrs)
    .Output("params_size: S")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(1));
      return Status::OK();
    })
    .Doc(strings::StrCat(R"doc(
Return the params size that can be used by the Cudnn RNN model. Subsequent
weight allocation and initialization should use this size.
)doc",
                         kCudnnRNNCommonAttrs,
                         R"doc(
num_layers: Specifies the number of layers in the RNN model.
num_units: Specifies the size of the hidden state.
input_size: Specifies the size of the input state.
params_size: The size of the params buffer that should be allocated and
    initialized for this RNN model. Note that this params buffer may not be
    compatible across GPUs. Please use CudnnRNNParamsWeights and
    CudnnRNNParamsBiases to save and restore them in a way that is compatible
    across different runs.
)doc",
                         kCudnnRNNParamsBuffer));

static string CudnnRNNForwardTensors() {
  return R"doc(
input: a 3-D tensor with the shape of [seq_length, batch_size, input_size].
input_h: a 3-D tensor with the shape of [num_layer * dir, batch_size,
    num_units].
input_c: For LSTM, a 3-D tensor with the shape of
    [num_layer * dir, batch, num_units]. For other models, it is ignored.
params: a 1-D tensor that contains the weights and biases in an opaque layout.
    The size must be created through CudnnRNNParamsSize, and initialized
    separately. Note that they might not be compatible across different
    generations. So it is a good idea to save and restore
output: a 3-D tensor with the shape of [seq_length, batch_size,
    dir * num_units].
output_h: the same shape has input_h.
output_c: the same shape as input_c for LSTM. An empty tensor for other models.
)doc";
}

REGISTER_OP("CudnnRNN")
    .Input("input: T")
    .Input("input_h: T")
    .Input("input_c: T")
    .Input("params: T")
    .Output("output: T")
    .Output("output_h: T")
    .Output("output_c: T")
    .Output("reserve_space: T")
    .Attr("T: {float}")
    .Attr(kRNNModeAttrs)
    .Attr(kRNNInputModeAttrs)
    .Attr(kRNNDirectionAttrs)
    .Attr("dropout: float")
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
    })
    .Doc(strings::StrCat(R"doc(
Computes the RNN from the input and initial states, with respect to the params
buffer.
)doc",
                         kCudnnRNNCommonAttrs, CudnnRNNForwardTensors(), R"doc(
is_training: Indicates whether this operation is used for inferenece or
    training.
reserve_space: an opaque tensor that can be used in backprop calculation. It
    is only produced if is_training is false.
)doc"));

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
    .Output("input_backprop: T")
    .Output("input_h_backprop: T")
    .Output("input_c_backprop: T")
    .Output("params_backprop: T")
    .Attr("T: {float}")
    .Attr(kRNNModeAttrs)
    .Attr(kRNNInputModeAttrs)
    .Attr(kRNNDirectionAttrs)
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
    })
    .Doc(strings::StrCat(R"doc(
Compute the backprop of both data and weights in a RNN.
)doc",
                         kCudnnRNNCommonAttrs, CudnnRNNForwardTensors(), R"doc(
output_backprop: A 3-D tensor with the same shape as output in the forward pass.
output_h_backprop: A 3-D tensor with the same shape as output_h in the forward
    pass.
output_c_backprop: A 3-D tensor with the same shape as output_c in the forward
    pass.
reserve_space: The same reserve_space produced in for forward operation.
input_backprop: The backprop to input in the forward pass. Has the same shape
    as input.
input_h_backprop: The backprop to input_h in the forward pass. Has the same
    shape as input_h.
input_c_backprop: The backprop to input_c in the forward pass. Has the same
    shape as input_c.
params_backprop: The backprop to the params buffer in the forward pass. Has the
    same shape as params.
)doc"));

// NOTE(zhengxq): this is not currently implemented yet. And may subject to
// change.
REGISTER_OP("CudnnRNNParamsToCanonical")
    .Input("num_layers: int32")
    .Input("num_units: int32")
    .Input("input_size: int32")
    .Input("params: T")
    .Output("canonical_weights: T")
    .Output("canonical_biases: T")
    .Attr("T: {float}")
    .Attr("N: int >= 1")
    .Attr(kRNNModeAttrs)
    .Attr(kRNNInputModeAttrs)
    .Attr(kRNNDirectionAttrs)
    .Doc(strings::StrCat(R"doc(
Retrieves a set of weights from the opaque params buffer that can be saved and
restored in a way compatible with future runs.
)doc",
                         kCudnnRNNCommonAttrs, kCudnnRNNParamsBuffer,
                         kCudnnRNNCanonicalParams));

// NOTE(zhengxq): this is not currently implemented yet. And may subject to
// change.
REGISTER_OP("CudnnRNNParamsFromCanonical")
    .Input("num_layers: int32")
    .Input("num_units: int32")
    .Input("input_size: int32")
    .Input("params: Ref(T)")
    .Input("canonical_weights: T")
    .Input("canonical_biases: T")
    .Attr("T: {float}")
    .Attr("N: int >= 1")
    .Attr(kRNNModeAttrs)
    .Attr(kRNNInputModeAttrs)
    .Attr(kRNNDirectionAttrs)
    .Doc(strings::StrCat(R"doc(
Writes a set of weights into the the opaque params buffer so they can be used in
upcoming training or inferences.
)doc",
                         kCudnnRNNCommonAttrs, kCudnnRNNParamsBuffer,
                         kCudnnRNNCanonicalParams));

}  // namespace tensorflow
