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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("LSTMBlockCell")
    .Input("x: T")
    .Input("cs_prev: T")
    .Input("h_prev: T")
    .Input("w: T")
    .Input("wci: T")
    .Input("wcf: T")
    .Input("wco: T")
    .Input("b: T")
    .Output("i: T")
    .Output("cs: T")
    .Output("f: T")
    .Output("o: T")
    .Output("ci: T")
    .Output("co: T")
    .Output("h: T")
    .Attr("forget_bias: float = 1.0")
    .Attr("cell_clip: float = 3.0")
    .Attr("use_peephole: bool = false")
    .Attr("T: {float}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, cs_prev;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &cs_prev));

      DimensionHandle batch_size = c->Dim(x, 0);
      DimensionHandle cell_size = c->Dim(cs_prev, 1);
      ShapeHandle output = c->Matrix(batch_size, cell_size);
      for (int i = 0; i < 7; ++i) {
        c->set_output(i, output);
      }
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Computes the LSTM cell forward propagation for 1 time step.

This implementation uses 1 weight matrix and 1 bias vector, and there's an
optional peephole connection.

This kernel op implements the following mathematical equations:

```python
xh = [x, h_prev]
[i, f, ci, o] = xh * w + b
f = f + forget_bias

if not use_peephole:
  wci = wcf = wco = 0

i = sigmoid(cs_prev * wci + i)
f = sigmoid(cs_prev * wcf + f)
ci = tanh(ci)

cs = ci .* i + cs_prev .* f
cs = clip(cs, cell_clip)

o = sigmoid(cs * wco + f)
co = tanh(cs)
h = co .* o
```

cell_clip: Value to clip the 'cs' value to.
use_peephole: Whether to use peephole weights.
forget_bias: The forget gate bias.

x: The input to the LSTM cell, shape (batch_size, num_inputs).
cs_prev: Value of the cell state at previous time step.
h_prev: Output of the previous cell at previous time step.
w: The weight matrix.
wci: The weight matrix for input gate peephole connection.
wcf: The weight matrix for forget gate peephole connection.
wco: The weight matrix for output gate peephole connection.
b: The bias vector.

i: The input gate.
cs: The cell state before the tanh.
f: The forget gate.
o: The output gate.
ci: The cell input.
co: The cell after the tanh.
h: The output h vector.
)doc");

REGISTER_OP("LSTMBlockCellGrad")
    .Input("x: T")
    .Input("cs_prev: T")
    .Input("h_prev: T")
    .Input("w: T")
    .Input("wci: T")
    .Input("wcf: T")
    .Input("wco: T")
    .Input("b: T")
    .Input("i: T")
    .Input("cs: T")
    .Input("f: T")
    .Input("o: T")
    .Input("ci: T")
    .Input("co: T")
    .Input("cs_grad: T")
    .Input("h_grad: T")
    .Output("cs_prev_grad: T")
    .Output("dicfo: T")
    .Output("wci_grad: T")
    .Output("wcf_grad: T")
    .Output("wco_grad: T")
    .Attr("use_peephole: bool")
    .Attr("T: {float}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, cs_prev;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &cs_prev));

      DimensionHandle batch_size = c->Dim(x, 0);
      DimensionHandle cell_size = c->Dim(cs_prev, 1);
      DimensionHandle cell_size_times_4;
      TF_RETURN_IF_ERROR(c->Multiply(cell_size, 4, &cell_size_times_4));
      ShapeHandle cell_size_vec = c->Vector(cell_size);

      c->set_output(0, c->Matrix(batch_size, cell_size));
      c->set_output(1, c->Matrix(batch_size, cell_size_times_4));
      c->set_output(2, cell_size_vec);
      c->set_output(3, cell_size_vec);
      c->set_output(4, cell_size_vec);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Computes the LSTM cell backward propagation for 1 timestep.

This implementation is to be used in conjunction of LSTMBlockCell.

use_peephole: Whether the cell uses peephole connections.
x: The input to the LSTM cell, shape (batch_size, num_inputs).
cs_prev: The previous cell state.
h_prev: The previous h state.
w: The weight matrix.
wci: The weight matrix for input gate peephole connection.
wcf: The weight matrix for forget gate peephole connection.
wco: The weight matrix for output gate peephole connection.
b: The bias vector.
i: The input gate.
cs: The cell state before the tanh.
f: The forget gate.
o: The output gate.
ci: The cell input.
co: The cell after the tanh.
cs_grad: The current gradient of cs.
h_grad: The gradient of h vector.
cs_prev_grad: The gradient of cs to be back-propped.
dicfo: The derivative wrt to [i, cs, f, o].
wci_grad: The gradient for wci to be back-propped.
wcf_grad: The gradient for wcf to be back-propped.
wco_grad: The gradient for wco to be back-propped.
)doc");

REGISTER_OP("BlockLSTM")
    .Input("seq_len_max: int64")
    .Input("x: T")
    .Input("cs_prev: T")
    .Input("h_prev: T")
    .Input("w: T")
    .Input("wci: T")
    .Input("wcf: T")
    .Input("wco: T")
    .Input("b: T")
    .Output("i: T")
    .Output("cs: T")
    .Output("f: T")
    .Output("o: T")
    .Output("ci: T")
    .Output("co: T")
    .Output("h: T")
    .Attr("forget_bias: float = 1.0")
    .Attr("cell_clip: float = 3.0")
    .Attr("use_peephole: bool = false")
    .Attr("T: {float}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, b;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs() - 1), 1, &b));

      DimensionHandle timelen = c->Dim(x, 0);
      DimensionHandle batch_size = c->Dim(x, 1);
      DimensionHandle cell_size;
      TF_RETURN_IF_ERROR(
          c->Divide(c->Dim(b, 0), 4, true /* evenly_divisible */, &cell_size));

      DCHECK_EQ(7, c->num_outputs());
      ShapeHandle output = c->MakeShape({timelen, batch_size, cell_size});
      for (int i = 0; i < 7; ++i) {
        c->set_output(i, output);
      }
      return Status::OK();
    })
    .Doc(R"doc(
Computes the LSTM cell forward propagation for all the time steps.

This is equivalent to applying LSTMBlockCell in a loop, like so:

```python
for x1 in unpack(x):
  i1, cs1, f1, o1, ci1, co1, h1 = LSTMBlock(
    x1, cs_prev, h_prev, w, wci, wcf, wco, b)
  cs_prev = cs1
  h_prev = h1
  i.append(i1)
  cs.append(cs1)
  f.append(f1)
  o.append(o1)
  ci.append(ci1)
  co.append(co1)
  h.append(h1)
return pack(i), pack(cs), pack(f), pack(o), pack(ci), pack(ch), pack(h)
```

cell_clip: Value to clip the 'cs' value to.
use_peephole: Whether to use peephole weights.
forget_bias: The forget gate bias.

seq_len_max: Maximum time length actually used by this input. Outputs are padded
  with zeros beyond this length.
x: The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
cs_prev: Value of the initial cell state.
h_prev: Initial output of cell (to be used for peephole).
w: The weight matrix.
wci: The weight matrix for input gate peephole connection.
wcf: The weight matrix for forget gate peephole connection.
wco: The weight matrix for output gate peephole connection.
b: The bias vector.

i: The input gate over the whole time sequence.
cs: The cell state before the tanh over the whole time sequence.
f: The forget gate over the whole time sequence.
o: The output gate over the whole time sequence.
ci: The cell input over the whole time sequence.
co: The cell after the tanh over the whole time sequence.
h: The output h vector over the whole time sequence.
)doc");

REGISTER_OP("BlockLSTMGrad")
    .Input("seq_len_max: int64")
    .Input("x: T")
    .Input("cs_prev: T")
    .Input("h_prev: T")
    .Input("w: T")
    .Input("wci: T")
    .Input("wcf: T")
    .Input("wco: T")
    .Input("b: T")
    .Input("i: T")
    .Input("cs: T")
    .Input("f: T")
    .Input("o: T")
    .Input("ci: T")
    .Input("co: T")
    .Input("h: T")
    .Input("cs_grad: T")
    .Input("h_grad: T")
    .Output("x_grad: T")
    .Output("cs_prev_grad: T")
    .Output("h_prev_grad: T")
    .Output("w_grad: T")
    .Output("wci_grad: T")
    .Output("wcf_grad: T")
    .Output("wco_grad: T")
    .Output("b_grad: T")
    .Attr("use_peephole: bool")
    .Attr("T: {float}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, cs_prev, h_prev, w, wci, wco, wcf, b;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &cs_prev));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &h_prev));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &w));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &wci));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 1, &wco));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 1, &wcf));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 1, &b));

      c->set_output(0, x);
      c->set_output(1, cs_prev);
      c->set_output(2, h_prev);
      c->set_output(3, w);
      c->set_output(4, wci);
      c->set_output(5, wco);
      c->set_output(6, wcf);
      c->set_output(7, b);

      return Status::OK();
    })
    .Doc(R"doc(
Computes the LSTM cell backward propagation for the entire time sequence.

This implementation is to be used in conjunction of LSTMBlock.

use_peephole: Whether to use peephole weights.

seq_len_max: Maximum time length actually used by this input. Outputs are padded
  with zeros beyond this length.
x: The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
cs_prev: Value of the initial cell state.
h_prev: Initial output of cell (to be used for peephole).
w: The weight matrix.
wci: The weight matrix for input gate peephole connection.
wcf: The weight matrix for forget gate peephole connection.
wco: The weight matrix for output gate peephole connection.
b: The bias vector.
i: The input gate over the whole time sequence.
cs: The cell state before the tanh over the whole time sequence.
f: The forget gate over the whole time sequence.
o: The output gate over the whole time sequence.
ci: The cell input over the whole time sequence.
co: The cell after the tanh over the whole time sequence.
h: The output h vector over the whole time sequence.
cs_grad: The current gradient of cs.
h_grad: The gradient of h vector.

x_grad: The gradient of x to be back-propped.
cs_prev_grad: The gradient of cs_prev to be back-propped.
h_prev_grad: The gradient of h_prev to be back-propped.
w_grad: The gradient for w to be back-propped.
wci_grad: The gradient for wci to be back-propped.
wcf_grad: The gradient for wcf to be back-propped.
wco_grad: The gradient for wco to be back-propped.
b_grad: The gradient for w to be back-propped.
)doc");

}  // end namespace tensorflow
