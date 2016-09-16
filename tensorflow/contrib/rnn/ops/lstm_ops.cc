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

This implementation uses 1 weight matrix and 1 bias vector, there is no
diagonal peephole connection.

This kernel op implements the following mathematical equations:

```python
xh = [x, h_prev]
[i, f, ci, o] = xh * w + b
f = f + forget_bias

i = sigmoid(i)
f = sigmoid(f)
ci = tanh(ci)
o = sigmoid(o)

cs = ci .* i + cs_prev .* f
co = tanh(cs)

h = co .* o
```

forget_bias: The forget gate bias.
x: The input to the LSTM cell.
w: The weight matrix.
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

x: The input to the LSTM cell.
cs_prev: The previous cell state.
h_prev: The previous h state.
w: The weight matrix.
b: The bias vector.
i: The input gate.
cs: The cell state before the tanh.
f: The forget gate.
o: The output gate.
ci: The cell input.
co: The cell after the tanh.
h_grad: THe gradient of h vector.
cs_prev_grad: The gradient of cs.
dicfo: The derivative wrt to [i, cs, f, o].
)doc");

REGISTER_OP("BlockLSTM")
    .Input("seq_len_max: int64")
    .Input("x: max_len * T")
    .Input("cs_prev: T")
    .Input("h_prev: T")
    .Input("w: T")
    .Input("wci: T")
    .Input("wcf: T")
    .Input("wco: T")
    .Input("b: T")
    .Output("i: max_len * T")
    .Output("cs: max_len * T")
    .Output("f: max_len * T")
    .Output("o: max_len * T")
    .Output("ci: max_len * T")
    .Output("co: max_len * T")
    .Output("h: max_len * T")
    .Attr("max_len: int")
    .Attr("forget_bias: float = 1.0")
    .Attr("cell_clip: float = 3.0")
    .Attr("use_peephole: bool = false")
    .Attr("T: {float}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, b;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs() - 1), 1, &b));

      DimensionHandle batch_size = c->Dim(x, 0);
      DimensionHandle cell_size;
      TF_RETURN_IF_ERROR(
          c->Divide(c->Dim(b, 0), 4, true /* evenly_divisible */, &cell_size));

      int64 max_len;
      TF_RETURN_IF_ERROR(c->GetAttr("max_len", &max_len));

      DCHECK_EQ(max_len * 7, c->num_outputs());
      ShapeHandle output = c->Matrix(batch_size, cell_size);
      for (int i = 0; i < max_len; ++i) {
        for (int j = 0; j < 7; ++j) {
          c->set_output(i * 7 + j, output);
        }
      }
      return Status::OK();
    })
    .Doc(R"doc(
)doc");

REGISTER_OP("BlockLSTMGrad")
    .Input("seq_len_max: int64")
    .Input("x: max_len * T")
    .Input("cs_prev: T")
    .Input("h_prev: T")
    .Input("w: T")
    .Input("wci: T")
    .Input("wcf: T")
    .Input("wco: T")
    .Input("b: T")
    .Input("i: max_len * T")
    .Input("cs: max_len * T")
    .Input("f: max_len * T")
    .Input("o: max_len * T")
    .Input("ci: max_len * T")
    .Input("co: max_len * T")
    .Input("h: max_len * T")
    .Input("cs_grad: max_len * T")
    .Input("h_grad: max_len * T")
    .Output("x_grad: max_len * T")
    .Output("cs_prev_grad: T")
    .Output("h_prev_grad: T")
    .Output("w_grad: T")
    .Output("wci_grad: T")
    .Output("wcf_grad: T")
    .Output("wco_grad: T")
    .Output("b_grad: T")
    .Attr("max_len: int")
    .Attr("use_peephole: bool")
    .Attr("T: {float}")
    .SetShapeFn([](InferenceContext* c) {
      int64 max_len;
      TF_RETURN_IF_ERROR(c->GetAttr("max_len", &max_len));

      ShapeHandle x, cs_prev, h_prev, w, wci, wco, wcf, b;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1 + max_len), 2, &cs_prev));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2 + max_len), 2, &h_prev));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3 + max_len), 2, &w));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4 + max_len), 1, &wci));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5 + max_len), 1, &wco));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6 + max_len), 1, &wcf));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7 + max_len), 1, &b));

      int out_idx = 0;
      for (int i = 0; i < max_len; ++i) c->set_output(out_idx++, x);
      c->set_output(out_idx++, cs_prev);
      c->set_output(out_idx++, h_prev);
      c->set_output(out_idx++, w);
      c->set_output(out_idx++, wci);
      c->set_output(out_idx++, wco);
      c->set_output(out_idx++, wcf);
      c->set_output(out_idx++, b);

      return Status::OK();
    })
    .Doc(R"doc(
)doc");

}  // end namespace tensorflow
