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

namespace tensorflow {

REGISTER_OP("LSTMFusedCell")
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

REGISTER_OP("LSTMFusedCellGrad")
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
    .Doc(R"doc(
Computes the LSTM cell backward propagation for 1 timestep.

This implementation is to be used in conjunction of LSTMFusedCell.

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

REGISTER_OP("FusedLSTM")
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
    .Doc(R"doc(
)doc");

REGISTER_OP("FusedLSTMGrad")
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
    .Doc(R"doc(
)doc");

}  // end namespace tensorflow
