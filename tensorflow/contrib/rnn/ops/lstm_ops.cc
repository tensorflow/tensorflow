#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("LSTMCellBlock")
    .Attr("cell_size: int")
    .Attr("forget_bias: float = 1.0")
    .Attr("T: {float, double}")
    .Input("x: T")
    .Input("states_prev: T")
    .Input("w: T")
    .Input("b: T")
    .Output("i: T")
    .Output("cs: T")
    .Output("f: T")
    .Output("o: T")
    .Output("ci: T")
    .Output("co: T")
    .Output("states: T")
    .Output("h: T")
    .Doc(R"doc(
Computes the LSTM cell forward propagation for 1 time step.

This implementation uses 1 weight matrix and 1 bias vector, there is no
diagonal peephole connection.

This kernel op implements the following mathematical equations:

```python
[cs_prev, h_prev] = states_prev

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
states = [cs, h]
```

cell_size: The LSTM cell size.
forget_bias: The forget gate bias.
x: The input to the LSTM cell.
states_prev: The previous LSTM state of [cs, h].
w: The weight matrix.
b: The bias vector.
i: The input gate.
cs: The cell state before the tanh.
f: The forget gate.
o: The output gate.
ci: The cell input.
co: The cell after the tanh.
states: The concatenation of [cs, h].
h: The output h vector.
)doc");

REGISTER_OP("LSTMCellBlockGrad")
    .Attr("cell_size: int")
    .Attr("T: {float, double}")
    .Input("x: T")
    .Input("states_prev: T")
    .Input("w: T")
    .Input("b: T")
    .Input("i: T")
    .Input("cs: T")
    .Input("f: T")
    .Input("o: T")
    .Input("ci: T")
    .Input("co: T")
    .Input("h: T")
    .Input("states_grad: T")
    .Input("h_grad: T")
    .Output("x_grad: T")
    .Output("states_prev_grad: T")
    .Output("dicfo: T")
    .Output("xh: T")
    .Doc(R"doc(
Computes the LSTM cell backward propagation for 1 timestep.

This implementation is to be used inconjunction of LSTMCellBlock.

cell_size: The LSTM cell size.
x: The input to the LSTM cell.
states_prev: The previous LSTM state (it is a concatenated vector of c[t - 1]
  and h[t - 1].
w: The weight matrix.
b: The bias vector.
i: The input gate.
cs: The cell state before the tanh.
f: The forget gate.
o: The output gate.
ci: The cell input.
co: The cell after the tanh.
states: The concatenation of [cs, h].
h: The output h vector.
states_grad: The gradient of states vector.
h_grad: THe gradient of h vector.
x_grad: The gradient of x.
states_prev_grad: The gradient of states_prev.
dicfo: The derivative wrt to [i, cs, f, o].
xh: The concatenated vector of [x, h].
)doc");

}  // end namespace tensorflow
