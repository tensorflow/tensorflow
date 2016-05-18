#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("LSTMCellBlock")
    .Attr("cell_size: int")
    .Attr("forget_bias: float = 1.0")
    .Attr("T: {float, double}")
    .Input("x: T")
    .Input("cs_prev: T")
    .Input("h_prev: T")
    .Input("w: T")
    .Input("b: T")
    .Output("i: T")
    .Output("cs: T")
    .Output("f: T")
    .Output("o: T")
    .Output("ci: T")
    .Output("co: T")
    .Output("h: T")
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

cell_size: The LSTM cell size.
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

REGISTER_OP("LSTMCellBlockGrad")
    .Attr("cell_size: int")
    .Attr("T: {float, double}")
    .Input("x: T")
    .Input("cs_prev: T")
    .Input("h_prev: T")
    .Input("w: T")
    .Input("b: T")
    .Input("i: T")
    .Input("cs: T")
    .Input("f: T")
    .Input("o: T")
    .Input("ci: T")
    .Input("co: T")
    .Input("cs_grad: T")
    .Input("h_grad: T")
    .Output("x_grad: T")
    .Output("cs_prev_grad: T")
    .Output("h_prev_grad: T")
    .Output("dicfo: T")
    .Output("xh: T")
    .Doc(R"doc(
Computes the LSTM cell backward propagation for 1 timestep.

This implementation is to be used in conjunction of LSTMCellBlock.

cell_size: The LSTM cell size.
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
states: The concatenation of [cs, h].
states_grad: The gradient of states vector.
h_grad: THe gradient of h vector.
x_grad: The gradient of x.
cs_prev_grad: The gradient of cs.
h_prev_grad: The gradient of h.
dicfo: The derivative wrt to [i, cs, f, o].
xh: The concatenated vector of [x, h].
)doc");

}  // end namespace tensorflow
