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

using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("GRUBlockCell")
    .Attr("T: {float}")
    .Input("x: T")
    .Input("h_prev: T")
    .Input("w_ru: T")
    .Input("w_c: T")
    .Input("b_ru: T")
    .Input("b_c: T")
    .Output("r: T")
    .Output("u: T")
    .Output("c: T")
    .Output("h: T")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, h_prev;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &h_prev));

      DimensionHandle batch_size = c->Dim(x, 0);
      DimensionHandle cell_size = c->Dim(h_prev, 1);
      ShapeHandle output = c->Matrix(batch_size, cell_size);
      for (int i = 0; i < 4; ++i) {
        c->set_output(i, output);
      }
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Computes the GRU cell forward propagation for 1 time step.

Args
    x: Input to the GRU cell.
    h_prev: State input from the previous GRU cell.
    w_ru: Weight matrix for the reset and update gate.
    w_c: Weight matrix for the cell connection gate.
    b_ru: Bias vector for the reset and update gate.
    b_c: Bias vector for the cell connection gate.

Returns
    r: Output of the reset gate.
    u: Output of the update gate.
    c: Output of the cell connection gate.
    h: Current state of the GRU cell.

Note on notation of the variables:

Concatenation of a and b is represented by a_b
Element-wise dot product of a and b is represented by ab
Element-wise dot product is represented by \circ
Matrix multiplication is represented by *

Biases are initialized with :
`b_ru` - constant_initializer(1.0)
`b_c` - constant_initializer(0.0)

This kernel op implements the following mathematical equations:

```
x_h_prev = [x, h_prev]

[r_bar u_bar] = x_h_prev * w_ru + b_ru

r = sigmoid(r_bar)
u = sigmoid(u_bar)

h_prevr = h_prev \circ r

x_h_prevr = [x h_prevr]

c_bar = x_h_prevr * w_c + b_c
c = tanh(c_bar)

h = (1-u) \circ c + u \circ h_prev
```
)doc");

REGISTER_OP("GRUBlockCellGrad")
    .Attr("T: {float}")
    .Input("x: T")
    .Input("h_prev: T")
    .Input("w_ru: T")
    .Input("w_c: T")
    .Input("b_ru: T")
    .Input("b_c: T")
    .Input("r: T")
    .Input("u: T")
    .Input("c: T")
    .Input("d_h: T")
    .Output("d_x: T")
    .Output("d_h_prev: T")
    .Output("d_c_bar: T")
    .Output("d_r_bar_u_bar: T")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, h_prev, w_ru;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &h_prev));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &w_ru));

      DimensionHandle batch_size = c->Dim(x, 0);
      DimensionHandle cell_size = c->Dim(h_prev, 1);
      DimensionHandle twice_cell_size = c->Dim(w_ru, 1);
      ShapeHandle batch_cell_shape = c->Matrix(batch_size, cell_size);

      c->set_output(0, x);
      c->set_output(1, batch_cell_shape);
      c->set_output(2, batch_cell_shape);
      c->set_output(3, c->Matrix(batch_size, twice_cell_size));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Computes the GRU cell back-propagation for 1 time step.

Args
    x: Input to the GRU cell.
    h_prev: State input from the previous GRU cell.
    w_ru: Weight matrix for the reset and update gate.
    w_c: Weight matrix for the cell connection gate.
    b_ru: Bias vector for the reset and update gate.
    b_c: Bias vector for the cell connection gate.
    r: Output of the reset gate.
    u: Output of the update gate.
    c: Output of the cell connection gate.
    d_h: Gradients of the h_new wrt to objective function.

Returns
    d_x: Gradients of the x wrt to objective function.
    d_h_prev: Gradients of the h wrt to objective function.
    d_c_bar Gradients of the c_bar wrt to objective function.
    d_r_bar_u_bar Gradients of the r_bar & u_bar wrt to objective function.

This kernel op implements the following mathematical equations:

Note on notation of the variables:

Concatenation of a and b is represented by a_b
Element-wise dot product of a and b is represented by ab
Element-wise dot product is represented by \circ
Matrix multiplication is represented by *

Additional notes for clarity:

`w_ru` can be segmented into 4 different matrices.
```
w_ru = [w_r_x w_u_x
        w_r_h_prev w_u_h_prev]
```
Similarly, `w_c` can be segmented into 2 different matrices.
```
w_c = [w_c_x w_c_h_prevr]
```
Same goes for biases.
```
b_ru = [b_ru_x b_ru_h]
b_c = [b_c_x b_c_h]
```
Another note on notation:
```
d_x = d_x_component_1 + d_x_component_2

where d_x_component_1 = d_r_bar * w_r_x^T + d_u_bar * w_r_x^T
and d_x_component_2 = d_c_bar * w_c_x^T

d_h_prev = d_h_prev_component_1 + d_h_prevr \circ r + d_h \circ u
where d_h_prev_componenet_1 = d_r_bar * w_r_h_prev^T + d_u_bar * w_r_h_prev^T
```

Mathematics behind the Gradients below:
```
d_c_bar = d_h \circ (1-u) \circ (1-c \circ c)
d_u_bar = d_h \circ (h-c) \circ u \circ (1-u)

d_r_bar_u_bar = [d_r_bar d_u_bar]

[d_x_component_1 d_h_prev_component_1] = d_r_bar_u_bar * w_ru^T

[d_x_component_2 d_h_prevr] = d_c_bar * w_c^T

d_x = d_x_component_1 + d_x_component_2

d_h_prev = d_h_prev_component_1 + d_h_prevr \circ r + u
```
Below calculation is performed in the python wrapper for the Gradients
(not in the gradient kernel.)
```
d_w_ru = x_h_prevr^T * d_c_bar

d_w_c = x_h_prev^T * d_r_bar_u_bar

d_b_ru = sum of d_r_bar_u_bar along axis = 0

d_b_c = sum of d_c_bar along axis = 0
```
)doc");
