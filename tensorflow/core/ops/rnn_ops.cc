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
      return OkStatus();
    });

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
      return OkStatus();
    });

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
    .Attr("T: {half, float}")
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
      return OkStatus();
    });

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
    .Attr("T: {half, float}")
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
      return OkStatus();
    });

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
    .Attr("T: {half, float}")
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
      return OkStatus();
    });

REGISTER_OP("BlockLSTMV2")
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
    .Attr("cell_clip: float = 0.0")
    .Attr("use_peephole: bool = false")
    .Attr("T: {half, float}")
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
      return OkStatus();
    });

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
    .Attr("T: {half, float}")
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

      return OkStatus();
    });

REGISTER_OP("BlockLSTMGradV2")
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
    .Attr("T: {half, float}")
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

      return OkStatus();
    });

}  // end namespace tensorflow
