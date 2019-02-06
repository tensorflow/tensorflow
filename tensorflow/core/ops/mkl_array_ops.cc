/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL

// This file contains the registration of MKL-DNN array ops.

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/strided_slice_op.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::UnchangedShape;

// Adding QuantizedConcatV2 op to be able to replace it by
// _MklQuantizedConcatV2 in the graph rewrite.
REGISTER_OP("QuantizedConcatV2")
    .Input("values: N * T")
    .Input("axis: Tidx")
    .Input("input_mins: N * float32")
    .Input("input_maxes: N * float32")
    .Output("output: T")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      const int n = (c->num_inputs() - 1) / 3;
      TF_RETURN_IF_ERROR(shape_inference::QuantizedConcatV2Shape(c, n));
      ShapeHandle unused;
      for (int i = n + 1; i < c->num_inputs(); ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &unused));
      }
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedConcatV2")
    .Input("values: N * T")
    .Input("axis: Tidx")
    .Input("input_mins:  N * float32")
    .Input("input_maxes: N * float32")
    .Input("mkl_values: N * uint8")
    .Input("mkl_axis: uint8")
    .Input("mkl_input_mins:  N * uint8")
    .Input("mkl_input_maxes: N * uint8")
    .Output("output: T")
    .Output("output_min: float")
    .Output("output_max: float")
    .Output("mkl_output: uint8")
    .Output("mkl_output_min: uint8")
    .Output("mkl_output_max: uint8")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      const int n = (c->num_inputs() / 2 - 1) / 3;
      TF_RETURN_IF_ERROR(shape_inference::QuantizedConcatV2Shape(c, n));
      ShapeHandle unused;
      for (int i = n + 1; i < c->num_inputs() / 2; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &unused));
      }
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });
}

#endif
