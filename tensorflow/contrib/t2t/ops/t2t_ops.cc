/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

static auto doc = R"doc(Performs mean subtraction and L2 normalization, then applies bias and scale:

out[..., k] = (in[...,k]-<in[...,k]>)/\\sigma(in[...,k]) * scale[k] + bias[k]

where \\sigma[...,k] = \\sqrt{eps+<(in[...,k]-<in[...,k]>)^2>}, ... represents all indexes but last, 
and all averages are taken over the last index.
in: a floating-point tensor (float16 and float32 types supported).
eps: a small constant used to avoid division by zero.
scale, bias: 1-D floating point vectors.
)doc";

static auto doc_Dropout = R"doc(Performs dropout with broadcast dimensions.

out[i, j, k] = in[i, j, k] * scale if rng[c_i, c_j, c_k] >= threshold
out[i, j, k] = 0                   if rng[c_i, c_j, c_k] < threshold

where c_i=i or c_i=0, etc. depending on whether the corresponding dimension of rng has size 1; scale = 1/(1-threshold)

in: a floating-point tensor (float16 and float32 tensors of 2 or 3 dimensions supported)
rng: a floating-point tensor with the same number of dimensions as 'in';
   each dimension must either match 'in' or be equal to 1
threshold: a scalar

")doc";


#define REGISTER_NORMS(X, Y) \
REGISTER_OP("CustomL2Norm") \
    .Attr("T: " X)   \
    .Attr("U: " Y)   \
    .Input("in: T") \
    .Input("eps: U")    \
    .Input("scale: U")  \
    .Input("bias: U")   \
    .Output("out: T")   \
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {    \
      c->set_output(0, c->input(0));    \
      return Status::OK();  \
    }) \
    .Doc(doc); \
    \
REGISTER_OP("CustomL2NormGrad") \
    .Attr("T: " X)   \
    .Attr("U: " Y)   \
    .Input("in: T") \
    .Input("eps: U")    \
    .Input("scale: U")  \
    .Input("bias: U")   \
    .Input("outgrad: T")   \
    .Output("out: T")   \
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {    \
      c->set_output(0, c->input(0));    \
      return Status::OK();  \
    }).Doc("Helper function for backpropagation through CustomL2Norm; do not call directly."); 


#define REGISTER_DROPOUTS(X) \
REGISTER_OP("CustomDropout") \
    .Attr("T: " X)   \
    .Input("in: T") \
    .Input("rng: T")    \
    .Input("threshold: T")  \
    .Output("out: T")   \
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {    \
      c->set_output(0, c->input(0));    \
      return Status::OK();  \
    }).Doc(doc_Dropout); 

REGISTER_NORMS("{float, float16}","{float, float16}")
REGISTER_DROPOUTS("{float, float16}")

}  // namespace tensorflow
