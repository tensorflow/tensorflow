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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/rng_alg.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

static Status StatelessShapeV2(InferenceContext* c) {
  // Check key and counter shapes
  ShapeHandle key;
  ShapeHandle counter;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &key));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &counter));
  shape_inference::ShapeHandle unused_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused_shape));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(key, 0), RNG_KEY_SIZE, &unused));

  // Set output shape
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
  c->set_output(0, out);
  return Status::OK();
}

#define REGISTER_STATELESS_OP(name)                           \
  REGISTER_OP(name)                                           \
      .Input("shape: Tshape")                                 \
      .Input("key: uint64")                                   \
      .Input("counter: uint64")                               \
      .Input("alg: int32")                                    \
      .Output("output: dtype")                                \
      .Attr("dtype: {half,bfloat16,float,double} = DT_FLOAT") \
      .Attr("Tshape: {int32, int64} = DT_INT32")              \
      .SetShapeFn(StatelessShapeV2)

REGISTER_STATELESS_OP("StatelessRandomUniformV2");
REGISTER_STATELESS_OP("StatelessRandomNormalV2");
REGISTER_STATELESS_OP("StatelessTruncatedNormalV2");

#undef REGISTER_STATELESS_OP

REGISTER_OP("StatelessRandomUniformIntV2")
    .Input("shape: Tshape")
    .Input("key: uint64")
    .Input("counter: uint64")
    .Input("alg: int32")
    .Input("minval: dtype")
    .Input("maxval: dtype")
    .Output("output: dtype")
    .Attr("dtype: {int32, int64, uint32, uint64}")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      Status s = c->WithRank(c->input(4), 0, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "minval must be a scalar; got a tensor of shape ",
            c->DebugString(c->input(4)));
      }
      s = c->WithRank(c->input(5), 0, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "maxval must be a scalar; got a tensor of shape ",
            c->DebugString(c->input(5)));
      }
      return StatelessShapeV2(c);
    });

REGISTER_OP("StatelessRandomUniformFullIntV2")
    .Input("shape: Tshape")
    .Input("key: uint64")
    .Input("counter: uint64")
    .Input("alg: int32")
    .Output("output: dtype")
    .Attr("dtype: {int32, int64, uint32, uint64} = DT_UINT64")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .SetShapeFn(StatelessShapeV2);

REGISTER_OP("StatelessRandomGetKeyCounterAlg")
    .Input("seed: Tseed")
    .Output("key: uint64")
    .Output("counter: uint64")
    .Output("alg: int32")
    .Attr("Tseed: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      // Check seed shape
      ShapeHandle seed;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &seed));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(seed, 0), 2, &unused));

      // Set output shapes
      c->set_output(0, c->MakeShape({RNG_KEY_SIZE}));
      c->set_output(1, c->MakeShape({RNG_MAX_COUNTER_SIZE}));
      c->set_output(2, c->MakeShape({}));
      return Status::OK();
    });

REGISTER_OP("StatelessRandomGetKeyCounter")
    .Input("seed: Tseed")
    .Output("key: uint64")
    .Output("counter: uint64")
    .Attr("Tseed: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      // Check seed shape
      ShapeHandle seed;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &seed));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(seed, 0), 2, &unused));

      // Set output shapes
      c->set_output(0, c->MakeShape({RNG_KEY_SIZE}));
      c->set_output(1, c->MakeShape({RNG_MAX_COUNTER_SIZE}));
      return Status::OK();
    });

REGISTER_OP("StatelessRandomGetAlg")
    .Output("alg: int32")
    .SetIsStateful()  // because outputs depend on device
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->MakeShape({}));
      return Status::OK();
    });

}  // namespace tensorflow
