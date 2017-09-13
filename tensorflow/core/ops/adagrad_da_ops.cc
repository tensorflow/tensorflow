/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/ops/training_ops.h"

namespace tensorflow {

static Status ApplyAdagradDAShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // grad_accumulator
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2),
                              &s));  // gradient_squared_accumulator
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 3 /* grad_idx */, &s));
  int idx = sparse ? 5 : 4;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // global step
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyAdagradDA")
    .Input("var: Ref(T)")
    .Input("gradient_accumulator: Ref(T)")
    .Input("gradient_squared_accumulator: Ref(T)")
    .Input("grad: T")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("global_step: int64")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdagradDAShapeFn(c, false /* sparse */);
    })
    .Doc(R"doc(
Update '*var' according to the proximal adagrad scheme.

var: Should be from a Variable().
gradient_accumulator: Should be from a Variable().
gradient_squared_accumulator: Should be from a Variable().
grad: The gradient.
lr: Scaling factor. Must be a scalar.
l1: L1 regularization. Must be a scalar.
l2: L2 regularization. Must be a scalar.
global_step: Training step number. Must be a scalar.
out: Same as "var".
use_locking: If True, updating of the var and accum tensors will be protected by
a lock; otherwise the behavior is undefined, but may exhibit less contention.
)doc");

REGISTER_OP("SparseApplyAdagradDA")
    .Input("var: Ref(T)")
    .Input("gradient_accumulator: Ref(T)")
    .Input("gradient_squared_accumulator: Ref(T)")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("global_step: int64")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdagradDAShapeFn(c, true /* sparse */);
    })
    .Doc(R"doc(
Update entries in '*var' and '*accum' according to the proximal adagrad scheme.

var: Should be from a Variable().
gradient_accumulator: Should be from a Variable().
gradient_squared_accumulator: Should be from a Variable().
grad: The gradient.
indices: A vector of indices into the first dimension of var and accum.
lr: Learning rate. Must be a scalar.
l1: L1 regularization. Must be a scalar.
l2: L2 regularization. Must be a scalar.
global_step: Training step number. Must be a scalar.
out: Same as "var".
use_locking: If True, updating of the var and accum tensors will be protected by
a lock; otherwise the behavior is undefined, but may exhibit less contention.
)doc");

REGISTER_OP("ResourceApplyAdagradDA")
    .Input("var: resource")
    .Input("gradient_accumulator: resource")
    .Input("gradient_squared_accumulator: resource")
    .Input("grad: T")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("global_step: int64")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdagradDAShapeFn(c, false /* sparse */);
    })
    .Doc(R"doc(
Update '*var' according to the proximal adagrad scheme.

var: Should be from a Variable().
gradient_accumulator: Should be from a Variable().
gradient_squared_accumulator: Should be from a Variable().
grad: The gradient.
lr: Scaling factor. Must be a scalar.
l1: L1 regularization. Must be a scalar.
l2: L2 regularization. Must be a scalar.
global_step: Training step number. Must be a scalar.
use_locking: If True, updating of the var and accum tensors will be protected by
a lock; otherwise the behavior is undefined, but may exhibit less contention.
)doc");

REGISTER_OP("ResourceSparseApplyAdagradDA")
    .Input("var: resource")
    .Input("gradient_accumulator: resource")
    .Input("gradient_squared_accumulator: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("global_step: int64")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdagradDAShapeFn(c, true /* sparse */);
    })
    .Doc(R"doc(
Update entries in '*var' and '*accum' according to the proximal adagrad scheme.

var: Should be from a Variable().
gradient_accumulator: Should be from a Variable().
gradient_squared_accumulator: Should be from a Variable().
grad: The gradient.
indices: A vector of indices into the first dimension of var and accum.
lr: Learning rate. Must be a scalar.
l1: L1 regularization. Must be a scalar.
l2: L2 regularization. Must be a scalar.
global_step: Training step number. Must be a scalar.
use_locking: If True, updating of the var and accum tensors will be protected by
a lock; otherwise the behavior is undefined, but may exhibit less contention.
)doc");

}  // namespace tensorflow
