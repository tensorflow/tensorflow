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

static Status ApplyFtrlShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // linear
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 3 /* grad_idx */, &s));
  int idx = sparse ? 5 : 4;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr_power
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyFtrl")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("linear: Ref(T)")
    .Input("grad: T")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("lr_power: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyFtrlShapeFn(c, false /* sparse */);
    })
    .Doc(R"doc(
Update '*var' according to the Ftrl-proximal scheme.

accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

var: Should be from a Variable().
accum: Should be from a Variable().
linear: Should be from a Variable().
grad: The gradient.
lr: Scaling factor. Must be a scalar.
l1: L1 regulariation. Must be a scalar.
l2: L2 regulariation. Must be a scalar.
lr_power: Scaling factor. Must be a scalar.
out: Same as "var".
use_locking: If `True`, updating of the var and accum tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

REGISTER_OP("SparseApplyFtrl")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("linear: Ref(T)")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("lr_power: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyFtrlShapeFn(c, true /* sparse */);
    })
    .Doc(R"doc(
Update relevant entries in '*var' according to the Ftrl-proximal scheme.

That is for rows we have grad for, we update var, accum and linear as follows:
accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

var: Should be from a Variable().
accum: Should be from a Variable().
linear: Should be from a Variable().
grad: The gradient.
indices: A vector of indices into the first dimension of var and accum.
lr: Scaling factor. Must be a scalar.
l1: L1 regularization. Must be a scalar.
l2: L2 regularization. Must be a scalar.
lr_power: Scaling factor. Must be a scalar.
out: Same as "var".
use_locking: If `True`, updating of the var and accum tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

REGISTER_OP("ResourceApplyFtrl")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("lr_power: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyFtrlShapeFn(c, false /* sparse */);
    })
    .Doc(R"doc(
Update '*var' according to the Ftrl-proximal scheme.

accum_new = accum + grad * grad
linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

var: Should be from a Variable().
accum: Should be from a Variable().
linear: Should be from a Variable().
grad: The gradient.
lr: Scaling factor. Must be a scalar.
l1: L1 regulariation. Must be a scalar.
l2: L2 regulariation. Must be a scalar.
lr_power: Scaling factor. Must be a scalar.
use_locking: If `True`, updating of the var and accum tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

REGISTER_OP("ResourceSparseApplyFtrl")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("lr_power: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyFtrlShapeFn(c, true /* sparse */);
    })
    .Doc(R"doc(
Update relevant entries in '*var' according to the Ftrl-proximal scheme.

That is for rows we have grad for, we update var, accum and linear as follows:
accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

var: Should be from a Variable().
accum: Should be from a Variable().
linear: Should be from a Variable().
grad: The gradient.
indices: A vector of indices into the first dimension of var and accum.
lr: Scaling factor. Must be a scalar.
l1: L1 regularization. Must be a scalar.
l2: L2 regularization. Must be a scalar.
lr_power: Scaling factor. Must be a scalar.
use_locking: If `True`, updating of the var and accum tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

REGISTER_OP("ApplyFtrlV2")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("linear: Ref(T)")
    .Input("grad: T")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l2_shrinkage: T")
    .Input("lr_power: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyFtrlShapeFn(c, false /* sparse */);
    })
    .Doc(R"doc(
Update '*var' according to the Ftrl-proximal scheme.

grad_with_shrinkage = grad + 2 * l2_shrinkage * var
accum_new = accum + grad_with_shrinkage * grad_with_shrinkage
linear += grad_with_shrinkage +
    (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

var: Should be from a Variable().
accum: Should be from a Variable().
linear: Should be from a Variable().
grad: The gradient.
lr: Scaling factor. Must be a scalar.
l1: L1 regulariation. Must be a scalar.
l2: online L2 regulariation. Must be a scalar.
l2: L2 shrinkage regulariation. Must be a scalar.
lr_power: Scaling factor. Must be a scalar.
out: Same as "var".
use_locking: If `True`, updating of the var and accum tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

REGISTER_OP("SparseApplyFtrlV2")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("linear: Ref(T)")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l2_shrinkage: T")
    .Input("lr_power: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyFtrlShapeFn(c, true /* sparse */);
    })
    .Doc(R"doc(
Update relevant entries in '*var' according to the Ftrl-proximal scheme.

That is for rows we have grad for, we update var, accum and linear as follows:
grad_with_shrinkage = grad + 2 * l2_shrinkage * var
accum_new = accum + grad_with_shrinkage * grad_with_shrinkage
linear += grad_with_shrinkage +
    (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

var: Should be from a Variable().
accum: Should be from a Variable().
linear: Should be from a Variable().
grad: The gradient.
indices: A vector of indices into the first dimension of var and accum.
lr: Scaling factor. Must be a scalar.
l1: L1 regularization. Must be a scalar.
l2: onine L2 regularization. Must be a scalar.
l2: L2 shrinkage regulariation. Must be a scalar.
lr_power: Scaling factor. Must be a scalar.
out: Same as "var".
use_locking: If `True`, updating of the var and accum tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

REGISTER_OP("ResourceApplyFtrlV2")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l2_shrinkage: T")
    .Input("lr_power: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyFtrlShapeFn(c, false /* sparse */);
    })
    .Doc(R"doc(
Update '*var' according to the Ftrl-proximal scheme.

grad_with_shrinkage = grad + 2 * l2_shrinkage * var
accum_new = accum + grad_with_shrinkage * grad_with_shrinkage
linear += grad_with_shrinkage +
    (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

var: Should be from a Variable().
accum: Should be from a Variable().
linear: Should be from a Variable().
grad: The gradient.
lr: Scaling factor. Must be a scalar.
l1: L1 regulariation. Must be a scalar.
l2: onine L2 regularization. Must be a scalar.
l2: L2 shrinkage regulariation. Must be a scalar.
lr_power: Scaling factor. Must be a scalar.
use_locking: If `True`, updating of the var and accum tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

REGISTER_OP("ResourceSparseApplyFtrlV2")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l2_shrinkage: T")
    .Input("lr_power: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyFtrlShapeFn(c, true /* sparse */);
    })
    .Doc(R"doc(
Update relevant entries in '*var' according to the Ftrl-proximal scheme.

That is for rows we have grad for, we update var, accum and linear as follows:
grad_with_shrinkage = grad + 2 * l2_shrinkage * var
accum_new = accum + grad_with_shrinkage * grad_with_shrinkage
linear += grad_with_shrinkage +
    (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new

var: Should be from a Variable().
accum: Should be from a Variable().
linear: Should be from a Variable().
grad: The gradient.
indices: A vector of indices into the first dimension of var and accum.
lr: Scaling factor. Must be a scalar.
l1: L1 regularization. Must be a scalar.
l2: onine L2 regularization. Must be a scalar.
l2: L2 shrinkage regulariation. Must be a scalar.
lr_power: Scaling factor. Must be a scalar.
use_locking: If `True`, updating of the var and accum tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

}  // namespace tensorflow
