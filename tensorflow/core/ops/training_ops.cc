/* Copyright 2015 Google Inc. All Rights Reserved.

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

REGISTER_OP("ApplyGradientDescent")
    .Input("var: Ref(T)")
    .Input("alpha: T")
    .Input("delta: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Doc(R"doc(
Update '*var' by subtracting 'alpha' * 'delta' from it.

var: Should be from a Variable().
alpha: Scaling factor. Must be a scalar.
delta: The change.
out: Same as "var".
use_locking: If `True`, the subtraction will be protected by a lock;
  otherwise the behavior is undefined, but may exhibit less contention.
)doc");

REGISTER_OP("ApplyAdadelta")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("accum_update: Ref(T)")
    .Input("lr: T")
    .Input("rho: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Doc(R"doc(
Update '*var' according to the adadelta scheme.

accum = rho() * accum + (1 - rho()) * grad.square();
update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
update_accum = rho() * update_accum + (1 - rho()) * update.square();
var -= update;

var: Should be from a Variable().
accum: Should be from a Variable().
accum_update: Should be from a Variable().
lr: Scaling factor. Must be a scalar.
rho: Decay factor. Must be a scalar.
epsilon: Constant factor. Must be a scalar.
grad: The gradient.
out: Same as "var".
use_locking: If True, updating of the var, accum and update_accum tensors will be protected by
a lock; otherwise the behavior is undefined, but may exhibit less contention.
)doc");

REGISTER_OP("SparseApplyAdadelta")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("accum_update: Ref(T)")
    .Input("lr: T")
    .Input("rho: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Doc(R"doc(
var: Should be from a Variable().
accum_grad: Should be from a Variable().
accum_update:: Should be from a Variable().
lr: Learning rate. Must be a scalar.
rho: Decay factor. Must be a scalar.
epsilon: Constant factor. Must be a scalar.
grad: The gradient.
indices: A vector of indices into the first dimension of var and accum.
out: Same as "var".
use_locking: If True, updating of the var and accum tensors will be protected by
a lock; otherwise the behavior is undefined, but may exhibit less contention.
)doc");

REGISTER_OP("ApplyAdagrad")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Doc(R"doc(
Update '*var' according to the adagrad scheme.

accum += grad * grad
var -= lr * grad * (1 / sqrt(accum))

var: Should be from a Variable().
accum: Should be from a Variable().
lr: Scaling factor. Must be a scalar.
grad: The gradient.
out: Same as "var".
use_locking: If `True`, updating of the var and accum tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

REGISTER_OP("SparseApplyAdagrad")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Doc(R"doc(
Update relevant entries in '*var' and '*accum' according to the adagrad scheme.

That is for rows we have grad for, we update var and accum as follows:
accum += grad * grad
var -= lr * grad * (1 / sqrt(accum))

var: Should be from a Variable().
accum: Should be from a Variable().
lr: Learning rate. Must be a scalar.
grad: The gradient.
indices: A vector of indices into the first dimension of var and accum.
out: Same as "var".
use_locking: If `True`, updating of the var and accum tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

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
l1: Scaling factor. Must be a scalar.
l2: Scaling factor. Must be a scalar.
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
l1: Scaling factor. Must be a scalar.
l2: Scaling factor. Must be a scalar.
lr_power: Scaling factor. Must be a scalar.
out: Same as "var".
use_locking: If `True`, updating of the var and accum tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

REGISTER_OP("ApplyMomentum")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("grad: T")
    .Input("momentum: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Doc(R"doc(
Update '*var' according to the momentum scheme.

accum = accum * momentum + grad
var -= lr * accum

var: Should be from a Variable().
accum: Should be from a Variable().
lr: Scaling factor. Must be a scalar.
grad: The gradient.
momentum: Momentum. Must be a scalar.
out: Same as "var".
use_locking: If `True`, updating of the var and accum tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

REGISTER_OP("SparseApplyMomentum")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("momentum: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Doc(R"doc(
Update relevant entries in '*var' and '*accum' according to the momentum scheme.

That is for rows we have grad for, we update var and accum as follows:

accum = accum * momentum + grad
var -= lr * accum

var: Should be from a Variable().
accum: Should be from a Variable().
lr: Learning rate. Must be a scalar.
grad: The gradient.
indices: A vector of indices into the first dimension of var and accum.
momentum: Momentum. Must be a scalar.
out: Same as "var".
use_locking: If `True`, updating of the var and accum tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

REGISTER_OP("ApplyAdam")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Doc(R"doc(
Update '*var' according to the Adam algorithm.

lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)

var: Should be from a Variable().
m: Should be from a Variable().
v: Should be from a Variable().
beta1_power: Must be a scalar.
beta2_power: Must be a scalar.
lr: Scaling factor. Must be a scalar.
beta1: Momentum factor. Must be a scalar.
beta2: Momentum factor. Must be a scalar.
epsilon: Ridge term. Must be a scalar.
grad: The gradient.
out: Same as "var".
use_locking: If `True`, updating of the var, m, and v tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

REGISTER_OP("ApplyRMSProp")
    .Input("var: Ref(T)")
    .Input("ms: Ref(T)")
    .Input("mom: Ref(T)")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Doc(R"doc(
Update '*var' according to the RMSProp algorithm.

mean_square = decay * mean_square + (1-decay) * gradient ** 2
Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
var <- var - mom

var: Should be from a Variable().
ms: Should be from a Variable().
mom: Should be from a Variable().
lr: Scaling factor. Must be a scalar.
epsilon: Ridge term. Must be a scalar.
rho: Decay rate. Must be a scalar.
grad: The gradient.
out: Same as "var".
use_locking: If `True`, updating of the var, m, and v tensors will be protected
  by a lock; otherwise the behavior is undefined, but may exhibit less
  contention.
)doc");

}  // namespace tensorflow
