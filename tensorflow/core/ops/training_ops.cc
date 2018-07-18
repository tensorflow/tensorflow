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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

static ShapeHandle ShapeOrHandleShape(InferenceContext* c, int input) {
  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

// Handle the gradient and, if <sparse>, indices inputs.
// <s> is an input+output parameter, containing the current known input shape to
// the gradient.
static Status HandleGradAndIndicesInputs(InferenceContext* c, bool sparse,
                                         int grad_idx, ShapeHandle* s) {
  ShapeHandle grad = ShapeOrHandleShape(c, grad_idx);
  if (!sparse) {
    TF_RETURN_IF_ERROR(c->Merge(*s, grad, s));
    return Status::OK();
  }
  // Indices is a vector where indices.dim[0].rank == grad[0].rank.
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(grad_idx + 1), 1, &indices));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused));

  // Trailing part of grad matches trailing part of *s.
  ShapeHandle grad_unknown_first;
  TF_RETURN_IF_ERROR(
      c->ReplaceDim(grad, 0, c->UnknownDim(), &grad_unknown_first));
  TF_RETURN_IF_ERROR(c->Merge(*s, grad_unknown_first, s));

  return Status::OK();
}

static Status ApplyGradientDescentShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                  // var
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));  // alpha
  TF_RETURN_IF_ERROR(c->Merge(s, c->input(2), &s));          // delta
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyGradientDescent")
    .Input("var: Ref(T)")
    .Input("alpha: T")
    .Input("delta: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyGradientDescentShapeFn);

REGISTER_OP("ResourceApplyGradientDescent")
    .Input("var: resource")
    .Input("alpha: T")
    .Input("delta: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyGradientDescentShapeFn);

static Status ApplyProximalGradientDescentShapeFn(InferenceContext* c,
                                                  bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                  // var
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));  // alpha
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 4 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyProximalGradientDescent")
    .Input("var: Ref(T)")
    .Input("alpha: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("delta: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyProximalGradientDescentShapeFn(c, false /* sparse */);
    });

REGISTER_OP("SparseApplyProximalGradientDescent")
    .Input("var: Ref(T)")
    .Input("alpha: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyProximalGradientDescentShapeFn(c, true /* sparse */);
    });

REGISTER_OP("ResourceApplyProximalGradientDescent")
    .Input("var: resource")
    .Input("alpha: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("delta: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyProximalGradientDescentShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceSparseApplyProximalGradientDescent")
    .Input("var: resource")
    .Input("alpha: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyProximalGradientDescentShapeFn(c, true /* sparse */);
    });

static Status ApplyAdadeltaShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape(c, 2), &s));            // accum update
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));  // rho
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 6 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

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
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdadeltaShapeFn(c, false /* sparse */);
    });

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
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdadeltaShapeFn(c, true /* sparse */);
    });

REGISTER_OP("ResourceApplyAdadelta")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("accum_update: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdadeltaShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceSparseApplyAdadelta")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("accum_update: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdadeltaShapeFn(c, true /* sparse */);
    });

static Status ApplyAdagradShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 3 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyAdagrad")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdagradShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceApplyAdagrad")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdagradShapeFn(c, false /* sparse */);
    });

static Status ApplyProximalAdagradShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // l2
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 5 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyProximalAdagrad")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyProximalAdagradShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceApplyProximalAdagrad")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyProximalAdagradShapeFn(c, false /* sparse */);
    });

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
    .Attr("update_slots: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdagradShapeFn(c, true /* sparse */);
    });

REGISTER_OP("ResourceSparseApplyAdagrad")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdagradShapeFn(c, true /* sparse */);
    });

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
    });

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
    });

REGISTER_OP("SparseApplyProximalAdagrad")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyProximalAdagradShapeFn(c, true /* sparse */);
    });

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
    });

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
    });

REGISTER_OP("ResourceSparseApplyProximalAdagrad")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyProximalAdagradShapeFn(c, true /* sparse */);
    });

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
    });

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
    });

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
    });

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
    });

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
    });

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
    });

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
    });

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
    });

static Status ApplyMomentumShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 3 /* grad_idx */, &s));
  int idx = sparse ? 5 : 4;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // momentum
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyMomentum")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("grad: T")
    .Input("momentum: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyMomentumShapeFn(c, false /* sparse */);
    });

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
    .Attr("use_nesterov: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyMomentumShapeFn(c, true /* sparse */);
    });

REGISTER_OP("ResourceApplyMomentum")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("grad: T")
    .Input("momentum: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyMomentumShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceSparseApplyMomentum")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("momentum: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyMomentumShapeFn(c, true /* sparse */);
    });

static Status ApplyAdamShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));       // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));       // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));       // epsilon
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 9 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

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
    .Attr("use_nesterov: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdamShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceApplyAdam")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdamShapeFn(c, false /* sparse */);
    });

static Status ApplyAdaMaxShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));       // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));       // epsilon
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 8 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyAdaMax")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("beta1_power: T")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaMaxShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceApplyAdaMax")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("beta1_power: T")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdaMaxShapeFn(c, false /* sparse */);
    });

static Status ApplyRMSPropShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // ms
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // mom
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // rho
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // momentum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));       // epsilon
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 7 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

static Status ApplyCenteredRMSPropShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // ms
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // mg
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 3), &s));  // mom
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // rho
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));       // momentum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));       // epsilon
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 8 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

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
    .SetShapeFn([](InferenceContext* c) {
      return ApplyRMSPropShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ApplyCenteredRMSProp")
    .Input("var: Ref(T)")
    .Input("mg: Ref(T)")
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
    .SetShapeFn([](InferenceContext* c) {
      return ApplyCenteredRMSPropShapeFn(c, false /* sparse */);
    });

REGISTER_OP("SparseApplyRMSProp")
    .Input("var: Ref(T)")
    .Input("ms: Ref(T)")
    .Input("mom: Ref(T)")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyRMSPropShapeFn(c, true /* sparse */);
    });

REGISTER_OP("SparseApplyCenteredRMSProp")
    .Input("var: Ref(T)")
    .Input("mg: Ref(T)")
    .Input("ms: Ref(T)")
    .Input("mom: Ref(T)")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyCenteredRMSPropShapeFn(c, true /* sparse */);
    });

REGISTER_OP("ResourceApplyRMSProp")
    .Input("var: resource")
    .Input("ms: resource")
    .Input("mom: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyRMSPropShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceApplyCenteredRMSProp")
    .Input("var: resource")
    .Input("mg: resource")
    .Input("ms: resource")
    .Input("mom: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyCenteredRMSPropShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceSparseApplyRMSProp")
    .Input("var: resource")
    .Input("ms: resource")
    .Input("mom: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyRMSPropShapeFn(c, true /* sparse */);
    });

REGISTER_OP("ResourceSparseApplyCenteredRMSProp")
    .Input("var: resource")
    .Input("mg: resource")
    .Input("ms: resource")
    .Input("mom: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyCenteredRMSPropShapeFn(c, true /* sparse */);
    });

static Status ApplyAddSignShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // alpha
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // sign_decay
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // beta
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 6 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyAddSign")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("lr: T")
    .Input("alpha: T")
    .Input("sign_decay: T")
    .Input("beta: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAddSignShapeFn(c, /*sparse=*/false);
    });

REGISTER_OP("ResourceApplyAddSign")
    .Input("var: resource")
    .Input("m: resource")
    .Input("lr: T")
    .Input("alpha: T")
    .Input("sign_decay: T")
    .Input("beta: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAddSignShapeFn(c, /*sparse=*/false);
    });

static Status ApplyPowerSignShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // logbase
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // sign_delay
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // beta
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 6 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyPowerSign")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("lr: T")
    .Input("logbase: T")
    .Input("sign_decay: T")
    .Input("beta: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyPowerSignShapeFn(c, /*sparse=*/false);
    });

REGISTER_OP("ResourceApplyPowerSign")
    .Input("var: resource")
    .Input("m: resource")
    .Input("lr: T")
    .Input("logbase: T")
    .Input("sign_decay: T")
    .Input("beta: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyPowerSignShapeFn(c, /*sparse=*/false);
    });

static Status ApplyRpropShapeFn(InferenceContext* c, bool withError) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // old_grad
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // delta
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // scalar
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // scalar
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // scalar
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));       // scalar
  // for iRprop+ which uses the error (loss) value
  if (withError) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));  // scalar
    TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));  // scalar
    TF_RETURN_IF_ERROR(c->Merge(s, c->input(9), &s));          // grad (delta)
  } else {
    TF_RETURN_IF_ERROR(c->Merge(s, c->input(7), &s));  // grad (delta)
  }

  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyIRpropPlus")
    .Input("var: Ref(T)")
    .Input("old_grad: Ref(T)")
    .Input("delta_update: Ref(T)")
    .Input("eta_minus: T")
    .Input("eta_plus: T")
    .Input("delta_min: T")
    .Input("delta_max: T")
    .Input("error: T")
    .Input("old_error: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyRpropShapeFn(c, /*error=*/true);
    });

REGISTER_OP("ResourceApplyIRpropPlus")
    .Input("var: resource")
    .Input("old_grad: resource")
    .Input("delta_update: resource")
    .Input("eta_minus: T")
    .Input("eta_plus: T")
    .Input("delta_min: T")
    .Input("delta_max: T")
    .Input("error: T")
    .Input("old_error: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyRpropShapeFn(c, /*error=*/true);
    });

REGISTER_OP("ApplyRpropMinus")
    .Input("var: Ref(T)")
    .Input("old_grad: Ref(T)")
    .Input("delta_update: Ref(T)")
    .Input("eta_minus: T")
    .Input("eta_plus: T")
    .Input("delta_min: T")
    .Input("delta_max: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyRpropShapeFn(c, /*error=*/false);
    });

REGISTER_OP("ResourceApplyRpropMinus")
    .Input("var: resource")
    .Input("old_grad: resource")
    .Input("delta_update: resource")
    .Input("eta_minus: T")
    .Input("eta_plus: T")
    .Input("delta_min: T")
    .Input("delta_max: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyRpropShapeFn(c, /*error=*/false);
    });

}  // namespace tensorflow
