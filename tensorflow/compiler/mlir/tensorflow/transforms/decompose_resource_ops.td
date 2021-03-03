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
include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

// Here, the element type can be any integer or float type. But, note that only
// 32 bit integers are supported for the values.
class GetScalarOfType<int value> : NativeCodeCall<
  "GetScalarOfType(getElementTypeOrSelf($0)," # value # ")">;

def NonScalarType : Type<Neg<HasAnyRankOfPred<[0]>>, "Non scalar type">;

// Creates a tf.ReadVariable op that reads a resource `$2` that has the same
// element type as `$1`. The op created will use location of `$0`.
def CreateTFReadVariableOp : NativeCodeCall<
    "$_builder.create<TF::ReadVariableOp>("
    "  $0.getLoc(),"
    "  GetResourceSubtypeOrDefault("
    "    $2, $1.getType().cast<TensorType>().getElementType()),"
    "  $2)"
    >;

def CheckHasResourceSubtype : Constraint<CPred<"HasResourceSubtype($0)">>;

def CreateTFReadVariableOpFromResourceHandle : NativeCodeCall<
    "$_builder.create<TF::ReadVariableOp>("
    "$0.getLoc(), GetResourceSubtype($1), $1)">;

def DecomposeAssignAddVariableOp :
  Pat<
    (TF_AssignAddVariableOp:$src_op $resource, $value),
    (TF_AssignVariableOp
      $resource,
      (TF_AddV2Op
        (CreateTFReadVariableOp $src_op, $value, $resource),
        $value
      )
    )
  >;

def DecomposeAssignSubVariableOp :
  Pat<
    (TF_AssignSubVariableOp:$src_op $resource, $value),
    (TF_AssignVariableOp
      $resource,
      (TF_SubOp
        (CreateTFReadVariableOp $src_op, $value, $resource),
        $value
      )
    )
  >;

// This decomposition is only correct inside XLA as it ignores use_locking
// attribute.
def DecomposeResourceApplyGradientDescentOp :
  Pat<
    (TF_ResourceApplyGradientDescentOp:$src_op $resource, $alpha, $delta, BoolAttr:$_),
    (TF_AssignVariableOp
      $resource,
      (TF_SubOp
        (CreateTFReadVariableOp $src_op, $alpha, $resource),
        (TF_MulOp $alpha, $delta)
      )
    )
  >;

// This decomposition is only correct inside XLA as it ignores use_locking
// attribute.
// accum = accum * momentum + grad;
// var -= accum * lr
def DecomposeResourceApplyMomentumOpNonNesterov :
  Pattern<
    (TF_ResourceApplyMomentumOp:$src_op
       $var_resource, $accum_resource, $lr, $grad, $momentum,
       BoolAttr:$_, ConstBoolAttrFalse:$use_nesterov
    ),
    [(TF_AddV2Op:$accum_new
      (TF_MulOp
        (CreateTFReadVariableOp $src_op, $grad, $accum_resource),
        $momentum
      ),
      $grad
     ),
     (TF_AssignVariableOp $accum_resource, $accum_new),
     (TF_AssignSubVariableOp $var_resource, (TF_MulOp $accum_new, $lr))
    ]
  >;

// This decomposition is only correct inside XLA as it ignores use_locking
// attribute.
// accum = accum * momentum + grad;
// var -= grad * lr + accum * momentum * lr
def DecomposeResourceApplyMomentumOpNesterov :
  Pattern<
    (TF_ResourceApplyMomentumOp:$src_op
       $var_resource, $accum_resource, $lr, $grad, $momentum,
       BoolAttr:$_, ConstBoolAttrTrue:$use_nesterov
    ),
    [(TF_AddV2Op:$accum_new
       (TF_MulOp
         (CreateTFReadVariableOp $src_op, $grad, $accum_resource),
         $momentum
       ),
       $grad
     ),
     (TF_AssignVariableOp $accum_resource, $accum_new),
     (TF_AssignSubVariableOp
       $var_resource,
       (TF_AddV2Op
         (TF_MulOp $grad, $lr),
         (TF_MulOp $accum_new, (TF_MulOp $momentum, $lr))
       )
     )
    ]
  >;

// This decomposition is only correct inside XLA as it ignores use_locking
// attribute.
// accum = accum * momentum - lr * grad
// var += accum
def DecomposeResourceApplyKerasMomentumOpNonNesterov :
  Pattern<
    (TF_ResourceApplyKerasMomentumOp:$src_op
       $var_resource, $accum_resource, $lr, $grad, $momentum,
       BoolAttr:$_, ConstBoolAttrFalse:$use_nesterov
    ),
    [(TF_SubOp:$accum_new
       (TF_MulOp
         (CreateTFReadVariableOp $src_op, $grad, $accum_resource),
         $momentum
       ),
       (TF_MulOp $grad, $lr)
     ),
     (TF_AssignVariableOp $accum_resource, $accum_new),
     (TF_AssignAddVariableOp $var_resource, $accum_new)
    ]
  >;

// This decomposition is only correct inside XLA as it ignores use_locking
// attribute.
// accum = accum * momentum - lr * grad
// var += accum * momentum - lr * grad
def DecomposeResourceApplyKerasMomentumOpNesterov :
  Pattern<
    (TF_ResourceApplyKerasMomentumOp:$src_op
      $var_resource, $accum_resource, $lr, $grad, $momentum,
      BoolAttr:$_, ConstBoolAttrTrue:$use_nesterov
    ),
    [(TF_SubOp:$accum_new
       (TF_MulOp
         (CreateTFReadVariableOp $src_op, $grad, $accum_resource),
         $momentum
       ),
       (TF_MulOp:$grad_lr $grad, $lr)
     ),
     (TF_AssignVariableOp $accum_resource, $accum_new),
     (TF_AssignAddVariableOp
       $var_resource,
       (TF_SubOp
         (TF_MulOp $accum_new, $momentum),
         $grad_lr
       )
     )
    ]
  >;

// Pattern to Decompose ResourceApplyAdagradV2.
// This decomposition is only correct inside XLA as it ignores use_locking
// attribute.
// accum <- accum + grad * grad
// variable <- variable - lr * grad / (sqrt(accum) + epsilon)
def DecomposeResourceApplyAdagradV2 :
  Pattern<
    (TF_ResourceApplyAdagradV2Op:$src_op
      $var_resource, $accum_resource, $lr, $epsilon, $grad, BoolAttr:$_,
      ConstBoolAttrTrue:$update_slots),
    [
      (TF_AddV2Op:$new_accum
        (CreateTFReadVariableOp $src_op, $grad, $accum_resource),
        (TF_MulOp $grad, $grad)
      ),
      (TF_AssignSubVariableOp
        $var_resource,
        (TF_DivOp
          (TF_MulOp $lr, $grad),
          (TF_AddV2Op (TF_SqrtOp $new_accum), $epsilon)
        )
      ),
      (TF_AssignVariableOp $accum_resource, $new_accum),
    ]
  >;

// ResourceApplyAdagrad op can be canonicalized to ResourceApplyAdagradV2 with
// zero epsilon and then decomposed using DecomposeResourceApplyAdagradV2
// pattern.
def DecomposeResourceApplyAdagrad :
  Pattern<
    (TF_ResourceApplyAdagradOp $var_resource, $accum_resource, $lr, $grad,
       $use_locking, $update_slots),
    [
      (TF_ConstOp:$zero_epsilon (GetScalarOfType<0> $grad)),
      (TF_ResourceApplyAdagradV2Op $var_resource, $accum_resource, $lr,
          $zero_epsilon, $grad, $use_locking, $update_slots
      )
    ]>;


// Pattern to Decompose ResourceApplyAdam without Nesterov momentum.
// This decomposition is only correct inside XLA as it ignores use_locking
// attribute.
// alpha <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
// m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
// v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
// variable <- variable - alpha * m_t / (sqrt(v_t) + epsilon)
def DecomposeResourceApplyAdamNonNesterov :
  Pattern<
    (TF_ResourceApplyAdamOp:$src_op
      $var_resource, $m_resource, $v_resource, $beta1_power, $beta2_power, $lr,
      $beta1, $beta2, $epsilon, $grad, BoolAttr:$_,
      ConstBoolAttrFalse:$use_nesterov),
    [
      (TF_ConstOp:$one (GetScalarOfType<1> $grad)),
      (TF_MulOp:$alpha
        $lr,
        (TF_DivOp
          (TF_SqrtOp (TF_SubOp $one, $beta2_power)),
          (TF_SubOp $one, $beta1_power)
        )
      ),
      (TF_AddV2Op:$new_m
        (TF_MulOp $beta1, (CreateTFReadVariableOp $src_op, $grad, $m_resource)),
        (TF_MulOp (TF_SubOp $one, $beta1), $grad)
      ),
      (TF_AddV2Op:$new_v
        (TF_MulOp $beta2, (CreateTFReadVariableOp $src_op, $grad, $v_resource)),
        (TF_MulOp
          (TF_SubOp $one, $beta2),
          (TF_SquareOp $grad)
        )
      ),
      (TF_AssignSubVariableOp
        $var_resource,
        (TF_DivOp
          (TF_MulOp $alpha, $new_m),
          (TF_AddV2Op (TF_SqrtOp $new_v), $epsilon)
        )
      ),
      (TF_AssignVariableOp $m_resource, $new_m),
      (TF_AssignVariableOp $v_resource, $new_v)
    ]
  >;

// Pattern to Decompose ResourceApplyAdam with Nesterov momentum.
// This decomposition is only correct inside XLA as it ignores use_locking
// attribute.
// alpha <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
// m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
// v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
// variable <- variable - (alpha * (m_t * beta1 + (1 - beta1) * g_t) /
//                        (sqrt(v_t) + epsilon))
def DecomposeResourceApplyAdamNesterov :
  Pattern<
    (TF_ResourceApplyAdamOp:$src_op
      $var_resource, $m_resource, $v_resource, $beta1_power, $beta2_power, $lr,
      $beta1, $beta2, $epsilon, $grad, BoolAttr:$_,
      ConstBoolAttrTrue:$use_nesterov),
    [
      (TF_ConstOp:$one (GetScalarOfType<1> $grad)),
      (TF_MulOp:$alpha
        $lr,
        (TF_DivOp
          (TF_SqrtOp (TF_SubOp $one, $beta2_power)),
          (TF_SubOp $one, $beta1_power)
        )
      ),
      (TF_AddV2Op:$new_m
        (TF_MulOp $beta1, (CreateTFReadVariableOp $src_op, $grad, $m_resource)),
        (TF_MulOp (TF_SubOp $one, $beta1), $grad)
      ),
      (TF_AddV2Op:$new_v
        (TF_MulOp $beta2, (CreateTFReadVariableOp $src_op, $grad, $v_resource)),
        (TF_MulOp
          (TF_SubOp $one, $beta2),
          (TF_SquareOp $grad)
        )
      ),
      (TF_AssignSubVariableOp
        $var_resource,
        (TF_DivOp
          (TF_MulOp
            $alpha,
            (TF_AddV2Op
              (TF_MulOp $new_m, $beta1),
              (TF_MulOp (TF_SubOp $one, $beta1), $grad)
            )
          ),
          (TF_AddV2Op (TF_SqrtOp $new_v), $epsilon)
        )
      ),
      (TF_AssignVariableOp $m_resource, $new_m),
      (TF_AssignVariableOp $v_resource, $new_v)
    ]
  >;

// Pattern to decompose tf.ResourceGather into tf.ReadVariable and tf.GatherV2.
def DecomposeResourceGather : Pat<
  (TF_ResourceGatherOp:$old_result
    $resource, $indices, $batch_dims, $validate_indices),
  (TF_GatherV2Op
    (CreateTFReadVariableOp $old_result, $old_result, $resource),
    $indices,
    (TF_ConstOp $batch_dims), // axis
    $batch_dims
  )>;

// Pattern to decompose tf.ResourceScatterUpdate into tf.ReadVariable,
// tf.TensorScatterUpdate, and tf.AssignVariable.
// TODO(b/178725329): Support scalar `updates` during decomposition.
def DecomposeResourceScatterUpdate : Pat<
  (TF_ResourceScatterUpdateOp:$src_op
    $resource, $indices, NonScalarType:$updates),
  (TF_AssignVariableOp
    $resource,
    (TF_TensorScatterUpdateOp
      (CreateTFReadVariableOp $src_op, $updates, $resource),
      $indices,
      $updates
    )
  )>;

// Pattern to decompose tf.VariableShape into tf.ReadVariable and tf.Shape.
def DecomposeVariableShape : Pat<
  (TF_VariableShapeOp:$src_op $resource),
  (TF_ShapeOp (CreateTFReadVariableOpFromResourceHandle $src_op, $resource)),
  [(CheckHasResourceSubtype $resource)]>;

// This decomposition is only correct inside XLA as it ignores use_locking
// attribute.
// ms <- rho * ms_{t-1} + (1-rho) * grad * grad
// mg = grad * (one - rho) + mg * rho;
// mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
//
def DecomposeResourceApplyCenteredRMSProp :
  Pattern<
    (TF_ResourceApplyCenteredRMSPropOp:$src_op
       $var_resource, $mg_resource, $ms_resource, $mom_resource, $lr, $rho, $momentum, $epsilon,
       $grad, $use_locking
    ),
    [(TF_ConstOp:$one (GetScalarOfType<1> $grad)),
     (CreateTFReadVariableOp $src_op, $grad, $ms_resource),
     (TF_AddV2Op:$ms_new
       (TF_MulOp
         (TF_MulOp $grad, $grad),
         (TF_SubOp $one, $rho)
       ),
       (TF_MulOp
          (CreateTFReadVariableOp $src_op, $grad, $ms_resource),
          $rho
       )
     ),
     (TF_AssignVariableOp $ms_resource, $ms_new),
     // mg = grad * (one - rho) + mg * rho;
     (TF_AddV2Op:$mg_new
       (TF_MulOp
         $grad,
         (TF_SubOp $one, $rho)
       ),
       (TF_MulOp
          (CreateTFReadVariableOp $src_op, $grad, $mg_resource),
          $rho
       )
     ),
     (TF_AssignVariableOp $mg_resource, $mg_new),
     // mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
     (TF_AddV2Op:$mom_new
      (TF_MulOp $momentum,
       (CreateTFReadVariableOp $src_op, $grad, $mom_resource)),
      (TF_DivOp
         (TF_MulOp $lr, $grad),
         (TF_SqrtOp
           (TF_SubOp
             $ms_new,
             (TF_AddV2Op
               (TF_MulOp
                 $mg_new,
                 $mg_new
               ),
               $epsilon
             )
           )
         )
      )
     ),
     (TF_AssignVariableOp $mom_resource, $mom_new),
     // var <- var - mom
     (TF_AssignSubVariableOp $var_resource, $mom_new)
   ]
   >;

// This decomposition is only correct inside XLA as it ignores use_locking
// attribute.
// ms <- rho * ms_{t-1} + (1-rho) * grad * grad
// mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
// var <- var - mom
def DecomposeResourceApplyRMSProp :
  Pattern<
    (TF_ResourceApplyRMSPropOp:$src_op
       $var_resource, $ms_resource, $mom_resource, $lr, $rho, $momentum, $epsilon,
       $grad, $use_locking
    ),
    [(TF_ConstOp:$one (GetScalarOfType<1> $grad)),
     (CreateTFReadVariableOp $src_op, $grad, $ms_resource),
     // ms <- rho * ms_{t-1} + (1-rho) * grad * grad
     (TF_AddV2Op:$ms_new
       (TF_MulOp
          (CreateTFReadVariableOp $src_op, $grad, $ms_resource),
          $rho
       ),
       (TF_MulOp
         (TF_SquareOp $grad),
         (TF_SubOp $one, $rho)
       )
     ),
     (TF_AssignVariableOp $ms_resource, $ms_new),
     // mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
     (TF_AddV2Op:$mom_new
      (TF_MulOp $momentum,
       (CreateTFReadVariableOp $src_op, $grad, $mom_resource)),
      (TF_DivOp
         (TF_MulOp $lr, $grad),
         (TF_SqrtOp
           (TF_AddV2Op $ms_new, $epsilon)
         )
      )
     ),
     (TF_AssignVariableOp $mom_resource, $mom_new),
     // var <- var - mom
     (TF_AssignSubVariableOp $var_resource, $mom_new)
   ]
   >;

def DecomposeResourceApplyProximalAdagrad :
  Pattern<
    (TF_ResourceApplyProximalAdagradOp:$src_op
       $var_resource, $accum_resource, $lr, $l1, $l2, $grad,
       $use_locking
    ),
    [(TF_ConstOp:$one (GetScalarOfType<1> $grad)),
     (TF_ConstOp:$zero (GetScalarOfType<0> $grad)),
     (TF_AddV2Op:$accum_new
       (CreateTFReadVariableOp $src_op, $grad, $accum_resource),
       (TF_SquareOp $grad)),
     (TF_MulOp:$adagrad_lr $lr, (TF_RsqrtOp $accum_new)),
     (TF_SubOp:$prox_var
       (CreateTFReadVariableOp $src_op, $grad, $var_resource),
       (TF_MulOp $grad, $adagrad_lr)),
     (TF_MulOp:$l1_gt_zero (TF_SignOp $prox_var),
       (TF_MaximumOp
          (TF_SubOp (TF_AbsOp $prox_var), (TF_MulOp $adagrad_lr, $l1)), $zero)),
     (TF_SelectV2Op:$var_numerator (TF_GreaterOp $l1, $zero),
       $l1_gt_zero, $prox_var),
     (TF_DivOp:$var_new
       $var_numerator, (TF_AddOp $one, (TF_MulOp $adagrad_lr, $l2))),
     (TF_AssignVariableOp $var_resource, $var_new),
     (TF_AssignVariableOp $accum_resource, $accum_new)
   ]
   >;
