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
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/tensorflow/transforms/rewrite_util.td"

def NonScalarRankedType : Type<And<[Neg<HasAnyRankOfPred<[0]>>, HasRankPred]>,
                         "Non scalar type">;

def HasSingleElement: Constraint<CPred<"($0.size() == 1)">>;

def EmptyList: NativeCodeCall<"llvm::SmallVector<mlir::Value>{}">;

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

def CreateConstBoolAttrFalse : NativeCodeCall<"$_builder.getBoolAttr(false)">;

def CreateTFReadVariableOpFromResourceHandle : NativeCodeCall<
    "$_builder.create<TF::ReadVariableOp>("
    "$0.getLoc(), GetResourceSubtype($1), $1)">;

def CreateTFSelectOp: NativeCodeCall<
    "$_builder.create<TF::SelectOp>("
    "$0.getLoc(), $3.getType(), $1, $2, $3)">;

def ConstAttrIfThenElse: NativeCodeCall<
    "($0).getValue() ? ($1) : ($2)">;

// Convert clamp(lo, x, hi) to clipbyvalue(x, lo, hi).
def Clamp: NativeCodeCall<
    "$_builder.create<TF::ClipByValueOp>("
    "  $0.getLoc(),"
    "  $2.getType(), $2, $1, $3)">;

def CopyXlaOutsideCompilationAttr: NativeCodeCallVoid<
    "CopyXlaOutsideCompilationAttributesAdaptor($0, $1)">;

def DecomposeAssignAddVariableOp :
  Pat<
    (TF_AssignAddVariableOp:$src_op $resource, $value),
    (TF_AssignVariableOp
      $resource,
      (TF_AddV2Op
        (CreateTFReadVariableOp $src_op, $value, $resource),
        $value
      ),
      (CreateConstBoolAttrFalse)
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
      ),
      (CreateConstBoolAttrFalse)
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
      ),
      (CreateConstBoolAttrFalse)
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
     (TF_AssignVariableOp $accum_resource, $accum_new, (CreateConstBoolAttrFalse)),
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
     (TF_AssignVariableOp $accum_resource, $accum_new, (CreateConstBoolAttrFalse)),
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
     (TF_AssignVariableOp $accum_resource, $accum_new, (CreateConstBoolAttrFalse)),
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
     (TF_AssignVariableOp $accum_resource, $accum_new, (CreateConstBoolAttrFalse)),
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
      (TF_AssignVariableOp $accum_resource, $new_accum, (CreateConstBoolAttrFalse)),
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
      (TF_AssignVariableOp $m_resource, $new_m, (CreateConstBoolAttrFalse)),
      (TF_AssignVariableOp $v_resource, $new_v, (CreateConstBoolAttrFalse))
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
      (TF_AssignVariableOp $m_resource, $new_m, (CreateConstBoolAttrFalse)),
      (TF_AssignVariableOp $v_resource, $new_v, (CreateConstBoolAttrFalse))
    ]
  >;

// Pattern to decompose tf.ResourceGather into tf.ReadVariable and tf.GatherV2.
def DecomposeResourceGather : Pat<
  (TF_ResourceGatherOp:$old_result
    $resource, $indices, $batch_dims, $validate_indices),
  (TF_GatherV2Op:$dest
    (CreateTFReadVariableOp $old_result, $old_result, $resource),
    $indices,
    (TF_ConstOp $batch_dims), // axis
    $batch_dims
  ), [], [(CopyXlaOutsideCompilationAttr $old_result, $dest)]>;

// Pattern to decompose tf.ResourceScatterAdd into tf.ReadVariable,
// tf.TensorScatterAdd, and tf.AssignVariable.
def DecomposeResourceScatterAdd : Pat<
  (TF_ResourceScatterAddOp:$src_op $resource, $indices, $updates),
  (TF_AssignVariableOp
    $resource,
    (TF_TensorScatterAddOp
      (CreateTFReadVariableOp $src_op, $updates, $resource),
      (TF_ExpandDimsOp $indices,
       (TF_ConstOp (GetScalarOfType<-1> $indices))),
      $updates
    ),
    (CreateConstBoolAttrFalse)
  )>;

// Pattern to decompose tf.ResourceScatterUpdate into tf.ReadVariable,
// tf.TensorScatterUpdate, and tf.AssignVariable.
def DecomposeResourceScatterUpdate : Pat<
  (TF_ResourceScatterUpdateOp:$src_op $resource, $indices, $updates),
  (TF_AssignVariableOp
    $resource,
    (TF_TensorScatterUpdateOp
      (CreateTFReadVariableOp $src_op, $updates, $resource),
      (TF_ExpandDimsOp $indices,
       (TF_ConstOp (GetScalarOfType<-1> $indices))),
      $updates
    ),
    (CreateConstBoolAttrFalse)
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
     (TF_AssignVariableOp $ms_resource, $ms_new, (CreateConstBoolAttrFalse)),
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
     (TF_AssignVariableOp $mg_resource, $mg_new, (CreateConstBoolAttrFalse)),
     // mom <- momentum * mom_{t-1} + lr * grad / sqrt((ms - mg * mg) + epsilon)
     (TF_AddV2Op:$mom_new
      (TF_MulOp $momentum,
       (CreateTFReadVariableOp $src_op, $grad, $mom_resource)),
      (TF_DivOp
        (TF_MulOp $lr, $grad),
        (TF_SqrtOp
          (TF_AddV2Op
            (TF_SubOp $ms_new, (TF_SquareOp $mg_new)),
            $epsilon
          )
        )
      )
     ),
     (TF_AssignVariableOp $mom_resource, $mom_new, (CreateConstBoolAttrFalse)),
     // var <- var - mom
     (TF_AssignSubVariableOp $var_resource, $mom_new)
   ]
   >;

// Same as DecomposeResourceApplyFtrlV2, with l2_shrinkage set to zero.
// TODO(kramm): Move this pattern to canonicalize.td?
def DecomposeResourceApplyFtrl : Pat<
  (TF_ResourceApplyFtrlOp $var, $accum, $linear, $grad, $lr, $l1, $l2,
     $lr_power, $use_locking, $multiply_linear_by_lr),
  (TF_ResourceApplyFtrlV2Op $var, $accum, $linear, $grad, $lr, $l1, $l2,
     (TF_ConstOp (GetScalarOfType<0> $lr_power)), $lr_power, $use_locking, $multiply_linear_by_lr)>;

// TODO(kramm): Remove use_locking. The old bridge doesn't use this, either.
def DecomposeResourceApplyFtrlV2: Pattern<
  (TF_ResourceApplyFtrlV2Op:$src_op $var, $accum, $linear, $grad, $lr, $l1, $l2,
     $l2_shrinkage, $lr_power, $use_locking, $multiply_linear_by_lr),
  [// grad_with_shrinkage = grad + 2 * l2_shrinkage * var
   (TF_ConstOp:$zero (GetScalarOfType<0> $lr)),
   (TF_ConstOp:$one (GetScalarOfType<1> $lr)),
   (TF_ConstOp:$two (GetScalarOfType<2> $lr)),
   (TF_AddV2Op:$grad_with_shrinkage
     $grad,
     (TF_MulOp
       $two,
       (TF_MulOp $l2_shrinkage, (CreateTFReadVariableOp $src_op, $grad, $var))
     )
   ),
   // accum_new = accum + grad * grad
   (TF_AddV2Op:$accum_new
     (CreateTFReadVariableOp $src_op, $grad, $accum),
     (TF_SquareOp $grad)
   ),
   // if multiply_linear_by_lr:
   //   linear += grad_with_shrinkage * lr - (accum_new^(-lr_power) - accum^(-lr_power)) * var
   // else:
   //   linear += grad_with_shrinkage - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
   (TF_AssignAddVariableOp $linear,
     (ConstAttrIfThenElse
       $multiply_linear_by_lr,
       (TF_SubOp
         (TF_MulOp
           $grad_with_shrinkage,
           $lr),
         (TF_MulOp
           (TF_SubOp:$sub
             (TF_PowOp $accum_new, (TF_NegOp $lr_power)),
             (TF_PowOp (CreateTFReadVariableOp $src_op, $grad, $accum), (TF_NegOp $lr_power))
           ),
           (CreateTFReadVariableOp $src_op, $grad, $var)
         )
       ),
       (TF_SubOp
         $grad_with_shrinkage,
         (TF_MulOp
           (TF_DivOp
             $sub,
             $lr
           ),
           (CreateTFReadVariableOp $src_op, $grad, $var)
         )
       )
     )
   ),
   // quadratic = (new_accum^(-lr_power) / (multiply_linear_by_lr ? 1 : lr))
   //           + 2 * l2 * (multiply_linear_by_lr ? lr : 1)
   (TF_AddV2Op:$quadratic
     (TF_DivOp
       (TF_PowOp
         $accum_new,
         (TF_NegOp $lr_power)
       ),
       (ConstAttrIfThenElse $multiply_linear_by_lr, $one, $lr)
     ),
     (TF_MulOp
       (TF_MulOp $two, $l2),
       (ConstAttrIfThenElse $multiply_linear_by_lr, $lr, $one))
   ),
   // linear_clipped_minus_linear =
   //   (multiply_linear_by_lr ? clamp(-l1 * lr, linear, l1 * lr)
   //                            clamp(-l1, linear, l1)) - linear
   (TF_SubOp:$linear_clipped_minus_linear
     (ConstAttrIfThenElse
       $multiply_linear_by_lr,
       (Clamp $src_op,
         (TF_NegOp (TF_MulOp $l1, $lr)),
         (CreateTFReadVariableOp $src_op, $l1, $linear),
         (TF_MulOp $l1, $lr)
       ),
       (Clamp $src_op,
         (TF_NegOp $l1),
         (CreateTFReadVariableOp $src_op, $l1, $linear),
         $l1
       )
     ),
     (CreateTFReadVariableOp $src_op, $l1, $linear)
   ),
   // var = linear_clipped_minus_linear / quadratic
   (TF_AssignVariableOp $var,
     (TF_DivOp
       $linear_clipped_minus_linear,
       $quadratic
     ),
     (CreateConstBoolAttrFalse)
   ),
   // accum = accum_new
   (TF_AssignVariableOp $accum, $accum_new, (CreateConstBoolAttrFalse)),
  ]>;

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
     (TF_AssignVariableOp $ms_resource, $ms_new, (CreateConstBoolAttrFalse)),
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
     (TF_AssignVariableOp $mom_resource, $mom_new, (CreateConstBoolAttrFalse)),
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
     (TF_AssignVariableOp $var_resource, $var_new, (CreateConstBoolAttrFalse)),
     (TF_AssignVariableOp $accum_resource, $accum_new, (CreateConstBoolAttrFalse))
   ]
   >;

def DecomposeCollectiveReduceV2 :
  Pattern<
    (TF_CollectiveReduceV2Op:$src_op
       $input, $group_size, $group_key, $instance_key, $ordering_token,
       $merge_op, $final_op, $communication_hint, $timeout_seconds,
       $is_stateless, $max_subdivs_per_device
    ),
    [(TF_CollectiveReduceV2Op
       $input, $group_size, $group_key, $instance_key, (EmptyList),
       $merge_op, $final_op, $communication_hint, $timeout_seconds,
       $is_stateless, $max_subdivs_per_device
    )],
    [(HasSingleElement $ordering_token)]
   >;
