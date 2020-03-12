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

// Creates a tf.ReadVariable op that reads a resource `$2` that has the same
// element type as `$1`. The op created will use location of `$1`.
def CreateTFReadVariableOp: NativeCodeCall<
    "$_builder.create<TF::ReadVariableOp>("
    "  $0.getLoc(),"
    "  GetResourceSubtypeOrDefault("
    "    $2, $1.getType().cast<TensorType>().getElementType()),"
    "  $2)"
    >;

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
      (TF_MulOp $grad, $lr)),
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
def DecomposeResourceScatterUpdate : Pat<
  (TF_ResourceScatterUpdateOp:$src_op $resource, $indices, $updates),
  (TF_AssignVariableOp
    $resource,
    (TF_TensorScatterUpdateOp
      (CreateTFReadVariableOp $src_op, $updates, $resource),
      $indices,
      $updates
    )
  )>;
