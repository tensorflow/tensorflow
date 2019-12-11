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
include "mlir/Dialect/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

def CreateTFReadVariableOp: NativeCodeCall<
    "$_builder.create<TF::ReadVariableOp>("
    "  $0.getLoc(),"
    "  UnrankedTensorType::get("
    "    $1->getType().cast<TensorType>().getElementType()),"
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
