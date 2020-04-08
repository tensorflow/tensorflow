/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

include "tensorflow/compiler/mlir/xla/ir/hlo_ops.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"

class Fused1Ops<string kernel> : NativeCodeCall<
  "FuseOps(&$_builder, {$0}, \"" # kernel # "\")">;
class Fused2Ops<string kernel> : NativeCodeCall<
  "FuseOps(&$_builder, {$0, $1}, \"" # kernel # "\")">;
class Fused3Ops<string kernel> : NativeCodeCall<
  "FuseOps(&$_builder, {$0, $1, $2}, \"" # kernel # "\")">;
class Fused4Ops<string kernel> : NativeCodeCall<
  "FuseOps(&$_builder, {$0, $1, $2, $3}, \"" # kernel # "\")">;

// We shouldn't revisit those ops which have been fused. This constraint is
// required because the greedy pattern rewriter will visit and match any new
// ops. So when the source pattern are matched and wrapped by the quant region
// op, these ops will be matched again. To prevent this, this constraint is
// added to bypass any ops inside a quant region.
def NeedsToBeFused : Constraint<CPred<
  "!$0.getDefiningOp()->getParentOfType<quant::QuantizeRegionOp>()">>;

// dummy example
def : Pat<(HLO_AddOp:$add (HLO_MulOp:$mul $_, $_, $_), $_, $_),
          (Fused2Ops<"generic.mul_add"> $mul, $add),
          [(NeedsToBeFused $add)]>;

// reduce_window: maxpool, avgpool
def : Pat<(HLO_ReduceWindowOp:$reduce $_, $_, $_, $_, $_, $_, $_),
          (Fused1Ops<"generic.reduce_window"> $reduce),
          [(NeedsToBeFused $reduce)]>;

// reshape
def : Pat<(HLO_ReshapeOp:$reshape $_), (Fused1Ops<"generic.reshape"> $reshape),
          [(NeedsToBeFused $reshape)]>;

// broadcast
def : Pat<(HLO_BroadcastInDimOp:$broadcast $_, $_),
          (Fused1Ops<"generic.broadcast"> $broadcast),
          [(NeedsToBeFused $broadcast)]>;

// dot -> add
def : Pat<(HLO_AddOp:$add (HLO_DotOp:$dot $_, $_, $_), $_, $_),
          (Fused2Ops<"generic.biased_dot"> $dot, $add),
          [(NeedsToBeFused $add)]>;

// conv -> add
def : Pat<(HLO_AddOp:$add
             (HLO_ConvOp:$conv $_, $_, $_, $_, $_, $_, $_, $_, $_, $_), $_, $_),
          (Fused2Ops<"generic.biased_conv"> $conv, $add),
          [(NeedsToBeFused $add)]>;
