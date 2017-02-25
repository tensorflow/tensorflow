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

// Kernel registrations for XLA JIT devices.

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

// CPU JIT device registrations.

REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("_Arg").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("_Arg").TypeConstraint("T", DT_RESOURCE));

REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("_ArrayToList"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("_ListToArray"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("_Retval").TypeConstraint("T", kCpuAllTypes));

REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Abs").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Add").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("AddN").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("All"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("Any"));
REGISTER_XLA_KERNEL(
    DEVICE_CPU_XLA_JIT,
    Name("AssignVariableOp").TypeConstraint("dtype", kCpuAllTypes));
REGISTER_XLA_KERNEL(
    DEVICE_CPU_XLA_JIT,
    Name("AssignAddVariableOp").TypeConstraint("dtype", kCpuNumericTypes));
REGISTER_XLA_KERNEL(
    DEVICE_CPU_XLA_JIT,
    Name("AssignSubVariableOp").TypeConstraint("dtype", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("AvgPool").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("AvgPoolGrad").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("BatchMatMul").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("BiasAdd").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("BiasAddV1").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("BiasAddGrad").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("BroadcastGradientArgs"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Cast")
                        .TypeConstraint("SrcT", kCpuAllTypes)
                        .TypeConstraint("DstT", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Ceil").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Concat").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("ConcatV2")
                        .TypeConstraint("T", kCpuAllTypes)
                        .TypeConstraint("Tidx", DT_INT32));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("ConcatOffset"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Conv2D").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(
    DEVICE_CPU_XLA_JIT,
    Name("Conv2DBackpropFilter").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(
    DEVICE_CPU_XLA_JIT,
    Name("Conv2DBackpropInput").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(
    DEVICE_CPU_XLA_JIT,
    Name("DepthwiseConv2dNative").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Diag").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("DiagPart").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Div").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("DynamicStitch").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Equal").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Exp").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("ExpandDims").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Fill").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Floor").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("FloorDiv").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("FloorMod").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Greater").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("GreaterEqual").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Inv").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Reciprocal").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("InvertPermutation").TypeConstraint("T", DT_INT32));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("L2Loss").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Less").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("LessEqual").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("LinSpace").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Log").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Log1p").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("LogicalAnd"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("LogicalNot"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("LogicalOr"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("LogSoftmax").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("LRN").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("LRNGrad").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Maximum").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("MatMul").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("MatrixDiag").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("MatrixDiagPart").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Max").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("MaxPool").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("MaxPoolGrad").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Mean").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Min").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Minimum").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Mod").TypeConstraint("T", kCpuIntTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Mul").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Neg").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("NotEqual").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Pack").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Pad").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Pow").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("PreventGradient").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Prod").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Range").TypeConstraint("Tidx", kCpuNumericTypes));
// TODO(b/34339814): implement inverse erf for double types and update the
// type constraint.
REGISTER_XLA_KERNEL(
    DEVICE_CPU_XLA_JIT,
    Name("RandomStandardNormal").TypeConstraint("dtype", DT_FLOAT));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("RandomUniform"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("RandomUniformInt"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("Rank"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("RealDiv").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(
    DEVICE_CPU_XLA_JIT,
    Name("ReadVariableOp").TypeConstraint("dtype", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Relu").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Relu6").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("ReluGrad").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Relu6Grad").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Reshape").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("ResourceApplyGradientDescent")
                                            .TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Rsqrt").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("RsqrtGrad").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Select").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("Shape"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("ShapeN"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Sigmoid").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("SigmoidGrad").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Sign").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("Size"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Slice").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Softmax").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(
    DEVICE_CPU_XLA_JIT,
    Name("SoftmaxCrossEntropyWithLogits").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Softplus").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("SoftplusGrad").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("SparseMatMul")
                        .TypeConstraint("Ta", kCpuFloatTypes)
                        .TypeConstraint("Tb", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Split").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("SplitV").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Square").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(
    DEVICE_CPU_XLA_JIT,
    Name("SquaredDifference").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Squeeze").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Sqrt").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("StopGradient").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("StridedSlice").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("StridedSliceGrad").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Sub").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Sum").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT, Name("SymbolicGradient"));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Tanh").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("TanhGrad").TypeConstraint("T", kCpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Tile").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Transpose").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("TruncateDiv").TypeConstraint("T", kCpuIntTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("TruncateMod").TypeConstraint("T", kCpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("Unpack").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_CPU_XLA_JIT,
                    Name("ZerosLike").TypeConstraint("T", kCpuNumericTypes));

REGISTER_XLA_JIT_ONLY_KERNEL(DEVICE_CPU_XLA_JIT,
                             Name("Const").TypeConstraint("dtype",
                                                          kCpuAllTypes));
REGISTER_XLA_JIT_ONLY_KERNEL(
    DEVICE_CPU_XLA_JIT, Name("Identity").TypeConstraint("T", kCpuAllTypes));
REGISTER_XLA_JIT_ONLY_KERNEL(DEVICE_CPU_XLA_JIT, Name("NoOp"));

// GPU JIT device registrations

REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("_Arg").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("_Arg").TypeConstraint("T", DT_RESOURCE));

REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("_ArrayToList"));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("_ListToArray"));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("_Retval").TypeConstraint("T", kGpuAllTypes));

REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Abs").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Add").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("AddN").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("All"));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("Any"));
REGISTER_XLA_KERNEL(
    DEVICE_GPU_XLA_JIT,
    Name("AssignVariableOp").TypeConstraint("dtype", kGpuAllTypes));
REGISTER_XLA_KERNEL(
    DEVICE_GPU_XLA_JIT,
    Name("AssignAddVariableOp").TypeConstraint("dtype", kGpuNumericTypes));
REGISTER_XLA_KERNEL(
    DEVICE_GPU_XLA_JIT,
    Name("AssignSubVariableOp").TypeConstraint("dtype", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("AvgPool").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("AvgPoolGrad").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("BatchMatMul").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("BiasAdd").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("BiasAddV1").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("BiasAddGrad").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("BroadcastGradientArgs"));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Cast")
                        .TypeConstraint("SrcT", kGpuAllTypes)
                        .TypeConstraint("DstT", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Ceil").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Concat").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("ConcatV2").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("ConcatOffset"));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Conv2D").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(
    DEVICE_GPU_XLA_JIT,
    Name("Conv2DBackpropFilter").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(
    DEVICE_GPU_XLA_JIT,
    Name("Conv2DBackpropInput").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(
    DEVICE_GPU_XLA_JIT,
    Name("DepthwiseConv2dNative").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Diag").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("DiagPart").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Div").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("DynamicStitch").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Equal").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Exp").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("ExpandDims").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Fill").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Floor").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("FloorDiv").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("FloorMod").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Greater").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("GreaterEqual").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Inv").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Reciprocal").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("InvertPermutation").TypeConstraint("T", DT_INT32));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("L2Loss").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Less").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("LessEqual").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("LinSpace").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Log").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Log1p").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("LogicalAnd"));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("LogicalNot"));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("LogicalOr"));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("LogSoftmax").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("LRN").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("LRNGrad").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Maximum").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("MatMul").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("MatrixDiag").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("MatrixDiagPart").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Max").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("MaxPool").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("MaxPoolGrad").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Mean").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Min").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Minimum").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Mod").TypeConstraint("T", kGpuIntTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Mul").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Neg").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("NotEqual").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Pack").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Pad").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Pow").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("PreventGradient").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Prod").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Range").TypeConstraint("Tidx", kGpuNumericTypes));
// TODO(b/31361304): disabled because of XLA bugs.
// REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("RandomStandardNormal"));
// REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("RandomUniform"));
// REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("RandomUniformInt"));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("Rank"));
REGISTER_XLA_KERNEL(
    DEVICE_GPU_XLA_JIT,
    Name("ReadVariableOp").TypeConstraint("dtype", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("RealDiv").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Relu").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Relu6").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("ReluGrad").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Relu6Grad").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Reshape").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("ResourceApplyGradientDescent")
                                            .TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Rsqrt").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("RsqrtGrad").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Select").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("Shape"));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("ShapeN"));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Sigmoid").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("SigmoidGrad").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Sign").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("Size"));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Slice").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Softmax").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(
    DEVICE_GPU_XLA_JIT,
    Name("SoftmaxCrossEntropyWithLogits").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Softplus").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("SoftplusGrad").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("SparseMatMul")
                        .TypeConstraint("Ta", kGpuFloatTypes)
                        .TypeConstraint("Tb", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Split").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("SplitV").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Square").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(
    DEVICE_GPU_XLA_JIT,
    Name("SquaredDifference").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Squeeze").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Sqrt").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("StopGradient").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("StridedSlice").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("StridedSliceGrad").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Sub").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Sum").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT, Name("SymbolicGradient"));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Tanh").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("TanhGrad").TypeConstraint("T", kGpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Tile").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Transpose").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("TruncateDiv").TypeConstraint("T", kGpuIntTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("TruncateMod").TypeConstraint("T", kGpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("Unpack").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_GPU_XLA_JIT,
                    Name("ZerosLike").TypeConstraint("T", kGpuNumericTypes));

REGISTER_XLA_JIT_ONLY_KERNEL(DEVICE_GPU_XLA_JIT,
                             Name("Const").TypeConstraint("dtype",
                                                          kGpuAllTypes));
REGISTER_XLA_JIT_ONLY_KERNEL(
    DEVICE_GPU_XLA_JIT, Name("Identity").TypeConstraint("T", kGpuAllTypes));
REGISTER_XLA_JIT_ONLY_KERNEL(DEVICE_GPU_XLA_JIT, Name("NoOp"));

}  // anonymous namespace
}  // namespace tensorflow
