/* Copyright 2017 Graphcore Ltd
 */

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

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/jit/kernels/xla_device_launch_op.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

const char* const DEVICE_XLA_IPU = "XLA_IPU";
const char* const DEVICE_IPU_XLA_JIT = "XLA_IPU_JIT";

constexpr std::array<DataType, 5> kIpuAllTypes =
        {{DT_INT32, DT_FLOAT, DT_BOOL}};
constexpr std::array<DataType, 2> kIpuIntTypes =
        {{DT_INT32}};
constexpr std::array<DataType, 2> kIpuFloatTypes =
        {{DT_FLOAT}};
constexpr std::array<DataType, 4> kIpuNumericTypes =
        {{DT_INT32, DT_FLOAT}};

class XlaIpuDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override;
};

Status XlaIpuDeviceFactory::CreateDevices(const SessionOptions& options,
                                          const string& name_prefix,
                                          std::vector<Device*>* devices) {
  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_XLA_IPU, DEVICE_IPU_XLA_JIT);
  (void)registrations;

  std::unique_ptr<XlaDevice> device;
  TF_RETURN_IF_ERROR(XlaDevice::Create("Poplar", DEVICE_XLA_IPU, 0,
                                       DEVICE_IPU_XLA_JIT, options, name_prefix,
                                       &device));
  devices->push_back(device.release());
  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_IPU, XlaIpuDeviceFactory);

// Kernel registrations

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_IPU, XlaDeviceLaunchOp, kIpuAllTypes);
REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_IPU, kIpuAllTypes);

// Register JIT kernels for IPU device
// synced @ eaa668e7e5d28072964ce8b78c155720aed951d3

REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("_Arg").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("_Arg").TypeConstraint("T", DT_RESOURCE));

REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("_ArrayToList"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("_ListToArray"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("_Retval").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Abs").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Add").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("AddN").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("All"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Any"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("AssignVariableOp").TypeConstraint("dtype", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("AssignAddVariableOp").TypeConstraint("dtype", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("AssignSubVariableOp").TypeConstraint("dtype", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("AvgPool").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("AvgPoolGrad").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("BatchMatMul").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("BiasAdd").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("BiasAddV1").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("BiasAddGrad").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("BroadcastGradientArgs"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Cast").TypeConstraint("SrcT", kIpuAllTypes)
                    .TypeConstraint("DstT", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Ceil").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Concat").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("ConcatV2").TypeConstraint("T", kIpuAllTypes)
                        .TypeConstraint("Tidx", DT_INT32));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("ConcatOffset"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Conv2D").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Conv2DBackpropFilter").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Conv2DBackpropInput").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("DepthwiseConv2dNative").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Diag").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("DiagPart").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Div").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("DynamicStitch").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Equal").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Exp").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("ExpandDims").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Fill").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Floor").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("FloorDiv").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("FloorMod").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Greater").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("GreaterEqual").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Inv").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Reciprocal").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("InvertPermutation").TypeConstraint("T", DT_INT32));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("L2Loss").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Less").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("LessEqual").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("LinSpace").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Log").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Log1p").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("LogicalAnd"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("LogicalNot"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("LogicalOr"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("LogSoftmax").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("LRN").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("LRNGrad").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Maximum").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("MatMul").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("MatrixDiag").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("MatrixDiagPart").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Max").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("MaxPool").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("MaxPoolGrad").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Mean").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Min").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Minimum").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Mod").TypeConstraint("T", kIpuIntTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Mul").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Neg").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("NotEqual").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("OneHot").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Pack").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Pad").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Pow").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("PreventGradient").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Prod").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Range").TypeConstraint("Tidx", kIpuNumericTypes));
// TODO(b/34339814): implement inverse erf for double types and update the
// type constraint.
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("RandomStandardNormal").TypeConstraint("dtype", DT_FLOAT));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("RandomUniform"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("RandomUniformInt"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Rank"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("RealDiv").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("ReadVariableOp").TypeConstraint("dtype", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Relu").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Relu6").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("ReluGrad").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Relu6Grad").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Reshape").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("ResourceApplyGradientDescent").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Rsqrt").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("RsqrtGrad").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Select").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Shape"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("ShapeN"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Sigmoid").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("SigmoidGrad").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Sign").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Size"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Slice").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Softmax").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("SoftmaxCrossEntropyWithLogits")
            .TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Softplus").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("SoftplusGrad").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("SparseMatMul").TypeConstraint("Ta", kIpuFloatTypes)
                            .TypeConstraint("Tb", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("SparseSoftmaxCrossEntropyWithLogits")
            .TypeConstraint("T", kCpuFloatTypes)
            .TypeConstraint("Tlabels", kCpuIntTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Split").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("SplitV").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Square").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("SquaredDifference").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Squeeze").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Sqrt").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("StopGradient").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("StridedSlice").TypeConstraint("T", kIpuAllTypes)
                            .TypeConstraint("Index", kIpuIntTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("StridedSliceGrad").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Sub").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Sum").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("SymbolicGradient"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Tanh").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("TanhGrad").TypeConstraint("T", kIpuFloatTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Tile").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Transpose").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("TruncateDiv").TypeConstraint("T", kIpuIntTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("TruncateMod").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Unpack").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("VarIsInitializedOp"));
REGISTER_XLA_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("ZerosLike").TypeConstraint("T", kIpuNumericTypes));
REGISTER_XLA_JIT_ONLY_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("Const").TypeConstraint("dtype",kIpuAllTypes));
REGISTER_XLA_JIT_ONLY_KERNEL(
        DEVICE_IPU_XLA_JIT, Name("Identity").TypeConstraint("T", kIpuAllTypes));
REGISTER_XLA_JIT_ONLY_KERNEL(DEVICE_IPU_XLA_JIT,
        Name("NoOp"));



}  // namespace tensorflow
