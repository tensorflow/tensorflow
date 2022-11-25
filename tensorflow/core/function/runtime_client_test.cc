/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/function/runtime_client.h"

#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/function/testing/test_pass.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace core {
namespace function {
namespace {

EagerContextPtr TestingEagerCtx() {
  SessionOptions opts;
  std::vector<std::unique_ptr<Device>> devices;
  Status&& device_init_status = DeviceFactory::AddDevices(
      opts, "/job:localhost/replica:0/task:0", &devices);
  CHECK(device_init_status.ok());  // Crash OK

  return EagerContextPtr(new EagerContext(
      opts, ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      /*async=*/false,
      /*device_mgr=*/new DynamicDeviceMgr(std::move(devices)),
      /*device_mgr_owned=*/true,
      /*rendezvous=*/nullptr,
      /*cluster_flr=*/nullptr,
      /*collective_executor_mgr=*/nullptr,
      /*run_eager_op_as_function=*/true));
}

int IntValue(ImmediateExecutionTensorHandle& h) {
  Status status;
  AbstractTensorPtr t(h.Resolve(&status));
  DCHECK(status.ok());
  switch (h.DataType()) {
    case DT_INT32:
      return *(static_cast<int32_t*>(t->Data()));
    case DT_INT64:
      return *(static_cast<int64_t*>(t->Data()));
    default:
      DCHECK(false) << "invalid data type";
      return 0;
  }
}

ImmediateTensorHandlePtr IntScalarTensor(EagerContext& ctx, int value) {
  AbstractTensorPtr tensor(ctx.CreateInt32Scalar(value));
  ImmediateTensorHandlePtr handle(ctx.CreateLocalHandle(tensor.get()));
  return handle;
}

FunctionDef MakeNullaryFunction() {
  FunctionDef fd;
  protobuf::TextFormat::Parser parser;
  CHECK(parser.ParseFromString(
      R"pb(signature {
             name: 'NullaryFunction'
             output_arg { name: 'o' type: DT_INT32 }
           }
           node_def {
             name: 'retval'
             op: 'Const'
             attr {
               key: 'dtype'
               value { type: DT_INT32 }
             }
             attr {
               key: 'value'
               value {
                 tensor {
                   dtype: DT_INT32
                   tensor_shape {}
                   int_val: 1
                 }
               }
             }
           }
           ret { key: 'o' value: 'retval:output' })pb",
      &fd));
  return fd;
}

FunctionDef MakeUnaryFunction() {
  FunctionDef fd;
  protobuf::TextFormat::Parser parser;
  CHECK(parser.ParseFromString(
      R"pb(signature {
             name: "UnaryFunction"
             input_arg { name: "x" type: DT_INT32 }
             output_arg { name: "ret" type: DT_INT32 }
           }
           node_def {
             name: "ret"
             op: "Identity"
             input: "x"
             attr {
               key: "T"
               value { type: DT_INT32 }
             }
           }
           ret { key: "ret" value: "ret:output:0" })pb",
      &fd));
  return fd;
}

FunctionDef MakeBinaryFunction() {
  FunctionDef fd;
  protobuf::TextFormat::Parser parser;
  CHECK(parser.ParseFromString(
      R"pb(signature {
             name: "BinaryFunction"
             input_arg { name: "x" type: DT_INT32 }
             input_arg { name: "y" type: DT_INT32 }
             output_arg { name: "ret" type: DT_INT32 }
           }
           node_def {
             name: "x_plus_y"
             op: "AddV2"
             input: "x"
             input: "y"
             attr {
               key: "T"
               value { type: DT_INT32 }
             }
           }
           node_def {
             name: "ret"
             op: "Identity"
             input: "x_plus_y:z:0"
             attr {
               key: "T"
               value { type: DT_INT32 }
             }
           }
           ret { key: "ret" value: "ret:output:0" })pb",
      &fd));
  return fd;
}

TEST(GlobalContext, Basic) {
  Runtime rt(GlobalEagerContext());
  TF_ASSERT_OK(rt.CreateFunction(MakeNullaryFunction()));

  StatusOr<ReturnValues> rets = rt.CallFunction("NullaryFunction", {});
  TF_ASSERT_OK(rets.status());
  ASSERT_EQ(rets->size(), 1);
  ASSERT_EQ(rets->at(0)->DataType(), DT_INT32);
  EXPECT_EQ(IntValue(*(rets->at(0))), 1);
}

TEST(CreateTest, Call) {
  EagerContextPtr ctx = TestingEagerCtx();
  Runtime rt(*ctx);
  TF_ASSERT_OK(rt.CreateFunction(MakeNullaryFunction()));

  StatusOr<ReturnValues> rets = rt.CallFunction("NullaryFunction", {});
  TF_ASSERT_OK(rets.status());
  ASSERT_EQ(rets->size(), 1);
  ASSERT_EQ(rets->at(0)->DataType(), DT_INT32);
  EXPECT_EQ(IntValue(*(rets->at(0))), 1);
}

TEST(CreateTest, GetRoundtrip) {
  EagerContextPtr ctx = TestingEagerCtx();
  Runtime rt(*ctx);
  TF_ASSERT_OK(rt.CreateFunction(MakeNullaryFunction()));

  StatusOr<FunctionDef> fdef_ret = rt.GetFunctionProto("NullaryFunction");
  TF_ASSERT_OK(fdef_ret.status());

  FunctionDef fdef = *fdef_ret;
  fdef.mutable_signature()->set_name("SecondFunction");

  TF_ASSERT_OK(rt.CreateFunction(fdef));

  StatusOr<ReturnValues> rets = rt.CallFunction("SecondFunction", {});
  TF_ASSERT_OK(rets.status());
  ASSERT_EQ(rets->size(), 1);
  ASSERT_EQ(rets->at(0)->DataType(), DT_INT32);
  EXPECT_EQ(IntValue(*(rets->at(0))), 1);
}

TEST(CreateTest, MlirFromGraphDef) {
  mlir::MLIRContext mctx;
  mctx.getOrLoadDialect<mlir::tfg::TFGraphDialect>();
  auto m = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
        module  {
          tfg.func @NullaryFunction()
               -> (tensor<i32> {tfg.dtype = i32, tfg.name = "o"})
           {
            %Const, %ctl = Const name("retval") {dtype = i32, value = dense<1> : tensor<i32>} : () -> (tensor<i32>)
            return(%Const) : tensor<i32>
          }
        }
      )mlir",
      &mctx);

  mlir::tfg::GraphFuncOp fop =
      *m->getBody()->op_begin<mlir::tfg::GraphFuncOp>();

  EagerContextPtr ectx = TestingEagerCtx();
  Runtime rt(*ectx);
  // Note: this is the price we'll have to pay until we can properly link
  // MLIR headers into pybind wrappers (not to be confused with pybind
  // converters, which are a separate thing - we just talk about header
  // dependencies here).
  OpaqueTfgGraphFuncOp* opaque_fop =
      reinterpret_cast<OpaqueTfgGraphFuncOp*>(&fop);
  TF_ASSERT_OK(rt.CreateFunction(opaque_fop));

  StatusOr<ReturnValues> rets = rt.CallFunction("NullaryFunction", {});
  TF_ASSERT_OK(rets.status());
  ASSERT_EQ(rets->size(), 1);
  ASSERT_EQ(rets->at(0)->DataType(), DT_INT32);
  EXPECT_EQ(IntValue(*(rets->at(0))), 1);
}

TEST(CallTest, Nullary) {
  EagerContextPtr ctx = TestingEagerCtx();
  Runtime rt(*ctx);
  TF_ASSERT_OK(rt.CreateFunction(MakeNullaryFunction()));

  StatusOr<ReturnValues> rets = rt.CallFunction("NullaryFunction", {});
  TF_ASSERT_OK(rets.status());
  ASSERT_EQ(rets->size(), 1);
  ASSERT_EQ(rets->at(0)->DataType(), DT_INT32);
  EXPECT_EQ(IntValue(*(rets->at(0))), 1);
}

TEST(CallTest, Unary) {
  EagerContextPtr ctx = TestingEagerCtx();
  Runtime rt(*ctx);
  TF_ASSERT_OK(rt.CreateFunction(MakeUnaryFunction()));

  auto x = IntScalarTensor(*ctx, 1);
  StatusOr<ReturnValues> rets = rt.CallFunction("UnaryFunction", {x.get()});
  TF_ASSERT_OK(rets.status());
  ASSERT_EQ(rets->size(), 1);
  ASSERT_EQ(rets->at(0)->DataType(), DT_INT32);
  EXPECT_EQ(IntValue(*(rets->at(0))), 1);
}

TEST(CallTest, Binary) {
  EagerContextPtr ctx = TestingEagerCtx();
  Runtime rt(*ctx);
  TF_ASSERT_OK(rt.CreateFunction(MakeBinaryFunction()));

  auto x = IntScalarTensor(*ctx, 1);
  auto y = IntScalarTensor(*ctx, 1);
  StatusOr<ReturnValues> rets =
      rt.CallFunction("BinaryFunction", {x.get(), y.get()});
  TF_ASSERT_OK(rets.status());
  ASSERT_EQ(rets->size(), 1);
  ASSERT_EQ(rets->at(0)->DataType(), DT_INT32);
  EXPECT_EQ(IntValue(*(rets->at(0))), 2);
}

TEST(TransformTest, TestPassOnBinaryFunction) {
  EagerContextPtr ctx = TestingEagerCtx();
  Runtime rt(*ctx);
  TF_ASSERT_OK(rt.CreateFunction(MakeBinaryFunction()));

  testing::RegisterTestPass();
  TF_EXPECT_OK(rt.TransformFunction("BinaryFunction", "test-pass"));

  auto x = IntScalarTensor(*ctx, 2);
  auto y = IntScalarTensor(*ctx, 3);
  StatusOr<ReturnValues> rets =
      rt.CallFunction("BinaryFunction", {x.get(), y.get()});
  TF_ASSERT_OK(rets.status());
  ASSERT_EQ(rets->size(), 1);
  ASSERT_EQ(rets->at(0)->DataType(), DT_INT32);
  EXPECT_EQ(IntValue(*(rets->at(0))), 6);
}

}  // namespace
}  // namespace function
}  // namespace core
}  // namespace tensorflow
