// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "tensorflow/core/function/runtime_client.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace tensorflow {
namespace fuzzing {

using namespace core::function;

EagerContextPtr LocalEagerCtx() {
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

FunctionDef EmptyFunctionDefGenerator(int number_of_input_arguments, int number_of_output_arguments) {
  std::vector<string> in_def_vec;
  in_def_vec.reserve(number_of_input_arguments);
  for (int c = 0; c < number_of_input_arguments; ++c) {
    in_def_vec.push_back("in" + std::to_string(c) + ":float");
  }
  std::vector<FunctionDefHelper::Node> body_nodes;
  if (number_of_output_arguments > number_of_input_arguments) {
    Tensor const_value(DataTypeToEnum<float>::value, {});
    const_value.scalar<float>()() = 0;
    body_nodes.push_back({{"zero"}, "Const", {}, {{"value", const_value}, {"dtype", DT_FLOAT}}});
  }
  std::vector<string> out_def_vec;
  out_def_vec.reserve(number_of_output_arguments);
  std::vector<std::pair<string, string>> ret_def;
  ret_def.reserve(number_of_output_arguments);
  for (int c = 0; c < number_of_output_arguments; ++c) {
    auto output_id = "out" + std::to_string(c);
    out_def_vec.push_back(output_id + ":float");
    if (c < number_of_input_arguments) {
      ret_def.emplace_back(output_id, "in" + std::to_string(c));
    } else {
      ret_def.emplace_back(output_id, "zero:output");
    }
  }
  return FunctionDefHelper::Create("TestFunction", in_def_vec, out_def_vec, {}, body_nodes, ret_def);
}

auto NumberOfInputArguments() {
  return fuzztest::InRange(0, 7);
}

auto NumberOfOutputArguments() {
  return fuzztest::InRange(1, 7);
}

auto FunctionDefDomain() {
  return fuzztest::Map(EmptyFunctionDefGenerator, NumberOfInputArguments(), NumberOfOutputArguments());
}

void CreateFunctionFuzz(FunctionDef def) {
  auto ctx = LocalEagerCtx();
  Runtime rt(*ctx);
  ASSERT_EQ(::tsl::OkStatus(), rt.CreateFunction(def));
}

FUZZ_TEST(FuzzRuntimeClient, CreateFunctionFuzz).WithDomains(FunctionDefDomain());

void CreateCallFunction(int number_of_input_arguments, int number_of_output_arguments) {
  auto ctx = LocalEagerCtx();
  Runtime rt(*ctx);
  ASSERT_EQ(::tsl::OkStatus(), rt.CreateFunction(EmptyFunctionDefGenerator(number_of_input_arguments, number_of_output_arguments)));

  AbstractTensorPtr tensor(ctx->CreateFloatScalar(42));
  ImmediateTensorHandlePtr handle(ctx->CreateLocalHandle(tensor.get()));
  std::vector<AbstractTensorHandle*> args(number_of_input_arguments, handle.get());

  StatusOr<ReturnValues> rets = rt.CallFunction("TestFunction", args);
  ASSERT_EQ(::tsl::OkStatus(), rets.status());
  ASSERT_EQ(rets->size(), number_of_output_arguments);
  ASSERT_EQ(rets->at(0)->DataType(), DT_FLOAT);
}

FUZZ_TEST(FuzzRuntimeClient, CreateCallFunction).WithDomains(NumberOfInputArguments(), NumberOfOutputArguments());



auto EagerContextWithFunctionDomain() {
  return fuzztest::Map(
      [](FunctionDef def) {
        EagerContextPtr ctx = LocalEagerCtx();
        Status status = ctx->AddFunctionDef(def);
        if (!status.ok()) {
          LOG(FATAL) << "Could not add function definition to EagerContext: "  // Crash OK
                     << status.error_message();
        }
        return ctx;
      },
      FunctionDefDomain());
}

void UpdateFunctionFuzz(EagerContextPtr ctx, FunctionDef def) {
  Runtime rt(*ctx);
  ASSERT_EQ(::tsl::OkStatus(), rt.CreateFunction(def));
}

FUZZ_TEST(FuzzRuntimeClient, UpdateFunctionFuzz).WithDomains(EagerContextWithFunctionDomain(), FunctionDefDomain());

void GetFunctionProtoFuzz(EagerContextPtr ctx) {
  Runtime rt(*ctx);
  StatusOr<FunctionDef> fdef_ret = rt.GetFunctionProto("TestFunction");
  ASSERT_EQ(::tsl::OkStatus(), fdef_ret.status());
}

FUZZ_TEST(FuzzRuntimeClient, GetFunctionProtoFuzz).WithDomains(EagerContextWithFunctionDomain());


std::string MlirFunctionDefGenerator(int number_of_input_arguments, int number_of_output_arguments) {
  std::stringstream in_def_ss;
  for (int c = 0; c < number_of_input_arguments; ++c) {
    if (c > 0) {
      in_def_ss << ", ";
    }
    in_def_ss << "%in" << c <<  ": tensor<i32> {tfg.name = \"in" << c << "\"}";
  }
  std::stringstream out_def_ss;
  for (int c = 0; c < number_of_output_arguments; ++c) {
    if (c > 0) {
      out_def_ss << ", ";
    }
    if (c < number_of_input_arguments) {
      out_def_ss << "%in" << c;
    } else {
      out_def_ss << "%Const";
    }
  }
  std::stringstream out_type_with_name_ss;
  for (int c = 1; c < number_of_output_arguments; ++c) {
    out_type_with_name_ss << ", tensor<i32> {tfg.dtype = i32, tfg.name = \"out" << c << "\"}";
  }
  std::stringstream out_type_ss;
  for (int c = 1; c < number_of_output_arguments; ++c) {
    out_type_ss << ", tensor<i32>";
  }
  std::stringstream ss;
  ss << "module  {\n"
        "  tfg.func @TestFunction(" << in_def_ss.str() << ") \n"
        "     -> (tensor<i32> {tfg.dtype = i32, tfg.name = \"out0\"}" << out_type_with_name_ss.str() << ")\n"
        "  {\n" <<
        (number_of_output_arguments > number_of_input_arguments ?
        "    %Const, %ctl = Const  name(\"retval\") {dtype = i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)\n" : "") <<
        "    return(" << out_def_ss.str() << ") : tensor<i32>" << out_type_ss.str() << "\n"
        "  }\n"
        "}\n";
  return ss.str();
}

auto MlirFunctionDefDomain() {
  return fuzztest::Map(MlirFunctionDefGenerator, NumberOfInputArguments(), NumberOfOutputArguments());
}

void CreateMlirFunctionFuzz(std::string def) {
  auto ctx = LocalEagerCtx();
  Runtime rt(*ctx);
  mlir::MLIRContext mctx;
  mctx.getOrLoadDialect<mlir::tfg::TFGraphDialect>();
  auto m = mlir::parseSourceString<mlir::ModuleOp>(def, &mctx);
  mlir::tfg::GraphFuncOp fop =
      *m->getBody()->op_begin<mlir::tfg::GraphFuncOp>();
  // Note: this is the price we'll have to pay until we can properly link
  // MLIR headers into pybind wrappers (not to be confused with pybind
  // converters, which are a separate thing - we just talk about header
  // dependencies here).
  OpaqueTfgGraphFuncOp* opaque_fop =
      reinterpret_cast<OpaqueTfgGraphFuncOp*>(&fop);
  ASSERT_EQ(::tsl::OkStatus(), rt.CreateFunction(opaque_fop));
}

FUZZ_TEST(FuzzRuntimeClient, CreateMlirFunctionFuzz).WithDomains(MlirFunctionDefDomain());

class TransformFunctionFuzzTest {
public:
  TransformFunctionFuzzTest() {
    EnsureTestPassRegistered();
  }

  void Fuzz(EagerContextPtr ctx) {
    Runtime rt(*ctx);
    ASSERT_EQ(::tsl::OkStatus(), rt.TransformFunction("TestFunction", "test-pass"));
  }

private:
  struct TestPassMock
      : public mlir::PassWrapper<TestPassMock, mlir::OperationPass<mlir::ModuleOp>> {
    TestPassMock() = default;
    llvm::StringRef getArgument() const final { return "test-pass"; }
    void runOnOperation() override {}
  };

  void EnsureTestPassRegistered() {
    mlir::MLIRContext ctx;
    mlir::PassManager pm(&ctx);
    std::string error;
    llvm::raw_string_ostream error_stream(error);
    if (mlir::failed(mlir::parsePassPipeline("test-pass", pm, error_stream))) {
      mlir::registerPass([] { return std::make_unique<TestPassMock>(); });
    }
  }
};

FUZZ_TEST_F(TransformFunctionFuzzTest, Fuzz).WithDomains(EagerContextWithFunctionDomain());

}  // end namespace fuzzing
}  // end namespace tensorflow
