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

#include "tensorflow/core/common_runtime/function.h"

#include <atomic>
#include <functional>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace {

using FDH = ::tensorflow::FunctionDefHelper;

using OutputControlSrc = InlineFunctionBodyOptions::OutputControlSource;

Status GetOpSig(const string& op, const OpDef** sig) {
  return OpRegistry::Global()->LookUpOpDef(op, sig);
}

void HasError(const Status& s, const error::Code code, StringPiece substr) {
  EXPECT_EQ(s.code(), code) << s;
  EXPECT_TRUE(absl::StrContains(s.error_message(), substr))
      << s << ", expected substring " << substr;
}

class FunctionTest : public ::testing::Test {
 protected:
  FunctionTest()
      : device_(DeviceFactory::NewDevice("CPU", {},
                                         "/job:localhost/replica:0/task:0")) {}

  void Create(const FunctionDef& fdef, test::function::Attrs attrs) {
    exec_ = nullptr;
    InstantiationResult result;
    TF_CHECK_OK(InstantiateFunction(fdef, attrs, GetOpSig, &result));

    arg_types_ = result.arg_types;
    ret_types_ = result.ret_types;

    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    opts.expect_device_spec = false;
    TF_CHECK_OK(ConvertNodeDefsToGraph(opts, result.nodes, g.get()));

    const int version = g->versions().producer();
    LocalExecutorParams params;
    params.device = device_.get();
    params.create_kernel =
        [this, version](const std::shared_ptr<const NodeProperties>& props,
                        OpKernel** kernel) {
          return CreateNonCachedKernel(device_.get(), nullptr, props, version,
                                       kernel);
        };
    params.delete_kernel = [](OpKernel* kernel) {
      DeleteNonCachedKernel(kernel);
    };
    Executor* exec;
    TF_CHECK_OK(NewLocalExecutor(params, *g, &exec));
    exec_.reset(exec);
  }

  void Run(const std::vector<Tensor>& args, std::vector<Tensor*> rets) {
    FunctionCallFrame frame(arg_types_, ret_types_);
    TF_CHECK_OK(frame.SetArgs(args));
    Executor::Args exec_args;
    exec_args.call_frame = &frame;
    exec_args.runner = test::function::FunctionTestSchedClosure;
    TF_CHECK_OK(exec_->Run(exec_args));
    std::vector<Tensor> computed;
    TF_CHECK_OK(frame.GetRetvals(&computed));
    CHECK_EQ(computed.size(), rets.size());
    for (int i = 0; i < rets.size(); ++i) {
      *(rets[i]) = computed[i];
    }
  }

  std::unique_ptr<Device> device_;
  std::unique_ptr<Executor> exec_;
  DataTypeVector arg_types_;
  DataTypeVector ret_types_;
};

TEST_F(FunctionTest, XTimesTwo) {
  Create(test::function::XTimesTwo(), {{"T", DT_FLOAT}});
  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;
  Run({x}, {&y});
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
}

TEST_F(FunctionTest, WXPlusB) {
  Create(test::function::WXPlusB(), {{"T", DT_FLOAT}});
  auto w = test::AsTensor<float>({1., 2., 3., 4.}, {2, 2});
  auto x = test::AsTensor<float>({1., 3., 2., 4.}, {2, 2});
  auto b = test::AsTensor<float>({0.5, 2.5}, {2});
  Tensor y;
  Run({w, x, b}, {&y});
  test::ExpectTensorEqual<float>(
      y, test::AsTensor<float>({5.5, 13.5, 11.5, 27.5}, {2, 2}));
}

class FunctionLibraryRuntimeTest : public ::testing::Test {
 protected:
  void Init(const std::vector<FunctionDef>& flib) {
    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", 3});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices));

    FunctionDefLibrary proto;
    for (const auto& fdef : flib) *(proto.add_function()) = fdef;
    lib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), proto));
    OptimizerOptions opts;
    device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
    pflr_.reset(new ProcessFunctionLibraryRuntime(
        device_mgr_.get(), Env::Default(), &options.config,
        TF_GRAPH_DEF_VERSION, lib_def_.get(), opts, /*thread_pool=*/nullptr,
        /*parent=*/nullptr, /*session_metadata=*/nullptr,
        Rendezvous::Factory{
            [](const int64_t, const DeviceMgr* device_mgr, Rendezvous** r) {
              *r = new IntraProcessRendezvous(device_mgr);
              return Status::OK();
            }}));
    flr0_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
    flr1_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:1");
    flr2_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:2");
    fdef_lib_ = lib_def_->ToProto();
  }

  Status Run(FunctionLibraryRuntime* flr, FunctionLibraryRuntime::Handle handle,
             FunctionLibraryRuntime::Options opts,
             const std::vector<Tensor>& args, std::vector<Tensor*> rets) {
    std::function<void(std::function<void()>)> runner =
        [](std::function<void()> fn) {
          test::function::FunctionTestSchedClosure(fn);
        };
    opts.runner = &runner;
    Notification done;
    std::vector<Tensor> out;
    Status status;
    flr->Run(opts, handle, args, &out, [&status, &done](const Status& s) {
      status = s;
      done.Notify();
    });
    done.WaitForNotification();
    if (!status.ok()) {
      return status;
    }
    CHECK_EQ(rets.size(), out.size());
    for (size_t i = 0; i < rets.size(); ++i) {
      *rets[i] = out[i];
    }
    return Status::OK();
  }

  Status Instantiate(FunctionLibraryRuntime* flr, const string& name,
                     test::function::Attrs attrs,
                     FunctionLibraryRuntime::Handle* handle) {
    return flr->Instantiate(name, attrs, handle);
  }

  Status Instantiate(FunctionLibraryRuntime* flr, const string& name,
                     test::function::Attrs attrs,
                     const FunctionLibraryRuntime::InstantiateOptions& options,
                     FunctionLibraryRuntime::Handle* handle) {
    return flr->Instantiate(name, attrs, options, handle);
  }

  Status InstantiateAndRun(FunctionLibraryRuntime* flr, const string& name,
                           test::function::Attrs attrs,
                           const std::vector<Tensor>& args,
                           std::vector<Tensor*> rets) {
    return InstantiateAndRun(flr, name, attrs,
                             FunctionLibraryRuntime::InstantiateOptions(), args,
                             std::move(rets));
  }

  Status InstantiateAndRun(
      FunctionLibraryRuntime* flr, const string& name,
      test::function::Attrs attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      const std::vector<Tensor>& args, std::vector<Tensor*> rets) {
    FunctionLibraryRuntime::Handle handle;
    Status status = flr->Instantiate(name, attrs, options, &handle);
    if (!status.ok()) {
      return status;
    }
    FunctionLibraryRuntime::Options opts;
    status = Run(flr, handle, opts, args, rets);
    if (!status.ok()) return status;

    // Release the handle and try running again. It should not succeed.
    status = flr->ReleaseHandle(handle);
    if (!status.ok()) return status;

    Status status2 = Run(flr, handle, opts, args, std::move(rets));
    EXPECT_TRUE(errors::IsNotFound(status2))
        << "Actual status: " << status2.ToString();
    EXPECT_TRUE(absl::StrContains(status2.error_message(), "Handle"));
    EXPECT_TRUE(absl::StrContains(status2.error_message(), "not found"));

    return status;
  }

  Status Run(FunctionLibraryRuntime* flr, FunctionLibraryRuntime::Handle handle,
             FunctionLibraryRuntime::Options opts, CallFrameInterface* frame) {
    std::function<void(std::function<void()>)> runner =
        [](std::function<void()> fn) {
          test::function::FunctionTestSchedClosure(fn);
        };
    opts.runner = &runner;
    Notification done;
    Status status;
    flr->Run(opts, handle, frame, [&status, &done](const Status& s) {
      status = s;
      done.Notify();
    });
    done.WaitForNotification();
    if (!status.ok()) {
      return status;
    }

    return Status::OK();
  }

  Status InstantiateAndRunViaCallFrameInterface(FunctionLibraryRuntime* flr,
                                                const string& name,
                                                test::function::Attrs attrs,
                                                const std::vector<Tensor>& args,
                                                std::vector<Tensor*> rets) {
    FunctionLibraryRuntime::Handle handle;
    Status status = flr->Instantiate(name, attrs, &handle);
    if (!status.ok()) {
      return status;
    }
    const FunctionBody* fbody = flr->GetFunctionBody(handle);
    FunctionCallFrame frame(fbody->arg_types, fbody->ret_types);
    TF_RETURN_IF_ERROR(frame.SetArgs(args));

    FunctionLibraryRuntime::Options opts;
    status = Run(flr, handle, opts, &frame);
    if (!status.ok()) return status;

    std::vector<Tensor> retvals;
    TF_RETURN_IF_ERROR(frame.GetRetvals(&retvals));
    CHECK_EQ(rets.size(), retvals.size());
    for (size_t i = 0; i < rets.size(); ++i) {
      *rets[i] = retvals[i];
    }

    // Release the handle and try running again. It should not succeed.
    status = flr->ReleaseHandle(handle);
    if (!status.ok()) return status;

    Status status2 = Run(flr, handle, opts, args, std::move(rets));
    EXPECT_TRUE(errors::IsNotFound(status2));
    EXPECT_TRUE(absl::StrContains(status2.error_message(), "Handle"));
    EXPECT_TRUE(absl::StrContains(status2.error_message(), "not found"));

    return status;
  }

  std::unique_ptr<Graph> GetFuncBody(FunctionLibraryRuntime* flr,
                                     const string& name,
                                     test::function::Attrs attrs) {
    FunctionLibraryRuntime::Handle handle;
    Status status = flr->Instantiate(name, attrs, &handle);
    if (!status.ok()) {
      LOG(ERROR) << status;
      return nullptr;
    }
    const FunctionBody* fbody = flr->GetFunctionBody(handle);
    CHECK_NOTNULL(fbody);
    std::unique_ptr<Graph> ret(new Graph(lib_def_.get()));
    CopyGraph(*fbody->graph, ret.get());
    return ret;
  }

  std::unique_ptr<Graph> GetGradBody(FunctionLibraryRuntime* flr,
                                     const string& func,
                                     test::function::Attrs attrs) {
    FunctionLibraryRuntime::Handle handle;
    Status status = flr->Instantiate(func, attrs, &handle);
    if (!status.ok()) {
      LOG(ERROR) << status;
      return nullptr;
    }
    const FunctionBody* fbody = flr->GetFunctionBody(handle);
    CHECK_NOTNULL(fbody);
    std::unique_ptr<FunctionBody> gbody(SymbolicGradient(*fbody));
    CHECK_NOTNULL(gbody);
    std::unique_ptr<Graph> ret(new Graph(lib_def_.get()));
    CopyGraph(*gbody->graph, ret.get());
    return ret;
  }

  FunctionLibraryRuntime* flr0_;
  FunctionLibraryRuntime* flr1_;
  FunctionLibraryRuntime* flr2_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  FunctionDefLibrary fdef_lib_;
};

TEST_F(FunctionLibraryRuntimeTest, IsStateful) {
  Init({});
  EXPECT_TRUE(flr0_->IsStateful("Variable"));
  EXPECT_TRUE(flr0_->IsStateful("VariableV2"));
  EXPECT_FALSE(flr0_->IsStateful("Matmul"));
}

TEST_F(FunctionLibraryRuntimeTest, XTimesTwo) {
  Init({test::function::XTimesTwo()});
  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;
  TF_CHECK_OK(
      InstantiateAndRun(flr0_, "XTimesTwo", {{"T", DT_FLOAT}}, {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
  TF_CHECK_OK(InstantiateAndRunViaCallFrameInterface(
      flr0_, "XTimesTwo", {{"T", DT_FLOAT}}, {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
}

TEST_F(FunctionLibraryRuntimeTest, InstantiationStackTraceCopying) {
  class DummyStackTrace : public AbstractStackTrace {
    absl::Span<StackFrame const> ToFrames() const override { return {}; }

    std::string ToString(const TracePrintingOptions& opts) const override {
      return "DummyStackTrace";
    }

    StackFrame LastUserFrame() const override { return StackFrame{}; }
  };

  FunctionDef func = test::function::XTimesTwo();
  Init({});

  StackTracesMap stack_traces;
  stack_traces["two"] = std::make_shared<DummyStackTrace>();

  TF_CHECK_OK(lib_def_->AddFunctionDef(func, stack_traces));

  FunctionLibraryRuntime::Handle handle;
  TF_CHECK_OK(Instantiate(flr0_, "XTimesTwo", {{"T", DT_FLOAT}}, {}, &handle));

  const FunctionBody* func_body = flr0_->GetFunctionBody(handle);
  for (const Node* node : func_body->graph->nodes()) {
    if (node->name() == "two") {
      EXPECT_EQ(node->GetStackTrace()->ToString({}), "DummyStackTrace");
    }
  }
  TF_CHECK_OK(flr0_->ReleaseHandle(handle));
}

TEST_F(FunctionLibraryRuntimeTest, XTimesTwo_MultiDeviceBacked) {
  Init({test::function::XTimesTwo()});
  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;

  FunctionLibraryRuntime::InstantiateOptions options;
  options.is_multi_device_function = true;

  TF_CHECK_OK(InstantiateAndRun(flr0_, "XTimesTwo", {{"T", DT_FLOAT}}, options,
                                {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
}

class ConsumeArgumentCallFrame : public CallFrameInterface {
 public:
  ConsumeArgumentCallFrame(Tensor* arg, Tensor* retval)
      : arg_(arg), retval_(retval) {}

  size_t num_args() const override { return 1; }
  size_t num_retvals() const override { return 1; }

  Status GetArg(int index, const Tensor** val) override {
    LOG(FATAL) << "Should not be called.";
  }

  bool CanConsumeArg(int index) const override { return index == 0; }

  void ConsumeArg(int index, Tensor* val) override { *val = std::move(*arg_); }

  Status SetRetval(int index, const Tensor& val) override {
    CHECK_EQ(index, 0);
    *retval_ = val;
    return Status::OK();
  }

 private:
  Tensor* const arg_;
  Tensor* const retval_;
};

TEST_F(FunctionLibraryRuntimeTest, XTimesTwo_ConsumeArgument_DefaultExecutor) {
  Init({test::function::XTimesTwo()});
  auto default_executor = metrics::TestDelta("flr_executor", "default");
  auto single_threaded = metrics::TestDelta("flr_executor", "single_threaded");
  FunctionLibraryRuntime::Handle handle;
  TF_CHECK_OK(flr0_->Instantiate(
      "XTimesTwo", test::function::Attrs({{"T", DT_FLOAT}}), &handle));

  auto x = test::AsTensor<float>({1, 2, 3, 4});
  float* x_base_ptr = &x.flat<float>()(0);
  Tensor y;
  ConsumeArgumentCallFrame frame(&x, &y);

  FunctionLibraryRuntime::Options opts;
  TF_CHECK_OK(Run(flr0_, handle, opts, &frame));

  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));

  // Expect that the buffer for `x` has been forwarded to and used as the buffer
  // for `y`.
  float* y_base_ptr = &y.flat<float>()(0);
  EXPECT_EQ(x_base_ptr, y_base_ptr);
  EXPECT_FALSE(x.IsInitialized());

  TF_CHECK_OK(flr0_->ReleaseHandle(handle));
  EXPECT_GT(default_executor.Get(), 0);
  EXPECT_EQ(single_threaded.Get(), 0);
}

TEST_F(FunctionLibraryRuntimeTest,
       XTimesTwo_ConsumeArgument_SingleThreadedExecutor) {
  Init({test::function::XTimesTwo()});
  auto default_executor = metrics::TestDelta("flr_executor", "default");
  auto single_threaded = metrics::TestDelta("flr_executor", "single_threaded");
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.executor_type = "SINGLE_THREADED_EXECUTOR";
  FunctionLibraryRuntime::Handle handle;
  TF_CHECK_OK(flr0_->Instantiate("XTimesTwo",
                                 test::function::Attrs({{"T", DT_FLOAT}}),
                                 instantiate_opts, &handle));

  auto x = test::AsTensor<float>({1, 2, 3, 4});
  float* x_base_ptr = &x.flat<float>()(0);
  Tensor y;
  ConsumeArgumentCallFrame frame(&x, &y);

  FunctionLibraryRuntime::Options opts;
  TF_CHECK_OK(Run(flr0_, handle, opts, &frame));

  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));

  // Expect that the buffer for `x` has been forwarded to and used as the buffer
  // for `y`.
  float* y_base_ptr = &y.flat<float>()(0);
  EXPECT_EQ(x_base_ptr, y_base_ptr);
  EXPECT_FALSE(x.IsInitialized());

  TF_CHECK_OK(flr0_->ReleaseHandle(handle));
  EXPECT_EQ(default_executor.Get(), 0);
  EXPECT_GT(single_threaded.Get(), 0);
}

TEST_F(FunctionLibraryRuntimeTest, XTimesN) {
  Init({test::function::XTimesTwo(), test::function::XTimesFour(),
        test::function::XTimes16()});
  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;
  TF_CHECK_OK(
      InstantiateAndRun(flr0_, "XTimesTwo", {{"T", DT_FLOAT}}, {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
  TF_CHECK_OK(
      InstantiateAndRun(flr0_, "XTimesFour", {{"T", DT_FLOAT}}, {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({4, 8, 12, 16}));
  TF_CHECK_OK(
      InstantiateAndRun(flr0_, "XTimes16", {{"T", DT_FLOAT}}, {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({16, 32, 48, 64}));
}

TEST_F(FunctionLibraryRuntimeTest, XTimesNInLibDef) {
  Init({});
  FunctionDefLibrary proto;
  *proto.add_function() = test::function::XTimesTwo();
  *proto.add_function() = test::function::XTimesFour();
  *proto.add_function() = test::function::XTimes16();
  std::unique_ptr<FunctionLibraryDefinition> lib_def(
      new FunctionLibraryDefinition(OpRegistry::Global(), proto));

  FunctionLibraryRuntime::InstantiateOptions options;
  options.lib_def = lib_def.get();

  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;

  // Ensure that the function is not installed in the base library.
  HasError(InstantiateAndRun(flr0_, "XTimesTwo", {{"T", DT_FLOAT}},
                             {} /* options */, {x}, {&y}),
           error::NOT_FOUND, "Function XTimesTwo is not defined.");

  TF_CHECK_OK(InstantiateAndRun(flr0_, "XTimesTwo", {{"T", DT_FLOAT}}, options,
                                {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
  TF_CHECK_OK(InstantiateAndRun(flr0_, "XTimesFour", {{"T", DT_FLOAT}}, options,
                                {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({4, 8, 12, 16}));
  TF_CHECK_OK(InstantiateAndRun(flr0_, "XTimes16", {{"T", DT_FLOAT}}, options,
                                {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({16, 32, 48, 64}));

  // Ensure that the function is still not installed in the base library.
  HasError(InstantiateAndRun(flr0_, "XTimesTwo", {{"T", DT_FLOAT}},
                             {} /* options */, {x}, {&y}),
           error::NOT_FOUND, "Function XTimesTwo is not defined.");
}

TEST_F(FunctionLibraryRuntimeTest, XTimesNInLibDefAndDelayedInstantiation) {
  using FDH = ::tensorflow::FunctionDefHelper;

  Init({});

  // Call XTimesFour via PartitionedCall which delays functions instantiation
  // to the first call to Compute/ComputeAsync.
  FunctionDef my_xt4 = FunctionDefHelper::Create(
      "MyXTimesFour", {"x:float"}, {"z:float"}, {},
      {{{"x_times_four"},
        "PartitionedCall",
        {"x"},
        {{"Tin", DataTypeSlice({DT_FLOAT})},
         {"Tout", DataTypeSlice({DT_FLOAT})},
         {"f", FDH::FunctionRef("XTimesFour", {{"T", DT_FLOAT}})}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "x_times_four:output:0"}});

  FunctionDefLibrary lib;
  *lib.add_function() = test::function::XTimesTwo();
  *lib.add_function() = test::function::XTimesFour();
  *lib.add_function() = my_xt4;
  std::unique_ptr<FunctionLibraryDefinition> lib_def(
      new FunctionLibraryDefinition(OpRegistry::Global(), lib));

  FunctionLibraryRuntime::InstantiateOptions options;
  options.lib_def = lib_def.get();

  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;

  // When we instantiate with `options` we should get x*4.
  TF_CHECK_OK(InstantiateAndRun(flr0_, "MyXTimesFour", {}, options, {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({4, 8, 12, 16}));

  // Create options that override XTimesFour body with XTimesTwo body.
  FunctionDef xt4_override = test::function::XTimesTwo();
  xt4_override.mutable_signature()->set_name("XTimesFour");
  FunctionDefLibrary lib_override;
  *lib_override.add_function() = xt4_override;
  *lib_override.add_function() = my_xt4;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_override(
      new FunctionLibraryDefinition(OpRegistry::Global(), lib_override));
  options.lib_def = lib_def_override.get();

  // When we instantiate with `options` we should get x*2.
  TF_CHECK_OK(InstantiateAndRun(flr0_, "MyXTimesFour", {}, options, {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
}

TEST_F(FunctionLibraryRuntimeTest, StateHandle) {
  auto T = DT_INT32;

  // The expected sequence of outputs from this function is [6, 4, 0, 1, ...].
  FunctionDef stateful_func = FDH::Define(
      // Name
      "RandomUniformWrapper",
      // Args
      {},
      // Return values
      {"y: int32"},
      // Attrs
      {},
      // Nodes
      {FDH::Const<int32>("shape", gtl::ArraySlice<int32>({1})),
       FDH::Const<int32>("minval", 0),
       FDH::Const<int32>("maxval", 10),
       // A stateful node.
       {{"y"},
        "RandomUniformInt",
        {"shape", "minval", "maxval"},
        {{"seed", 37}, {"seed2", 48}, {"Tout", T}, {"T", T}}}});
  Init({stateful_func});

  FunctionLibraryRuntime::Handle handle;
  TF_CHECK_OK(Instantiate(flr0_, "RandomUniformWrapper", {}, &handle));

  FunctionLibraryRuntime::Options opts;
  Tensor y;
  {
    // Simple case: instantiating with no state_handle.
    for (int32_t expected : {6, 4}) {
      TF_CHECK_OK(Run(flr0_, handle, opts, {}, {&y}));
      test::ExpectTensorEqual<int>(y, test::AsTensor<int32>({expected}));
    }
  }

  {
    // Instantiating again with no state_handle should yield the same handle and
    // the continuation of the same sequence.
    FunctionLibraryRuntime::Handle handle_non_isolated;
    TF_CHECK_OK(
        Instantiate(flr0_, "RandomUniformWrapper", {}, &handle_non_isolated));
    EXPECT_EQ(handle, handle_non_isolated);
    for (int32_t expected : {0, 1}) {
      TF_CHECK_OK(Run(flr0_, handle_non_isolated, opts, {}, {&y}));
      test::ExpectTensorEqual<int>(y, test::AsTensor<int32>({expected}));
    }
  }

  {
    // Instantiating with a given state handle will create new state and yield
    // the original sequence.
    FunctionLibraryRuntime::InstantiateOptions options;
    FunctionLibraryRuntime::Handle handle_isolated;
    options.state_handle = "handle_1";
    TF_CHECK_OK(Instantiate(flr0_, "RandomUniformWrapper", {}, options,
                            &handle_isolated));
    EXPECT_NE(handle, handle_isolated);
    for (int32_t expected : {6, 4, 0, 1}) {
      TF_CHECK_OK(Run(flr0_, handle_isolated, opts, {}, {&y}));
      test::ExpectTensorEqual<int>(y, test::AsTensor<int32>({expected}));
    }
  }

  {
    // Instantiating with a different given state handle will create new state
    // and yield the original sequence.
    FunctionLibraryRuntime::InstantiateOptions options;
    FunctionLibraryRuntime::Handle handle_isolated;
    options.state_handle = "handle_2";
    TF_CHECK_OK(Instantiate(flr0_, "RandomUniformWrapper", {}, options,
                            &handle_isolated));
    EXPECT_NE(handle, handle_isolated);
    for (int32_t expected : {6, 4, 0, 1}) {
      TF_CHECK_OK(Run(flr0_, handle_isolated, opts, {}, {&y}));
      test::ExpectTensorEqual<int>(y, test::AsTensor<int32>({expected}));
    }
  }

  {
    // Reinstantiating after releasing a handle will yield the original sequence
    // multiple times.
    FunctionLibraryRuntime::InstantiateOptions options;
    FunctionLibraryRuntime::Handle handle_isolated;
    options.state_handle = "handle_3";

    for (int i = 0; i < 2; ++i) {
      TF_CHECK_OK(Instantiate(flr0_, "RandomUniformWrapper", {}, options,
                              &handle_isolated));
      EXPECT_NE(handle, handle_isolated);
      for (int32_t expected : {6, 4, 0, 1}) {
        TF_CHECK_OK(Run(flr0_, handle_isolated, opts, {}, {&y}));
        test::ExpectTensorEqual<int>(y, test::AsTensor<int32>({expected}));
      }
      TF_CHECK_OK(flr0_->ReleaseHandle(handle_isolated));
    }
  }
}

namespace {
class DummyExecutorRegistrar {
 public:
  DummyExecutorRegistrar() {
    ExecutorFactory::Register("DUMMY", new Factory());
  }

 private:
  class Factory : public ExecutorFactory {
    Status NewExecutor(const LocalExecutorParams& params, const Graph& graph,
                       std::unique_ptr<Executor>* out_executor) override {
      return errors::Internal("This is a dummy.");
    }
  };
};
static DummyExecutorRegistrar registrar;
}  // namespace

TEST_F(FunctionLibraryRuntimeTest, ExecutorFactory) {
  Init({test::function::XTimesTwo()});

  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;

  // Test that the default executor works.
  {
    FunctionLibraryRuntime::InstantiateOptions options;
    options.executor_type = "";
    TF_CHECK_OK(InstantiateAndRun(flr0_, "XTimesTwo", {{"T", DT_FLOAT}},
                                  options, {x}, {&y}));
    test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
  }

  // Test the explicit registration for the default executor.
  {
    FunctionLibraryRuntime::InstantiateOptions options;
    options.executor_type = "DEFAULT";
    TF_CHECK_OK(InstantiateAndRun(flr0_, "XTimesTwo", {{"T", DT_FLOAT}},
                                  options, {x}, {&y}));
    test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
  }

  // Test that a non-default executor factory can be invoked.
  {
    FunctionLibraryRuntime::InstantiateOptions options;
    options.executor_type = "DUMMY";
    HasError(InstantiateAndRun(flr0_, "XTimesTwo", {{"T", DT_FLOAT}}, options,
                               {x}, {&y}),
             error::INTERNAL, "This is a dummy.");
  }

  // Test that a non-default executor factory can be invoked via an attr.
  {
    FunctionLibraryRuntime::InstantiateOptions options;
    HasError(InstantiateAndRun(flr0_, "XTimesTwo",
                               {{"T", DT_FLOAT}, {"_executor", "DUMMY"}},
                               options, {x}, {&y}),
             error::INTERNAL, "This is a dummy.");
  }

  // Test that a non-default executor factory specified via an
  // `InstantiateOptions` supersedes the attr when both are present.
  {
    FunctionLibraryRuntime::InstantiateOptions options;
    options.executor_type = "DUMMY";
    HasError(
        InstantiateAndRun(flr0_, "XTimesTwo",
                          {{"T", DT_FLOAT}, {"_executor", "UNKNOWN_EXECUTOR"}},
                          options, {x}, {&y}),
        error::INTERNAL, "This is a dummy.");
  }

  // Test that non-existent executor types trigger an error.
  {
    FunctionLibraryRuntime::InstantiateOptions options;
    options.executor_type = "UNKNOWN_EXECUTOR";
    HasError(InstantiateAndRun(flr0_, "XTimesTwo", {{"T", DT_FLOAT}}, options,
                               {x}, {&y}),
             error::NOT_FOUND,
             "No executor factory registered for the given executor "
             "type: UNKNOWN_EXECUTOR");
  }
  {
    FunctionLibraryRuntime::InstantiateOptions options;
    HasError(
        InstantiateAndRun(flr0_, "XTimesTwo",
                          {{"T", DT_FLOAT}, {"_executor", "UNKNOWN_EXECUTOR"}},
                          options, {x}, {&y}),
        error::NOT_FOUND,
        "No executor factory registered for the given executor "
        "type: UNKNOWN_EXECUTOR");
  }
}

TEST_F(FunctionLibraryRuntimeTest, ExpandInlineFunctions) {
  Init({test::function::XTimesTwo(), test::function::XTimesFour(),
        test::function::XTimes16()});
  std::unique_ptr<Graph> g = GetFuncBody(flr0_, "XTimes16", {{"T", DT_FLOAT}});
  ASSERT_TRUE(g != nullptr);

  {
    Scope s = Scope::NewRootScope();
    TF_ASSERT_OK(s.graph()->AddFunctionLibrary(fdef_lib_));
    auto arg = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto a = test::function::Call(&s, "x4", "XTimesFour", {arg});
    auto b = test::function::Call(&s, "y", "XTimesFour", {a});
    auto ret = ops::_Retval(s.WithOpName("y_RetVal"), b, 0);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }

  ExpandInlineFunctions(flr0_, g.get());
  {
    Scope s = Scope::NewRootScope();
    TF_ASSERT_OK(s.graph()->AddFunctionLibrary(fdef_lib_));
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto func0 = ops::Identity(s.WithOpName("Func/x4/input/_0"), x);
    auto x4_x2 = test::function::Call(&s, "x4/x2", "XTimesTwo", {func0});
    auto x4_y = test::function::Call(&s, "x4/y", "XTimesTwo", {x4_x2});
    auto func1 = ops::Identity(s.WithOpName("Func/x4/output/_1"), x4_y);
    auto func2 = ops::Identity(s.WithOpName("Func/y/input/_2"), func1);
    auto y_x2 = test::function::Call(&s, "y/x2", "XTimesTwo", {func2});
    auto y_y = test::function::Call(&s, "y/y", "XTimesTwo", {y_x2});
    auto func3 = ops::Identity(s.WithOpName("Func/y/output/_3"), y_y);
    auto ret = ops::_Retval(s.WithOpName("y_RetVal"), func3, 0);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }

  ExpandInlineFunctions(flr0_, g.get());
  GraphDef e2;
  {
    Scope s = Scope::NewRootScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto x4_x2_two = ops::Const<int64_t>(s.WithOpName("x4/x2/two"), int64_t{2});
    auto x4_y_two = ops::Const<int64_t>(s.WithOpName("x4/y/two"), int64_t{2});
    auto y_x2_two = ops::Const<int64_t>(s.WithOpName("y/x2/two"), int64_t{2});
    auto y_y_two = ops::Const<int64_t>(s.WithOpName("y/y/two"), int64_t{2});
    auto x4_x2_scale =
        ops::Cast(s.WithOpName("x4/x2/scale"), x4_x2_two, DT_FLOAT);
    auto x4_y_scale = ops::Cast(s.WithOpName("x4/y/scale"), x4_y_two, DT_FLOAT);
    auto y_x2_scale = ops::Cast(s.WithOpName("y/x2/scale"), y_x2_two, DT_FLOAT);
    auto y_y_scale = ops::Cast(s.WithOpName("y/y/scale"), y_y_two, DT_FLOAT);
    auto func0 = ops::Identity(s.WithOpName("Func/x4/input/_0"), x);
    auto func4 = ops::Identity(s.WithOpName("Func/x4/x2/input/_4"), func0);
    auto x4_x2_y = ops::Mul(s.WithOpName("x4/x2/y"), func4, x4_x2_scale);
    auto func5 = ops::Identity(s.WithOpName("Func/x4/x2/output/_5"), x4_x2_y);
    auto func6 = ops::Identity(s.WithOpName("Func/x4/y/input/_6"), func5);
    auto x4_y_y = ops::Mul(s.WithOpName("x4/y/y"), func6, x4_y_scale);
    auto func7 = ops::Identity(s.WithOpName("Func/x4/y/output/_7"), x4_y_y);
    auto func1 = ops::Identity(s.WithOpName("Func/x4/output/_1"), func7);
    auto func2 = ops::Identity(s.WithOpName("Func/y/input/_2"), func1);
    auto func8 = ops::Identity(s.WithOpName("Func/y/x2/input/_8"), func2);
    auto y_x2_y = ops::Mul(s.WithOpName("y/x2/y"), func8, y_x2_scale);
    auto func9 = ops::Identity(s.WithOpName("Func/y/x2/output/_9"), y_x2_y);
    auto func10 = ops::Identity(s.WithOpName("Func/y/y/input/_10"), func9);
    auto y_y_y = ops::Mul(s.WithOpName("y/y/y"), func10, y_y_scale);
    auto func11 = ops::Identity(s.WithOpName("Func/y/y/output/_11"), y_y_y);
    auto func3 = ops::Identity(s.WithOpName("Func/y/output/_3"), func11);
    auto ret = ops::_Retval(s.WithOpName("y_RetVal"), func3, 0);
    TF_ASSERT_OK(s.ToGraphDef(&e2));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(e2, actual);
  }

  // No further inlining.
  ExpandInlineFunctions(flr0_, g.get());
  {
    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(e2, actual);
  }

  // Get rid of redundant Identity nodes.
  RemoveIdentityNodes(g.get());
  {
    Scope s = Scope::NewRootScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto x4_x2_two = ops::Const<int64_t>(s.WithOpName("x4/x2/two"), int64_t{2});
    auto x4_y_two = ops::Const<int64_t>(s.WithOpName("x4/y/two"), int64_t{2});
    auto y_x2_two = ops::Const<int64_t>(s.WithOpName("y/x2/two"), int64_t{2});
    auto y_y_two = ops::Const<int64_t>(s.WithOpName("y/y/two"), int64_t{2});
    auto x4_x2_scale =
        ops::Cast(s.WithOpName("x4/x2/scale"), x4_x2_two, DT_FLOAT);
    auto x4_y_scale = ops::Cast(s.WithOpName("x4/y/scale"), x4_y_two, DT_FLOAT);
    auto y_x2_scale = ops::Cast(s.WithOpName("y/x2/scale"), y_x2_two, DT_FLOAT);
    auto y_y_scale = ops::Cast(s.WithOpName("y/y/scale"), y_y_two, DT_FLOAT);
    auto x4_x2_y = ops::Mul(s.WithOpName("x4/x2/y"), x, x4_x2_scale);
    auto x4_y_y = ops::Mul(s.WithOpName("x4/y/y"), x4_x2_y, x4_y_scale);
    auto y_x2_y = ops::Mul(s.WithOpName("y/x2/y"), x4_y_y, y_x2_scale);
    auto y_y_y = ops::Mul(s.WithOpName("y/y/y"), y_x2_y, y_y_scale);
    auto ret = ops::_Retval(s.WithOpName("y_RetVal"), y_y_y, 0);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }
}

// Verifies that control dependencies on the caller are added as control
// dependencies on any function calls created by inlining.
TEST_F(FunctionLibraryRuntimeTest, ExpandInlineFunctionsWithInputControlEdges) {
  Init({test::function::XTimesTwo(), test::function::XTimesFour()});

  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  {
    Scope s = Scope::NewRootScope();
    TF_ASSERT_OK(s.graph()->AddFunctionLibrary(fdef_lib_));
    auto a = ops::_Arg(s.WithOpName("a"), DT_FLOAT, 0);
    auto c = ops::NoOp(s.WithOpName("c"));
    auto b = test::function::Call(&s, "b", "XTimesFour", {a});
    s.graph()->AddControlEdge(c.operation.node(), b.node());
    auto ret = ops::_Retval(s.WithOpName("b_RetVal"), b, 0);
    TF_ASSERT_OK(s.ToGraph(g.get()));
  }

  ExpandInlineFunctions(flr0_, g.get());
  {
    Scope s = Scope::NewRootScope();
    TF_ASSERT_OK(s.graph()->AddFunctionLibrary(fdef_lib_));
    auto a = ops::_Arg(s.WithOpName("a"), DT_FLOAT, 0);
    auto c = ops::NoOp(s.WithOpName("c"));
    auto func0 = ops::NoOp(s.WithOpName("Func/b/input_control_node/_0")
                               .WithControlDependencies({c}));
    auto func1 = ops::Identity(
        s.WithOpName("Func/b/input/_1").WithControlDependencies({func0}), a);
    auto b_x2 = test::function::Call(&s, "b/x2", "XTimesTwo", {func1});
    s.graph()->AddControlEdge(func0.operation.node(), b_x2.node());
    auto b_y = test::function::Call(&s, "b/y", "XTimesTwo", {b_x2});
    s.graph()->AddControlEdge(func0.operation.node(), b_y.node());
    auto func2 = ops::Identity(s.WithOpName("Func/b/output/_2"), b_y);
    auto ret = ops::_Retval(s.WithOpName("b_RetVal"), func2, 0);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }

  ExpandInlineFunctions(flr0_, g.get());
  {
    Scope s = Scope::NewRootScope();
    TF_ASSERT_OK(s.graph()->AddFunctionLibrary(fdef_lib_));
    auto a = ops::_Arg(s.WithOpName("a"), DT_FLOAT, 0);
    auto c = ops::NoOp(s.WithOpName("c"));
    auto func0 = ops::NoOp(s.WithOpName("Func/b/input_control_node/_0")
                               .WithControlDependencies({c}));
    auto func1 = ops::Identity(
        s.WithOpName("Func/b/input/_1").WithControlDependencies({func0}), a);

    auto func3 = ops::NoOp(s.WithOpName("Func/b/x2/input_control_node/_3")
                               .WithControlDependencies({func0}));
    auto func4 = ops::Identity(
        s.WithOpName("Func/b/x2/input/_4").WithControlDependencies({func3}),
        func1);
    auto b_x2_two = ops::Const(
        s.WithOpName("b/x2/two").WithControlDependencies({func3}), int64_t{2});
    auto b_x2_scale = ops::Cast(s.WithOpName("b/x2/scale"), b_x2_two, DT_FLOAT);
    auto b_x2_y = ops::Mul(s.WithOpName("b/x2/y"), func4, b_x2_scale);
    auto func5 = ops::Identity(s.WithOpName("Func/b/x2/output/_5"), b_x2_y);

    auto func6 = ops::NoOp(s.WithOpName("Func/b/y/input_control_node/_6")
                               .WithControlDependencies({func0}));
    auto func7 = ops::Identity(
        s.WithOpName("Func/b/y/input/_7").WithControlDependencies({func6}),
        func5);
    auto b_y_two = ops::Const(
        s.WithOpName("b/y/two").WithControlDependencies({func6}), int64_t{2});
    auto b_y_scale = ops::Cast(s.WithOpName("b/y/scale"), b_y_two, DT_FLOAT);
    auto b_y_y = ops::Mul(s.WithOpName("b/y/y"), func7, b_y_scale);
    auto func8 = ops::Identity(s.WithOpName("Func/b/y/output/_8"), b_y_y);

    auto func2 = ops::Identity(s.WithOpName("Func/b/output/_2"), func8);
    auto ret = ops::_Retval(s.WithOpName("b_RetVal"), func2, 0);

    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }
}

TEST_F(FunctionLibraryRuntimeTest,
       ExpandInlineFunctionsWithOutputControlEdges) {
  using test::function::NDef;

  // `add` node is not required to compute regular output `o`, but it must
  // execute because it is in `control_ret`.
  const FunctionDef func =
      FDH::Create("AddAndMul", {"i: float"}, {"o: float"}, {},
                  {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
                   {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
                  /*ret_def=*/{{"o", "ret:z:0"}},
                  /*control_ret_def=*/{{"must_execute", "add"}});

  Init({func});

  // Construct a graph for the function call:
  //
  //   a = Arg[dtype=DT_FLOAT]
  //   b = AddAndMul(a)
  //   c = NoOp(^b)
  //   ret = RetVal(b, ^c)
  const auto init_graph = [this](std::unique_ptr<Graph>* g) -> void {
    *g = absl::make_unique<Graph>(OpRegistry::Global());

    Scope s = Scope::NewRootScope();
    TF_ASSERT_OK(s.graph()->AddFunctionLibrary(fdef_lib_));
    auto a = ops::_Arg(s.WithOpName("a"), DT_FLOAT, 0);
    auto b = test::function::Call(&s, "b", "AddAndMul", {a});
    auto c = ops::NoOp(s.WithOpName("c"));
    auto ret = ops::_Retval(s.WithOpName("ret"), b, 0);
    s.graph()->AddControlEdge(b.node(), c.operation.node());
    s.graph()->AddControlEdge(c.operation.node(), ret.operation.node());
    TF_ASSERT_OK(s.ToGraph(g->get()));
  };

  std::unique_ptr<Graph> g;
  ExpandInlineFunctionsOptions opts;

  const string input_node = "Func/b/input/_0";
  const string output_node = "Func/b/output/_1";
  const string output_control_node = "Func/b/output_control_node/_2";

  // Use data outputs as output control source.
  opts.native_options.output_control_src = OutputControlSrc::kDataOutputs;

  init_graph(&g);
  ExpandInlineFunctions(flr0_, g.get(), opts);
  {
    GraphDef expected = test::function::GDef(
        {NDef("a", "_Arg", {}, {{"T", DT_FLOAT}, {"index", 0}}),
         NDef(input_node, "Identity", {"a"}, {{"T", DT_FLOAT}}),
         NDef("b/add", "Add", {input_node, input_node}, {{"T", DT_FLOAT}}),
         NDef("b/ret", "Mul", {input_node, input_node}, {{"T", DT_FLOAT}}),
         NDef(output_node, "Identity", {"b/ret"}, {{"T", DT_FLOAT}}),
         NDef(output_control_node, "NoOp", {"^Func/b/output/_1"}, {}),
         NDef("c", "NoOp", {"^" + output_control_node}, {}),
         NDef("ret", "_Retval", {output_node, "^c"},
              {{"T", DT_FLOAT}, {"index", 0}})},
        {func});

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }

  // Use control outputs as output control source.
  opts.native_options.output_control_src = OutputControlSrc::kControlOutputs;

  init_graph(&g);
  ExpandInlineFunctions(flr0_, g.get(), opts);
  {
    GraphDef expected = test::function::GDef(
        {NDef("a", "_Arg", {}, {{"T", DT_FLOAT}, {"index", 0}}),
         NDef(input_node, "Identity", {"a"}, {{"T", DT_FLOAT}}),
         NDef("b/add", "Add", {input_node, input_node}, {{"T", DT_FLOAT}}),
         NDef("b/ret", "Mul", {input_node, input_node}, {{"T", DT_FLOAT}}),
         NDef(output_node, "Identity", {"b/ret"}, {{"T", DT_FLOAT}}),
         NDef(output_control_node, "NoOp", {"^b/add"}, {}),
         NDef("c", "NoOp", {"^" + output_control_node}, {}),
         NDef("ret", "_Retval", {output_node, "^c"},
              {{"T", DT_FLOAT}, {"index", 0}})},
        {func});

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }
}

TEST_F(FunctionLibraryRuntimeTest, ExpandInlineFunctionsAndKeepCallerNode) {
  using test::function::NDef;
  using KeepCallerNode = InlineFunctionBodyOptions::KeepCallerNode;

  const FunctionDef func =
      FDH::Create("AddAndMul", {"i: float"}, {"o: float"}, {},
                  {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
                   {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
                  /*ret_def=*/{{"o", "ret:z:0"}},
                  /*control_ret_def=*/{{"must_execute", "add"}});
  Init({func});

  // Construct a graph:
  //   a = Arg[dtype=DT_FLOAT]
  //   b = FunctionWithControlOutputs(a)
  auto construct_graph = [this](std::unique_ptr<Graph>* g) -> Status {
    Scope s = Scope::NewRootScope();
    TF_RETURN_IF_ERROR(s.graph()->AddFunctionLibrary(fdef_lib_));
    auto a = ops::_Arg(s.WithOpName("a"), DT_FLOAT, 0);
    auto b = test::function::Call(&s, "b", "AddAndMul", {a});
    TF_RETURN_IF_ERROR(s.ToGraph(g->get()));
    return Status::OK();
  };

  const string input_node = "Func/b/input/_0";
  const string output_node = "Func/b/output/_1";
  const string output_control_node = "Func/b/output_control_node/_2";

  // Construct expected graph after function inlining.
  auto expected_graph = [&](const NodeDef& caller) -> GraphDef {
    return test::function::GDef(
        {
            NDef("a", "_Arg", {}, {{"T", DT_FLOAT}, {"index", 0}}),
            NDef(input_node, "Identity", {"a"}, {{"T", DT_FLOAT}}),
            NDef("b/add", "Add", {input_node, input_node}, {{"T", DT_FLOAT}}),
            NDef("b/ret", "Mul", {input_node, input_node}, {{"T", DT_FLOAT}}),
            NDef(output_node, "Identity", {"b/ret"}, {{"T", DT_FLOAT}}),
            NDef(output_control_node, "NoOp", {"^b/add"}, {}),
            caller,  // Keep node in a graph with the same name as caller node.
        },
        {func});
  };

  ExpandInlineFunctionsOptions opts;
  opts.native_options.output_control_src = OutputControlSrc::kControlOutputs;

  // Keep inlined function call node fetchable.
  {
    opts.native_options.keep_caller_node = KeepCallerNode::kFetchable;

    std::unique_ptr<Graph> g = absl::make_unique<Graph>(OpRegistry::Global());
    TF_ASSERT_OK(construct_graph(&g));

    ExpandInlineFunctions(flr0_, g.get(), opts);
    GraphDef expected =
        expected_graph(/*caller=*/
                       NDef("b", "IdentityN",
                            {output_node, "^" + output_control_node},
                            {{"T", DataTypeSlice{DT_FLOAT}}}));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }

  // Keep inlined function call node targetable.
  {
    opts.native_options.keep_caller_node = KeepCallerNode::kTargetable;

    std::unique_ptr<Graph> g = absl::make_unique<Graph>(OpRegistry::Global());
    TF_ASSERT_OK(construct_graph(&g));

    ExpandInlineFunctions(flr0_, g.get(), opts);
    GraphDef expected =
        expected_graph(/*caller=*/
                       NDef("b", "NoOp", {"^" + output_control_node}, {}));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }
}

TEST_F(FunctionLibraryRuntimeTest, ExpandInlineFunctionsAndPlaceInlinedNodes) {
  using test::function::NDef;
  using KeepCallerNode = InlineFunctionBodyOptions::KeepCallerNode;

  const string arg_device = "/job:arg/replica:0/task:0/device:GPU";
  const string call_device = "/job:call/replica:0/task:1/device:GPU";
  const string body_device = "/job:body/replica:0/task:1/device:CPU";

  const FunctionDef func = FDH::Create(
      "AddFunc", {"i: float"}, {"o: float"}, {},
      {{{"ret"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}, {}, body_device}},
      /*ret_def=*/{{"o", "ret:z:0"}});
  Init({func});

  // Construct a graph:
  //   a = Arg[dtype=DT_FLOAT, _device=arg_device]
  //   b = AddFunc[_device=call_device](a)
  auto construct_graph = [&](std::unique_ptr<Graph>* g) -> Status {
    Scope s = Scope::NewRootScope();
    TF_RETURN_IF_ERROR(s.graph()->AddFunctionLibrary(fdef_lib_));
    auto a = ops::_Arg(s.WithOpName("a").WithDevice(arg_device), DT_FLOAT, 0);
    auto b = test::function::Call(&s, "b", "AddFunc", {a});
    TF_RETURN_IF_ERROR(s.ToGraph(g->get()));
    for (Node* node : (*g)->op_nodes()) {
      if (node->name() == "b") node->set_requested_device(call_device);
    }
    return Status::OK();
  };

  const string input_node = "Func/b/input/_0";
  const string output_node = "Func/b/output/_1";
  const string output_control_node = "Func/b/output_control_node/_2";

  // Construct expected graph after function inlining.
  auto expected_graph = [&](const std::vector<string>& placed) -> GraphDef {
    return test::function::GDef(
        {
            NDef("a", "_Arg", {}, {{"T", DT_FLOAT}, {"index", 0}}, placed[0]),
            NDef(input_node, "Identity", {"a"}, {{"T", DT_FLOAT}}, placed[1]),
            NDef("b/ret", "Add", {input_node, input_node}, {{"T", DT_FLOAT}},
                 placed[2]),
            NDef(output_node, "Identity", {"b/ret"}, {{"T", DT_FLOAT}},
                 placed[3]),
            NDef(output_control_node, "NoOp", {"^" + output_node}, {},
                 placed[4]),
        },
        {func});
  };

  ExpandInlineFunctionsOptions opts;
  opts.native_options.keep_caller_node = KeepCallerNode::kDoNotKeep;

  // Place only input nodes to match input device.
  {
    opts.native_options.inlined_function_body_placer =
        InlinedFunctionBodyPlacer::Default();

    auto g = absl::make_unique<Graph>(OpRegistry::Global());
    TF_ASSERT_OK(construct_graph(&g));

    ExpandInlineFunctions(flr0_, g.get(), opts);
    GraphDef expected = expected_graph({/*a*/ arg_device,       //
                                        /*input*/ arg_device,   //
                                        /*body*/ body_device,   //
                                        /*output*/ "",          //
                                        /*control_output*/ ""}  //
    );

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }

  // Place all nodes on the call node device.
  {
    opts.native_options.inlined_function_body_placer =
        InlinedFunctionBodyPlacer::SingleDevice();

    auto g = absl::make_unique<Graph>(OpRegistry::Global());
    TF_ASSERT_OK(construct_graph(&g));

    ExpandInlineFunctions(flr0_, g.get(), opts);
    GraphDef expected = expected_graph({/*a*/ arg_device,                //
                                        /*input*/ call_device,           //
                                        /*body*/ call_device,            //
                                        /*output*/ call_device,          //
                                        /*control_output*/ call_device}  //
    );

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }

  // Multi device function placement.
  {
    opts.native_options.inlined_function_body_placer =
        InlinedFunctionBodyPlacer::MultiDevice();

    auto g = absl::make_unique<Graph>(OpRegistry::Global());
    TF_ASSERT_OK(construct_graph(&g));

    const string merged_device = "/job:body/replica:0/task:1/device:CPU:*";

    ExpandInlineFunctions(flr0_, g.get(), opts);
    GraphDef expected = expected_graph({/*a*/ arg_device,                //
                                        /*input*/ arg_device,            //
                                        /*body*/ merged_device,          //
                                        /*output*/ "",                   //
                                        /*control_output*/ call_device}  //
    );

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }
}

TEST_F(FunctionLibraryRuntimeTest, PruneBody) {
  auto T = DT_INT32;
  FunctionDef stateful_func = FDH::Define(
      // Name
      "SquareAndAddOneWithStatefulNodes",
      // Args
      {"x: int32", "y: float32"},
      // Return values
      {"z: int32"},
      // Attrs
      {},
      // Nodes
      {// a = Square<T>(x)
       {{"a"}, "Square", {"x"}, {{"T", T}}},
       // 1
       FDH::Const("o", 1),
       // A bunch of extra arithmetic that y doesn't depend on
       {{"x1"}, "Add", {"o", "o"}, {{"T", T}}},
       {{"x2"}, "Mul", {"a", "x1"}, {{"T", T}}},
       {{"x3"}, "Mul", {"x1", "x2"}, {{"T", T}}},
       FDH::Const<int32>("shape", {1, 2}),
       // A stateful node.
       {{"keep_me"},
        "RandomUniform",
        {"shape"},
        {{"T", T}, {"dtype", DT_FLOAT}}},
       // z = Add<T>(a, o)
       {{"z"}, "Add", {"a", "o"}, {{"T", T}}}});
  Init({stateful_func});

  auto x = test::AsTensor<int32>({1, 2, 3, 4});
  auto y = test::AsTensor<float>({1.0, 2.0, 3.0, 4.0});
  Tensor z;

  FunctionLibraryRuntime::Handle handle;
  TF_CHECK_OK(
      Instantiate(flr0_, "SquareAndAddOneWithStatefulNodes", {}, &handle));

  StepStats stats;
  StepStatsCollector stats_collector(&stats);
  FunctionLibraryRuntime::Options opts;
  opts.stats_collector = &stats_collector;
  TF_CHECK_OK(Run(flr0_, handle, opts, {x, y}, {&z}));
  TF_CHECK_OK(flr0_->ReleaseHandle(handle));

  TF_CHECK_OK(InstantiateAndRun(flr0_, "SquareAndAddOneWithStatefulNodes", {},
                                {x, y}, {&z}));
  test::ExpectTensorEqual<int>(z, test::AsTensor<int32>({2, 5, 10, 17}));

  stats_collector.FinalizeAndSwap(&stats);

  // Note that we do not expect the nodes named "y", "x1", "x2", or "x3" to
  // execute.
  std::set<string> expected_node_names(
      {"_SOURCE", "shape", "x", "o", "a", "keep_me", "z", "z_RetVal"});
  std::set<string> executed_node_names;
  for (const auto& node_stats : stats.dev_stats()[0].node_stats()) {
    executed_node_names.insert(node_stats.node_name());
  }
  EXPECT_EQ(expected_node_names, executed_node_names);
}

TEST_F(FunctionLibraryRuntimeTest, DoNotPruneControlOutputsFromBody) {
  // `add` node is not required to compute regular output `o`, but it must
  // execute because it is in `control_ret`.
  const FunctionDef func =
      FDH::Create("FunctionWithControlOutputs", {"i: float"}, {"o: float"}, {},
                  {
                      {{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
                      {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}},
                  },
                  /*ret_def=*/{{"o", "ret:z:0"}},
                  /*control_ret_def=*/{{"must_execute", "add"}});

  Init({func});

  auto x = test::AsTensor<float>({1.25});
  Tensor z;

  FunctionLibraryRuntime::Handle handle;
  TF_CHECK_OK(Instantiate(flr1_, "FunctionWithControlOutputs", {}, &handle));

  StepStats stats;
  StepStatsCollector stats_collector(&stats);
  FunctionLibraryRuntime::Options opts;
  opts.stats_collector = &stats_collector;
  TF_CHECK_OK(Run(flr1_, handle, opts, {x}, {&z}));
  TF_CHECK_OK(flr1_->ReleaseHandle(handle));

  TF_CHECK_OK(
      InstantiateAndRun(flr1_, "FunctionWithControlOutputs", {}, {x}, {&z}));
  test::ExpectTensorEqual<float>(z, test::AsTensor<float>({1.25 * 1.25}));

  stats_collector.FinalizeAndSwap(&stats);

  std::set<string> expected_node_names(
      {"_SOURCE", "i", "add", "ret", "o_RetVal"});
  std::set<string> executed_node_names;
  for (const auto& node_stats : stats.dev_stats()[0].node_stats()) {
    executed_node_names.insert(node_stats.node_name());
  }
  EXPECT_EQ(expected_node_names, executed_node_names);
}

// Constant folding generates names using a global counter.
// This function invokes constant folding and parses the counter
// from the generated node name.
int GetConstantFoldingCounter() {
  Graph g(OpRegistry::Global());
  Scope s = Scope::NewRootScope();
  auto a = ops::Const<float>(s, {1.0}, {});
  auto b = ops::Const<float>(s, {2.0}, {});

  auto add = ops::Add(s.WithOpName("add"), a, b);
  auto send =
      ops::_Send(s.WithOpName("s1"), add, "add", "sender", 0, "receiver");

  TF_CHECK_OK(s.ToGraph(&g));
  bool was_mutated;
  ConstantFoldingOptions opt{};
  TF_CHECK_OK(
      ConstantFold(opt, nullptr, Env::Default(), nullptr, &g, &was_mutated));
  GraphDef def;
  g.ToGraphDef(&def);
  for (const NodeDef& node : def.node()) {
    if (absl::StartsWith(node.name(), "add/")) {
      std::vector<std::string> v = absl::StrSplit(node.name(), "__cf__");
      CHECK_GT(v.size(), 1);
      int counter;
      CHECK(absl::SimpleAtoi(v[v.size() - 1], &counter));
      return counter;
    }
  }
  LOG(FATAL) << "Should have found a node that replaced add";
}

TEST_F(FunctionLibraryRuntimeTest, OptimizeGraph) {
  Init({test::function::XTimesTwo(), test::function::XTimesFour(),
        test::function::XTimes16()});
  std::unique_ptr<Graph> g = GetFuncBody(flr0_, "XTimes16", {{"T", DT_FLOAT}});
  ASSERT_TRUE(g != nullptr);
  ExpandInlineFunctions(flr0_, g.get());
  int cf_counter = GetConstantFoldingCounter();
  OptimizeGraph(flr0_, &g);
  {
    Scope s = Scope::NewRootScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto x4_x2_scale = ops::Const<float>(
        s.WithOpName("x4/x2/scale/_12__cf__" + std::to_string(cf_counter + 1))
            .WithDevice("/job:localhost/replica:0/task:0/device:CPU:0"),
        2.0f);
    auto x4_x2_y = ops::Mul(s.WithOpName("x4/x2/y"), x, x4_x2_scale);
    auto x4_y_y = ops::Mul(s.WithOpName("x4/y/y"), x4_x2_y, x4_x2_scale);
    auto y_x2_y = ops::Mul(s.WithOpName("y/x2/y"), x4_y_y, x4_x2_scale);
    auto y_y_y = ops::Mul(s.WithOpName("y/y/y"), y_x2_y, x4_x2_scale);
    auto ret = ops::_Retval(s.WithOpName("y_RetVal"), y_y_y, 0);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }
}

TEST_F(FunctionLibraryRuntimeTest, ManySwapsNodeDef) {
  auto func = FDH::Create(  // Creates a FunctionDef using NodeDefs
                            // Name
      "ManySwapsNodeDef",
      // Input
      {"x: float", "y: float"},
      // Output
      {"o: float"},
      // Attr
      {},
      // Nodes
      {{{"a"}, "Swap", {"x", "y"}, {{"T", DT_FLOAT}}},
       {{"b"}, "Swap", {"a:o0", "a:o1"}, {{"T", DT_FLOAT}}},
       {{"c"}, "Swap", {"b:o0", "b:o1"}, {{"T", DT_FLOAT}}},
       {{"d"}, "Swap", {"c:o0", "c:o1"}, {{"T", DT_FLOAT}}},
       {{"e"}, "Swap", {"d:o0", "d:o1"}, {{"T", DT_FLOAT}}},
       {{"f"}, "Swap", {"e:o0", "e:o1"}, {{"T", DT_FLOAT}}},
       {{"g"}, "Identity", {"f:o0"}, {{"T", DT_FLOAT}}}},
      // Return
      {{"o", "g:output"}});
  Init({test::function::Swap(), func});
  std::unique_ptr<Graph> g = GetFuncBody(flr0_, "ManySwapsNodeDef", {});
  ASSERT_TRUE(g != nullptr);
  OptimizeGraph(flr0_, &g);
  const char* e0 = R"P(
(n2:float, n3:float) -> (n2:float) {
}
)P";
  EXPECT_EQ(e0, DebugString(g.get()));
}

TEST_F(FunctionLibraryRuntimeTest, ControlDeps) {
  auto func = FDH::Create(
      // Name
      "ManySwapsFirst",
      // Args
      {"x: float", "y: float"},
      // Return values
      {"o: float"},
      // attr def
      {},
      // Nodes
      //
      // o = x*x + y*y.  Furthermore, The 1st swap depends on x2, and
      // y2 depends on the 2nd swap.  The 2nd swap has data dependency
      // on the 1st swap. The optimization should maintain the control
      // dependencies.
      {{{"a0"}, "Swap", {"x", "y"}, {{"T", DT_FLOAT}}, {"x2"}},
       {{"a1"}, "Swap", {"a0:o0:0", "a0:o1:0"}, {{"T", DT_FLOAT}}},
       {{"x2"}, "Mul", {"x", "x"}, {{"T", DT_FLOAT}}},
       {{"y2"}, "Mul", {"y", "y"}, {{"T", DT_FLOAT}}, {"a1"}},
       {{"o"}, "Add", {"x2:z:0", "y2:z:0"}, {{"T", DT_FLOAT}}}},
      {{"o", "o:z:0"}});
  Init({test::function::Swap(), func});
  std::unique_ptr<Graph> g = GetFuncBody(flr0_, "ManySwapsFirst", {});
  ASSERT_TRUE(g != nullptr);
  OptimizeGraph(flr0_, &g);

  // NOTE: We can remove func0, func1, func2, func9 with a control edge
  // n8->n5. But we don't have a pass doing that.
  {
    Scope s = Scope::NewRootScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto y = ops::_Arg(s.WithOpName("y"), DT_FLOAT, 1);
    auto x2 = ops::Mul(s.WithOpName("x2"), x, x);
    auto func0 = ops::NoOp(s.WithOpName("Func/a0/input_control_node/_0")
                               .WithControlDependencies(x2));
    auto func1 = ops::Identity(
        s.WithOpName("Func/a0/input/_1").WithControlDependencies({func0}), x);
    auto func2 = ops::Identity(
        s.WithOpName("Func/a0/input/_2").WithControlDependencies({func0}), y);
    auto func9 = ops::NoOp(
        s.WithOpName("Func/a1/output_control_node/_9")
            .WithControlDependencies({func1.output.op(), func2.output.op()}));
    auto y2 =
        ops::Mul(s.WithOpName("y2").WithControlDependencies({func9}), y, y);
    auto o = ops::Add(s.WithOpName("o"), x2, y2);
    auto ret = ops::_Retval(s.WithOpName("o_RetVal"), o, 0);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }
}

TEST_F(FunctionLibraryRuntimeTest, Error_NotFound) {
  Init({test::function::XTimesTwo(), test::function::XTimesFour()});
  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;
  HasError(InstantiateAndRun(flr0_, "Foo", {{"T", DT_FLOAT}}, {x}, {&y}),
           error::NOT_FOUND, "Function Foo is not defined.");
}

TEST_F(FunctionLibraryRuntimeTest, Error_InstantiationError) {
  auto bad_x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"y"}, "Add", {"x", "x"}, {{"no_T", "$T"}}},
      });
  Init({bad_x_times_two, test::function::XTimesFour(),
        test::function::XTimes16()});

  // Instantiating "XTimesTwo" should fail.
  FunctionLibraryRuntime::Handle handle;
  HasError(flr0_->Instantiate(
               "XTimesTwo", test::function::Attrs({{"T", DT_FLOAT}}), &handle),
           error::NOT_FOUND, "type attr not found");

  // But XTimesFour and XTimes16 instantiation should succeed. Only
  // when they run, they fail because XTimesTwo is bad.
  TF_CHECK_OK(flr0_->Instantiate(
      "XTimesFour", test::function::Attrs({{"T", DT_FLOAT}}), &handle));
  TF_CHECK_OK(flr0_->Instantiate(
      "XTimes16", test::function::Attrs({{"T", DT_FLOAT}}), &handle));

  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;
  HasError(InstantiateAndRun(flr0_, "XTimes16", {{"T", DT_FLOAT}}, {x}, {&y}),
           error::NOT_FOUND, "type attr not found");
}

TEST_F(FunctionLibraryRuntimeTest, Error_BadControlFlow) {
  Init({test::function::InvalidControlFlow()});
  auto x = test::AsTensor<int32>({0});
  DCHECK_EQ(x.dtype(), DT_INT32);
  Tensor y;
  HasError(InstantiateAndRun(flr0_, "InvalidControlFlow", {}, {x}, {&y}),
           error::INVALID_ARGUMENT,
           "{{node add}} has inputs from different frames. The input"
           " {{node enter}} is in frame 'while'. The input {{node i}} is in"
           " frame ''.");
}

TEST_F(FunctionLibraryRuntimeTest, Gradient_XTimesTwo) {
  Init({test::function::XTimesTwo(), test::function::XTimesFour(),
        test::function::XTimes16()});
  std::unique_ptr<Graph> f = GetFuncBody(flr0_, "XTimesTwo", {{"T", DT_FLOAT}});
  {
    Scope s = Scope::NewRootScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto two = ops::Const(s.WithOpName("two"), int64_t{2});
    auto scale = ops::Cast(s.WithOpName("scale"), two, DT_FLOAT);
    auto y = ops::Mul(s.WithOpName("y"), x, scale);
    auto ret = ops::_Retval(s.WithOpName("y_RetVal"), y, 0);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    f->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }

  std::unique_ptr<Graph> g = GetGradBody(flr0_, "XTimesTwo", {{"T", DT_FLOAT}});

  {
    Scope s = Scope::NewRootScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto func0 = ops::_Arg(s.WithOpName("Func/_0"), DT_FLOAT, 1);
    auto two = ops::Const(s.WithOpName("two"), int64_t{2});
    auto scale = ops::Cast(s.WithOpName("scale"), two, DT_FLOAT);
    auto y = ops::Mul(s.WithOpName("y"), x, scale);
    NameAttrList fn0;
    fn0.set_name("Mul");
    (*fn0.mutable_attr())["T"].set_type(DT_FLOAT);
    auto func1 = ops::SymbolicGradient(
        s.WithOpName("Func/_1"), std::initializer_list<Input>{x, scale, func0},
        {DT_FLOAT, DT_FLOAT}, fn0);
    NameAttrList fn1;
    fn1.set_name("Cast");
    (*fn1.mutable_attr())["SrcT"].set_type(DT_INT64);
    (*fn1.mutable_attr())["DstT"].set_type(DT_FLOAT);
    (*fn1.mutable_attr())["Truncate"].set_b(false);
    auto func2 = ops::SymbolicGradient(
        s.WithOpName("Func/_2"),
        std::initializer_list<Input>{two, func1.output[1]}, {DT_INT64}, fn1);
    auto func3 = ops::_Retval(s.WithOpName("Func/_3"), func1[0], 0);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }

  int cf_counter = GetConstantFoldingCounter();
  OptimizeGraph(flr0_, &g);
  {
    Scope s = Scope::NewRootScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto func0 = ops::_Arg(s.WithOpName("Func/_0"), DT_FLOAT, 1);
    auto scale = ops::Const(
        s.WithOpName("scale/_6__cf__" + std::to_string(cf_counter + 2))
            .WithDevice("/job:localhost/replica:0/task:0/device:CPU:0"),
        2.0f);
    auto func1_gx = ops::Mul(s.WithOpName("Func/_1/gx"), func0, scale);
    auto func1_sx = ops::Shape(s.WithOpName("Func/_1/sx"), x);
    auto const0 = ops::Const(
        s.WithOpName("Func/_1/sy/_5__cf__" + std::to_string(cf_counter + 1))
            .WithDevice("/job:localhost/replica:0/task:0/device:CPU:0"),
        0, {0});
    auto func1_rx = ops::internal::BroadcastGradientArgs(
        s.WithOpName("Func/_1/rx"), func1_sx, const0);
    auto func1_sum_gx =
        ops::Sum(s.WithOpName("Func/_1/sum_gx"), func1_gx, func1_rx.r0);
    auto func1_dx =
        ops::Reshape(s.WithOpName("Func/_1/dx"), func1_sum_gx, func1_sx);
    auto func2 = ops::_Retval(s.WithOpName("Func/_3"), func1_dx, 0);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }
}

TEST_F(FunctionLibraryRuntimeTest, Gradient_Select) {
  FunctionDef my_select = FunctionDefHelper::Create(
      "MySelect",
      // Args
      {"condition: bool", "t: float32", "e: float32"},
      // Return values
      {"z: float32"},
      // Attrs
      {},
      // Nodes
      {
          {{"select0"}, "Select", {"condition", "t", "e"}, {{"T", DT_FLOAT}}},
          {{"select1"}, "Select", {"condition", "t", "e"}, {{"T", DT_FLOAT}}},
          {{"add"},
           "Add",
           {"select0:output", "select1:output"},
           {{"T", DT_FLOAT}}},
      },
      // Output mapping
      {{"z", "add:z"}});
  FunctionDef select_grad = FunctionDefHelper::Create(
      "MySelectGrad",
      // Args
      {"condition: bool", "t:float32", "e: float32", "dz: float32"},
      // Return values
      {"dt: float32"},
      // Attrs
      {},
      // Nodes
      {{
          {"grad"},
          "SymbolicGradient",
          {"condition", "t", "e", "dz"},
          {
              {"f", FunctionDefHelper::FunctionRef("MySelect")},
              {"Tin", DataTypeSlice({DT_BOOL, DT_FLOAT, DT_FLOAT, DT_FLOAT})},
              {"Tout", DataTypeSlice({DT_BOOL, DT_FLOAT, DT_FLOAT})},
          },
      }},
      // Output mapping
      {{"dt", "grad:output:1"}});
  Init({my_select, select_grad});

  auto condition = test::AsTensor<bool>({false});
  auto t = test::AsTensor<float>({13.0});
  auto e = test::AsTensor<float>({15.0});
  auto dz = test::AsTensor<float>({1.0});
  Tensor y;
  TF_EXPECT_OK(InstantiateAndRun(flr0_, "MySelectGrad", {},
                                 {condition, t, e, dz}, {&y}));
}

TEST_F(FunctionLibraryRuntimeTest, Gradient_Add) {
  Init({});
  auto T = DT_FLOAT;
  std::unique_ptr<Graph> g = GetFuncBody(
      flr0_, "SymbolicGradient", {{"f", FDH::FunctionRef("Add", {{"T", T}})}});
  {
    Scope s = Scope::NewRootScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto y = ops::_Arg(s.WithOpName("y"), DT_FLOAT, 1);
    auto dz = ops::_Arg(s.WithOpName("dz"), DT_FLOAT, 2);
    auto gx = ops::Identity(s.WithOpName("gx"), dz);
    auto gy = ops::Identity(s.WithOpName("gy"), dz);
    auto sx = ops::Shape(s.WithOpName("sx"), x);
    auto sy = ops::Shape(s.WithOpName("sy"), y);
    auto rx = ops::internal::BroadcastGradientArgs(s.WithOpName("rx"), sx, sy);
    auto sum_gx = ops::Sum(s.WithOpName("sum_gx"), gx, rx.r0);
    auto sum_gy = ops::Sum(s.WithOpName("sum_gy"), gy, rx.r1);
    auto dx = ops::Reshape(s.WithOpName("dx"), sum_gx, sx);
    auto dy = ops::Reshape(s.WithOpName("dy"), sum_gy, sy);
    auto dx_ret = ops::_Retval(s.WithOpName("dx_RetVal"), dx, 0);
    auto dy_ret = ops::_Retval(s.WithOpName("dy_RetVal"), dy, 1);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }
}

TEST_F(FunctionLibraryRuntimeTest, Gradient_Mul) {
  Init({});
  auto T = DT_FLOAT;
  std::unique_ptr<Graph> g = GetFuncBody(
      flr0_, "SymbolicGradient", {{"f", FDH::FunctionRef("Mul", {{"T", T}})}});
  {
    Scope s = Scope::NewRootScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto y = ops::_Arg(s.WithOpName("y"), DT_FLOAT, 1);
    auto dz = ops::_Arg(s.WithOpName("dz"), DT_FLOAT, 2);
    auto gx = ops::Mul(s.WithOpName("gx"), dz, y);
    auto sx = ops::Shape(s.WithOpName("sx"), x);
    auto gy = ops::Mul(s.WithOpName("gy"), x, dz);
    auto sy = ops::Shape(s.WithOpName("sy"), y);
    auto rx = ops::internal::BroadcastGradientArgs(s.WithOpName("rx"), sx, sy);
    auto sum_gx = ops::Sum(s.WithOpName("sum_gx"), gx, rx.r0);
    auto sum_gy = ops::Sum(s.WithOpName("sum_gy"), gy, rx.r1);
    auto dx = ops::Reshape(s.WithOpName("dx"), sum_gx, sx);
    auto dy = ops::Reshape(s.WithOpName("dy"), sum_gy, sy);
    auto dx_ret = ops::_Retval(s.WithOpName("dx_RetVal"), dx, 0);
    auto dy_ret = ops::_Retval(s.WithOpName("dy_RetVal"), dy, 1);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }
}

TEST_F(FunctionLibraryRuntimeTest, Gradient_AddSum) {
  // Sum(Add(x, y))
  auto T = DT_FLOAT;
  auto test = FDH::Define("Test", {"x:float", "y:float"}, {"l:float"}, {},
                          {
                              {{"z"}, "Add", {"x", "y"}, {{"T", T}}},
                              FDH::Const("zero", 0),
                              FDH::Const("one", 1),
                              {{"r"}, "Rank", {"z"}, {{"T", T}}},
                              {{"indices"}, "Range", {"zero", "r", "one"}},
                              {{"l"}, "Sum", {"z", "indices"}, {{"T", T}}},
                          });

  // TestGrad = Test'(x, y)
  auto grad = FDH::Define("TestGrad", {"x:float", "y:float"},
                          {"dx:float", "dy:float"}, {},
                          {FDH::Const<float>("dz", 1),
                           {{"grad0", "grad1"},
                            "SymbolicGradient",
                            {"x", "y", "dz"},
                            {
                                {"f", FDH::FunctionRef("Test")},
                                {"Tin", DataTypeSlice{T, T, T}},
                                {"Tout", DataTypeSlice{T, T}},
                            }},
                           {{"dx"}, "Identity", {"grad0"}, {{"T", DT_FLOAT}}},
                           {{"dy"}, "Identity", {"grad1"}, {{"T", DT_FLOAT}}}});

  Init({test, grad});

  std::unique_ptr<Graph> g = GetFuncBody(flr0_, "TestGrad", {});
  ASSERT_TRUE(g != nullptr);
  {
    Scope s = Scope::NewRootScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto y = ops::_Arg(s.WithOpName("y"), DT_FLOAT, 1);
    auto dz = ops::Const(s.WithOpName("dz"), 1.0f);
    NameAttrList fn;
    fn.set_name("Test");
    auto grad0 = ops::SymbolicGradient(s.WithOpName("grad0"),
                                       std::initializer_list<Input>{x, y, dz},
                                       {DT_FLOAT, DT_FLOAT}, fn);
    auto dx = ops::Identity(s.WithOpName("dx"), grad0[0]);
    auto dy = ops::Identity(s.WithOpName("dy"), grad0[1]);
    auto dx_retval = ops::_Retval(s.WithOpName("dx_RetVal"), dx, 0);
    auto dy_retval = ops::_Retval(s.WithOpName("dy_RetVal"), dy, 1);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }

  ExpandInlineFunctions(flr0_, g.get());
  {
    Scope s = Scope::NewRootScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto y = ops::_Arg(s.WithOpName("y"), DT_FLOAT, 1);
    auto dz = ops::Const(s.WithOpName("dz"), 1.0f);
    auto grad0_zero = ops::Const(s.WithOpName("grad0/zero"), 0);
    auto grad0_one = ops::Const(s.WithOpName("grad0/one"), 1);
    auto func0 = ops::Identity(s.WithOpName("Func/grad0/input/_0"), x);
    auto func1 = ops::Identity(s.WithOpName("Func/grad0/input/_1"), y);
    auto func2 = ops::Identity(s.WithOpName("Func/grad0/input/_2"), dz);
    auto grad0_z = ops::Add(s.WithOpName("grad0/z"), func0, func1);
    auto grad0_r = ops::Rank(s.WithOpName("grad0/r"), grad0_z);
    auto grad0_indices = ops::Range(s.WithOpName("grad0/indices"), grad0_zero,
                                    grad0_r, grad0_one);
    auto grad0_l = ops::Sum(s.WithOpName("grad0/l"), grad0_z, grad0_indices);

    NameAttrList sum;
    sum.set_name("Sum");
    (*sum.mutable_attr())["T"].set_type(DT_FLOAT);
    (*sum.mutable_attr())["Tidx"].set_type(DT_INT32);
    (*sum.mutable_attr())["keep_dims"].set_b(false);
    auto grad0_func1 = ops::SymbolicGradient(
        s.WithOpName("grad0/Func/_1"),
        std::initializer_list<Input>{grad0_z, grad0_indices, func2},
        {DT_FLOAT, DT_INT32}, sum);

    auto grad0_func2 =
        ops::ZerosLike(s.WithOpName("grad0/Func/_2"), grad0_zero);
    auto grad0_func3 = ops::ZerosLike(s.WithOpName("grad0/Func/_3"), grad0_r);
    auto grad0_func4 = ops::ZerosLike(s.WithOpName("grad0/Func/_4"), grad0_one);

    NameAttrList add;
    add.set_name("Add");
    (*add.mutable_attr())["T"].set_type(DT_FLOAT);
    auto grad0_func5 = ops::SymbolicGradient(
        s.WithOpName("grad0/Func/_5"),
        std::initializer_list<Input>{func0, func1, grad0_func1[0]},
        {DT_FLOAT, DT_FLOAT}, add);

    auto func3 =
        ops::Identity(s.WithOpName("Func/grad0/output/_3"), grad0_func5[0]);
    auto func4 =
        ops::Identity(s.WithOpName("Func/grad0/output/_4"), grad0_func5[1]);
    auto dx = ops::Identity(s.WithOpName("dx"), func3);
    auto dy = ops::Identity(s.WithOpName("dy"), func4);
    auto dx_retval = ops::_Retval(s.WithOpName("dx_RetVal"), dx, 0);
    auto dy_retval = ops::_Retval(s.WithOpName("dy_RetVal"), dy, 1);

    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    g->ToGraphDef(&actual);
    TF_EXPECT_GRAPH_EQ(expected, actual);
  }

  OptimizeGraph(flr0_, &g);
  {
    Scope s = Scope::NewRootScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_FLOAT, 0);
    auto y = ops::_Arg(s.WithOpName("y"), DT_FLOAT, 1);
    auto dz = ops::Const(s.WithOpName("dz"), 1.0f);
    auto grad0_zero = ops::Const(s.WithOpName("grad0/zero"), 0);
    auto grad0_one = ops::Const(s.WithOpName("grad0/one"), 1);
    auto grad0_z = ops::Add(s.WithOpName("grad0/z"), x, y);
    auto grad0_r = ops::Rank(s.WithOpName("grad0/r"), grad0_z);
    auto grad0_indices = ops::Range(s.WithOpName("grad0/indices"), grad0_zero,
                                    grad0_r, grad0_one);
    auto i_shape =
        ops::Shape(s.WithOpName("grad0/Func/_1/i_shape"), grad0_indices);
    auto stitch_val = ops::Fill(s.WithOpName("grad0/Func/_1/stitch_val1"),
                                i_shape, grad0_one);
    auto x_shape = ops::Shape(s.WithOpName("grad0/Func/_1/x_shape"), grad0_z);
    auto y_shape = ops::DynamicStitch(
        s.WithOpName("grad0/Func/_1/y_shape"),
        std::initializer_list<Input>{grad0_indices, grad0_indices},
        std::initializer_list<Input>{x_shape, stitch_val});
    auto dy_reshaped =
        ops::Reshape(s.WithOpName("grad0/Func/_1/dy_reshaped"), dz, y_shape);
    auto tile_scaling =
        ops::Div(s.WithOpName("grad0/Func/_1/tile_scaling"), x_shape, y_shape);
    auto func1_dx =
        ops::Tile(s.WithOpName("grad0/Func/_1/dx"), dy_reshaped, tile_scaling);

    auto sx = ops::Shape(s.WithOpName("grad0/Func/_3/sx"), x);
    auto sy = ops::Shape(s.WithOpName("grad0/Func/_3/sy"), y);
    auto rx = ops::internal::BroadcastGradientArgs(
        s.WithOpName("grad0/Func/_3/rx"), sx, sy);
    auto sum_gx =
        ops::Sum(s.WithOpName("grad0/Func/_3/sum_gx"), func1_dx, rx.r0);
    auto sum_gy =
        ops::Sum(s.WithOpName("grad0/Func/_3/sum_gy"), func1_dx, rx.r1);
    auto dx = ops::Reshape(s.WithOpName("grad0/Func/_3/dx"), sum_gx, sx);
    auto dy = ops::Reshape(s.WithOpName("grad0/Func/_3/dy"), sum_gy, sy);

    auto dx_retval = ops::_Retval(s.WithOpName("dx_RetVal"), dx, 0);
    auto dy_retval = ops::_Retval(s.WithOpName("dy_RetVal"), dy, 1);

    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));

    GraphDef actual;
    g->ToGraphDef(&actual);
    // The optimizer is non-deterministic, so we only check that the number of
    // nodes is not greater than expected.
    EXPECT_LE(actual.node_size(), expected.node_size());
  }
}

TEST_F(FunctionLibraryRuntimeTest, CrossDevice) {
  Init({test::function::FindDevice()});
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/device:CPU:1";
  FunctionLibraryRuntime::Handle handle;
  TF_CHECK_OK(Instantiate(flr0_, "FindDevice", {}, instantiate_opts, &handle));

  Tensor y;
  FunctionLibraryRuntime::Options opts;
  PrivateIntraProcessRendezvous rendezvous(device_mgr_.get());
  opts.rendezvous = &rendezvous;
  opts.source_device = "/device:CPU:1";
  // Run on flr1_, flr2_ and make sure that the device it ran on was cpu:1.
  TF_CHECK_OK(Run(flr1_, handle, opts, {}, {&y}));
  test::ExpectTensorEqual<tstring>(
      y,
      test::AsTensor<tstring>({"/job:localhost/replica:0/task:0/device:CPU:1"},
                              TensorShape({})));
  opts.remote_execution = true;
  opts.source_device = "/job:localhost/replica:0/task:0/cpu:2";
  TF_CHECK_OK(Run(flr2_, handle, opts, {}, {&y}));
  test::ExpectTensorEqual<tstring>(
      y,
      test::AsTensor<tstring>({"/job:localhost/replica:0/task:0/device:CPU:1"},
                              TensorShape({})));
}

class AreAllKernelsInlineOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* ctx) override {
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output));
    output->scalar<bool>()() = ctx->run_all_kernels_inline();
  }
};

REGISTER_OP("AreAllKernelsInline").Output("result : bool").SetIsStateful();
REGISTER_KERNEL_BUILDER(Name("AreAllKernelsInline").Device(DEVICE_CPU),
                        AreAllKernelsInlineOp);

TEST_F(FunctionLibraryRuntimeTest, RunAllKernelsInline) {
  // Create a function "F" that includes an AreAllKernelsInline op, and a
  // function "G" that calls "F".
  auto f = FDH::Create(
      // Name
      "F",
      // Args
      {},
      // Return values
      {"ret: bool"},
      // Attrs
      {},
      // Nodes
      {// y = AreAllKernelsInline()
       {{"y"}, "AreAllKernelsInline", {}, {}}},
      {{"ret", "y:result:0"}});

  auto g = FDH::Create(
      // Name
      "G",
      // Args
      {},
      // Return values
      {"ret: bool"},
      // Attrs
      {},
      // Nodes
      {// y = F()
       {{"y"}, "F", {}, {}}},
      {{"ret", "y:ret:0"}});

  Init({f, g});
  FunctionLibraryRuntime::Handle handle;
  TF_CHECK_OK(Instantiate(flr0_, "G", {}, &handle));

  // Test that the `run_all_kernels_inline` flag is inherited by the kernel
  // running inside the called function.
  for (bool inline_option : {false, true}) {
    FunctionLibraryRuntime::Options opts;
    opts.run_all_kernels_inline = inline_option;
    Tensor result;
    TF_ASSERT_OK(Run(flr0_, handle, opts, {}, {&result}));
    EXPECT_EQ(result.scalar<bool>()(), inline_option);
  }
}

class UserIntraOpThreadPoolOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  class DummyThreadPool : public thread::ThreadPoolInterface {
   public:
    void Schedule(std::function<void()> fn) override { fn(); }
    int NumThreads() const override { return 1; }
    int CurrentThreadId() const override { return -1; }
  };

  static DummyThreadPool& dummy_thread_pool() {
    static DummyThreadPool& thread_pool = *new DummyThreadPool();
    return thread_pool;
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* result;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &result));
    result->scalar<bool>()() =
        ctx->device()
            ->tensorflow_cpu_worker_threads()
            ->workers->AsEigenThreadPool() == &dummy_thread_pool();
  }
};

REGISTER_OP("UserIntraOpThreadPool").Output("result: bool").SetIsStateful();
REGISTER_KERNEL_BUILDER(Name("UserIntraOpThreadPool").Device(DEVICE_CPU),
                        UserIntraOpThreadPoolOp);

TEST_F(FunctionLibraryRuntimeTest, RunUserIntraOpThreadPool) {
  // Create a function "F" that includes an AreAllKernelsInline op, and a
  // function "G" that calls "F".
  auto f = FDH::Create(
      // Name
      "F",
      // Args
      {},
      // Return values
      {"ret: bool"},
      // Attrs
      {},
      // Nodes
      {// y = UserIntraOpThreadPool()
       {{"y"}, "UserIntraOpThreadPool", {}, {}}},
      {{"ret", "y:result:0"}});

  Init({f});
  FunctionLibraryRuntime::Handle handle;
  TF_CHECK_OK(Instantiate(flr0_, "F", {}, &handle));

  FunctionLibraryRuntime::Options opts;
  opts.user_intra_op_threadpool = &UserIntraOpThreadPoolOp::dummy_thread_pool();

  Tensor result;
  TF_ASSERT_OK(Run(flr0_, handle, opts, {}, {&result}));
  EXPECT_TRUE(result.scalar<bool>()());
}

namespace {

bool DoNothing(Graph* g) { return false; }

GraphDef Optimize(const std::function<bool(Graph* g)>& pass,
                  const FunctionDef& fdef) {
  InstantiationResult result;
  TF_CHECK_OK(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result));
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = false;
  TF_CHECK_OK(ConvertNodeDefsToGraph(opts, result.nodes, g.get()));
  pass(g.get());
  std::unique_ptr<Graph> g1(new Graph(OpRegistry::Global()));
  CopyGraph(*g, g1.get());
  g = nullptr;
  GraphDef gdef;
  g1->ToGraphDef(&gdef);
  return gdef;
}

}  // end namespace

TEST(OptimizationTest, RemoveDeadNodes) {
  auto T = DT_INT32;
  auto func = FDH::Define(
      // Name
      "F",
      // Args
      {"x: int32"},
      // Return values
      {"y: int32"},
      // Attrs
      {},
      // Nodes
      {// a = Square<T>(x)
       {{"a"}, "Square", {"x"}, {{"T", T}}},
       // 1
       FDH::Const("o", 1),
       // A bunch of extra arithmetic that y doesn't depend on
       {{"x1"}, "Add", {"o", "o"}, {{"T", T}}},
       {{"x2"}, "Mul", {"a", "x1"}, {{"T", T}}},
       {{"x3"}, "Mul", {"x1", "x2"}, {{"T", T}}},
       // A stateful node.
       {{"keep_me"}, "RandomUniform", {"o"}, {{"T", T}, {"dtype", DT_FLOAT}}},
       // y = Add<T>(a, o)
       {{"y"}, "Add", {"a", "o"}, {{"T", T}}}});

  GraphDef expected;
  {
    Scope s = Scope::DisabledShapeInferenceScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_INT32, 0);
    auto o = ops::Const(s.WithOpName("o"), 1);
    auto keep_me = ops::RandomUniform(s.WithOpName("keep_me"), {o}, DT_FLOAT);
    auto x1 = ops::Add(s.WithOpName("x1"), o, o);
    auto a = ops::Square(s.WithOpName("a"), x);
    auto y = ops::Add(s.WithOpName("y"), a, o);
    auto x2 = ops::Mul(s.WithOpName("x2"), a, x1);
    auto x3 = ops::Mul(s.WithOpName("x3"), x1, x2);
    auto ret = ops::_Retval(s.WithOpName("y_RetVal"), y, 0);
    TF_ASSERT_OK(s.ToGraphDef(&expected));
  }
  TF_EXPECT_GRAPH_EQ(expected, Optimize(DoNothing, func));

  // TODO(zhifengc): Comes up another test case.
  TF_EXPECT_GRAPH_EQ(expected, Optimize(::tensorflow::RemoveDeadNodes, func));
}

TEST(OptimizationTest, RemoveIdentityNodes_Ref) {
  auto T = DT_FLOAT;
  auto func = FDH::Define(
      // Name
      "F",
      // Args
      {},
      // Return values
      {"ret: float"},
      // Attrs
      {},
      // Nodes
      {// variable
       {{"v"}, "VariableV2", {}, {{"dtype", T}, {"shape", TensorShape({})}}},
       // read the variable. Shouldn't be removed.
       {{"v_read"}, "Identity", {"v"}, {{"T", T}}},
       // returns v + v
       {{"ret"}, "Add", {"v_read", "v_read"}, {{"T", T}}}});

  GraphDef expected;
  {
    Scope s = Scope::NewRootScope();
    auto v = ops::Variable(s.WithOpName("v"), PartialTensorShape({}), DT_FLOAT);
    auto v_read = ops::Identity(s.WithOpName("v_read"), v);
    auto ret = ops::Add(s.WithOpName("ret"), v_read, v_read);
    auto ret_retval = ops::_Retval(s.WithOpName("ret_RetVal"), ret, 0);
    TF_ASSERT_OK(s.ToGraphDef(&expected));
  }
  TF_EXPECT_GRAPH_EQ(expected, Optimize(DoNothing, func));
  TF_EXPECT_GRAPH_EQ(expected,
                     Optimize(::tensorflow::RemoveIdentityNodes, func));
}

TEST(OptimizationTest, RemoveIdentityNodes) {
  auto T = DT_INT32;
  auto func = FDH::Define(
      // Name
      "F",
      // Args
      {"x: int32"},
      // Return values
      {"y: int32"},
      // Attrs
      {},
      // Nodes
      {// a = Square<T>(x)
       {{"a"}, "Square", {"x"}, {{"T", T}}},
       // 1
       FDH::Const("o", 1),
       // A bunch of extra arithmetic that y doesn't depend on
       {{"x1"}, "Identity", {"a"}, {{"T", T}}},
       {{"x2"}, "Identity", {"x1"}, {{"T", T}}},
       {{"x3"}, "Identity", {"x2"}, {{"T", T}}},
       // A stateful node.
       {{"keep_me"},
        "RandomUniform",
        {"o"},
        {{"T", T}, {"dtype", DT_FLOAT}},
        {"x3"}},
       // y = Add<T>(a, o)
       {{"y"}, "Add", {"a", "o"}, {{"T", T}}}});

  {
    Scope s = Scope::DisabledShapeInferenceScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_INT32, 0);
    auto o = ops::Const(s.WithOpName("o"), 1);
    auto a = ops::Square(s.WithOpName("a"), x);
    auto y = ops::Add(s.WithOpName("y"), a, o);
    auto x1 = ops::Identity(s.WithOpName("x1"), a);
    auto x2 = ops::Identity(s.WithOpName("x2"), x1);
    auto x3 = ops::Identity(s.WithOpName("x3"), x2);
    auto keep_me = ops::RandomUniform(
        s.WithOpName("keep_me").WithControlDependencies(x3), {o}, DT_FLOAT);
    auto ret = ops::_Retval(s.WithOpName("y_RetVal"), y, 0);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));
    TF_EXPECT_GRAPH_EQ(expected, Optimize(DoNothing, func));
  }

  {
    Scope s = Scope::DisabledShapeInferenceScope();
    auto x = ops::_Arg(s.WithOpName("x"), DT_INT32, 0);
    auto o = ops::Const(s.WithOpName("o"), 1);
    auto a = ops::Square(s.WithOpName("a"), x);
    auto y = ops::Add(s.WithOpName("y"), a, o);
    auto keep_me = ops::RandomUniform(
        s.WithOpName("keep_me").WithControlDependencies(a), {o}, DT_FLOAT);
    auto ret = ops::_Retval(s.WithOpName("y_RetVal"), y, 0);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));
    TF_EXPECT_GRAPH_EQ(expected,
                       Optimize(::tensorflow::RemoveIdentityNodes, func));
  }
}

TEST(OptimizationTest, RemoveListArrayConverter) {
  auto func = FDH::Create(
      // Name
      "Test",
      // Args
      {"i: float"},
      // Return signature
      {"o: float"},
      // Attrs
      {},
      // Nodes
      {FDH::Const("zero", 0),
       {{"s"},
        "Split",
        {"zero:output:0", "i"},
        {{"num_split", 4}, {"T", DT_FLOAT}}},
       {{"a"},
        "_ArrayToList",
        {"s:output"},
        {{"N", 4},
         {"T", DT_FLOAT},
         {"out_types", DataTypeSlice{DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT}}}},
       {{"l"}, "Mul", {"a:output:0", "a:output:1"}, {{"T", DT_FLOAT}}},
       {{"r"}, "Mul", {"a:output:2", "a:output:3"}, {{"T", DT_FLOAT}}},
       {{"x"},
        "_ListToArray",
        {"l:z", "r:z"},
        {{"N", 2},
         {"T", DT_FLOAT},
         {"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}}}},
       {{"o"}, "AddN", {"x:output"}, {{"N", 2}, {"T", DT_FLOAT}}}},
      // Return values
      {{"o", "o:sum"}});

  {
    Scope scope = Scope::DisabledShapeInferenceScope();
    auto i = ops::_Arg(scope.WithOpName("i"), DT_FLOAT, 0);
    auto zero = ops::Const(scope.WithOpName("zero"), 0);
    auto s = ops::Split(scope.WithOpName("s"), zero, i, 4);
    auto a = ops::_ArrayToList(scope.WithOpName("a"), s.output,
                               {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT});
    auto r = ops::Mul(scope.WithOpName("r"), a[2], a[3]);
    auto l = ops::Mul(scope.WithOpName("l"), a[0], a[1]);
    auto x = ops::_ListToArray(scope.WithOpName("x"),
                               std::initializer_list<Input>{l, r}, DT_FLOAT, 2);
    auto o = ops::AddN(scope.WithOpName("o"), x.output);
    auto o_ret = ops::_Retval(scope.WithOpName("o_RetVal"), o, 0);
    GraphDef expected;
    TF_ASSERT_OK(scope.ToGraphDef(&expected));
    TF_EXPECT_GRAPH_EQ(expected, Optimize(DoNothing, func));
  }

  {
    Scope scope = Scope::NewRootScope();
    auto i = ops::_Arg(scope.WithOpName("i"), DT_FLOAT, 0);
    auto zero = ops::Const(scope.WithOpName("zero"), 0);
    auto s = ops::Split(scope.WithOpName("s"), zero, i, 4);
    auto func_0 = ops::Identity(scope.WithOpName("Func/a/input/_0"), s[0]);
    auto func_1 = ops::Identity(scope.WithOpName("Func/a/input/_1"), s[1]);
    auto func_2 = ops::Identity(scope.WithOpName("Func/a/input/_2"), s[2]);
    auto func_3 = ops::Identity(scope.WithOpName("Func/a/input/_3"), s[3]);
    auto r = ops::Mul(scope.WithOpName("r"), func_2, func_3);
    auto l = ops::Mul(scope.WithOpName("l"), func_0, func_1);
    auto func_4 = ops::Identity(scope.WithOpName("Func/x/input/_4"), l);
    auto func_5 = ops::Identity(scope.WithOpName("Func/x/input/_5"), r);
    auto o = ops::AddN(scope.WithOpName("o"),
                       std::initializer_list<Input>{func_4, func_5});
    auto o_ret = ops::_Retval(scope.WithOpName("o_RetVal"), o, 0);
    GraphDef expected;
    TF_ASSERT_OK(scope.ToGraphDef(&expected));
    TF_EXPECT_GRAPH_EQ(expected, Optimize(RemoveListArrayConverter, func));
  }

  {
    Scope scope = Scope::NewRootScope();
    auto i = ops::_Arg(scope.WithOpName("i"), DT_FLOAT, 0);
    auto zero = ops::Const(scope.WithOpName("zero"), 0);
    auto s = ops::Split(scope.WithOpName("s"), zero, i, 4);
    auto r = ops::Mul(scope.WithOpName("r"), s[2], s[3]);
    auto l = ops::Mul(scope.WithOpName("l"), s[0], s[1]);
    auto o =
        ops::AddN(scope.WithOpName("o"), std::initializer_list<Input>{l, r});
    auto o_ret = ops::_Retval(scope.WithOpName("o_RetVal"), o, 0);
    GraphDef expected;
    TF_ASSERT_OK(scope.ToGraphDef(&expected));

    auto remove_listarray_and_identity = [](Graph* g) {
      return RemoveListArrayConverter(g) && RemoveIdentityNodes(g);
    };
    TF_EXPECT_GRAPH_EQ(expected, Optimize(remove_listarray_and_identity, func));
  }
}

TEST(OptimizationTest, RemoveListArrayConverter_WithControlDeps) {
  auto func = FDH::Create(
      // Name
      "Test",
      // Args
      {"i: float"},
      // Return values
      {"o: float"},
      // Attrs
      {},
      // Nodes
      {FDH::Const("dummy", 0),
       {{"x"},
        "_ListToArray",
        {"i", "i"},
        {{"N", 2}, {"T", DT_FLOAT}, {"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}}},
        // Control dep
        {"dummy"}},
       {{"o"},
        "AddN",
        {"x:output"},
        {{"N", 2}, {"T", DT_FLOAT}},
        // Control dep
        {"x"}}},
      {{"o", "o:sum"}});

  {
    Scope s = Scope::DisabledShapeInferenceScope();
    auto i = ops::_Arg(s.WithOpName("i"), DT_FLOAT, 0);
    auto dummy = ops::Const(s.WithOpName("dummy"), 0);
    auto x = ops::_ListToArray(s.WithOpName("x").WithControlDependencies(dummy),
                               std::initializer_list<Input>{i, i}, DT_FLOAT, 2);
    auto o =
        ops::AddN(s.WithOpName("o").WithControlDependencies({x.output[0].op()}),
                  x.output);
    auto o_ret = ops::_Retval(s.WithOpName("o_RetVal"), o, 0);
    GraphDef expected;
    TF_ASSERT_OK(s.ToGraphDef(&expected));
    TF_EXPECT_GRAPH_EQ(expected, Optimize(DoNothing, func));
  }

  GraphDef expected;
  {
    Scope s = Scope::NewRootScope();
    auto i = ops::_Arg(s.WithOpName("i"), DT_FLOAT, 0);
    auto dummy = ops::Const(s.WithOpName("dummy"), 0);
    auto func_2 = ops::NoOp(s.WithOpName("Func/x/input_control_node/_2")
                                .WithControlDependencies(dummy));
    auto func_0 = ops::Identity(
        s.WithOpName("Func/x/input/_0").WithControlDependencies({func_2}), i);
    auto func_1 = ops::Identity(
        s.WithOpName("Func/x/input/_1").WithControlDependencies({func_2}), i);
    auto func_3 = ops::NoOp(
        s.WithOpName("Func/x/output_control_node/_3")
            .WithControlDependencies({func_0.output.op(), func_1.output.op()}));
    auto o = ops::AddN(s.WithOpName("o").WithControlDependencies({func_3}),
                       std::initializer_list<Input>{func_0, func_1});
    auto o_ret = ops::_Retval(s.WithOpName("o_RetVal"), o, 0);
    TF_ASSERT_OK(s.ToGraphDef(&expected));
  }
  TF_EXPECT_GRAPH_EQ(expected, Optimize(RemoveListArrayConverter, func));

  auto remove_listarray_and_identity = [](Graph* g) {
    return RemoveListArrayConverter(g) && RemoveIdentityNodes(g);
  };
  // NOTE: We are not removing Identity nodes with any control
  // dependencies yet.
  TF_EXPECT_GRAPH_EQ(expected, Optimize(remove_listarray_and_identity, func));
}

}  // namespace
}  // namespace tensorflow
