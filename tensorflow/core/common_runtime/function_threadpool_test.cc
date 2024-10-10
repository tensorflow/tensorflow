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

#include <atomic>
#include <utility>

#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace {

class FunctionLibraryRuntimeTest : public ::testing::Test {
 protected:
  void Init(const std::vector<FunctionDef>& flib,
            thread::ThreadPool* default_thread_pool) {
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
    device_mgr_ = std::make_unique<StaticDeviceMgr>(std::move(devices));
    pflr_.reset(new ProcessFunctionLibraryRuntime(
        device_mgr_.get(), Env::Default(), /*config=*/nullptr,
        TF_GRAPH_DEF_VERSION, lib_def_.get(), opts, default_thread_pool,
        /*parent=*/nullptr, /*session_metadata=*/nullptr,
        Rendezvous::Factory{[](const int64_t, const DeviceMgr* device_mgr,
                               tsl::core::RefCountPtr<Rendezvous>* r) {
          *r = tsl::core::RefCountPtr<Rendezvous>(
              new IntraProcessRendezvous(device_mgr));
          return absl::OkStatus();
        }}));
    flr0_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
  }

  absl::Status Run(FunctionLibraryRuntime* flr,
                   FunctionLibraryRuntime::Handle handle,
                   FunctionLibraryRuntime::Options opts,
                   const std::vector<Tensor>& args, std::vector<Tensor*> rets,
                   bool add_runner = true) {
    std::atomic<int32> call_count(0);
    std::function<void(std::function<void()>)> runner =
        [&call_count](std::function<void()> fn) {
          ++call_count;
          test::function::FunctionTestSchedClosure(fn);
        };
    if (add_runner) {
      opts.runner = &runner;
    } else {
      opts.runner = nullptr;
    }
    Notification done;
    std::vector<Tensor> out;
    absl::Status status;
    flr->Run(opts, handle, args, &out, [&status, &done](const absl::Status& s) {
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

    if (add_runner) {
      EXPECT_GE(call_count, 1);  // Test runner is used.
    }

    return absl::OkStatus();
  }

  absl::Status Instantiate(FunctionLibraryRuntime* flr, const string& name,
                           test::function::Attrs attrs,
                           FunctionLibraryRuntime::Handle* handle) {
    return flr->Instantiate(name, attrs, handle);
  }

  absl::Status Instantiate(
      FunctionLibraryRuntime* flr, const string& name,
      test::function::Attrs attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      FunctionLibraryRuntime::Handle* handle) {
    return flr->Instantiate(name, attrs, options, handle);
  }

  absl::Status InstantiateAndRun(FunctionLibraryRuntime* flr,
                                 const string& name,
                                 test::function::Attrs attrs,
                                 const std::vector<Tensor>& args,
                                 std::vector<Tensor*> rets,
                                 bool add_runner = true) {
    return InstantiateAndRun(flr, name, attrs,
                             FunctionLibraryRuntime::InstantiateOptions(), args,
                             std::move(rets), add_runner);
  }

  absl::Status InstantiateAndRun(
      FunctionLibraryRuntime* flr, const string& name,
      test::function::Attrs attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      const std::vector<Tensor>& args, std::vector<Tensor*> rets,
      bool add_runner = true) {
    FunctionLibraryRuntime::Handle handle;
    absl::Status status = flr->Instantiate(name, attrs, options, &handle);
    if (!status.ok()) {
      return status;
    }
    FunctionLibraryRuntime::Options opts;
    status = Run(flr, handle, opts, args, rets, add_runner);
    if (!status.ok()) return status;

    // Release the handle and try running again. It should not succeed.
    status = flr->ReleaseHandle(handle);
    if (!status.ok()) return status;

    absl::Status status2 = Run(flr, handle, opts, args, std::move(rets));
    EXPECT_TRUE(errors::IsNotFound(status2));
    EXPECT_TRUE(absl::StrContains(status2.message(), "Handle"));
    EXPECT_TRUE(absl::StrContains(status2.message(), "not found"));

    return status;
  }

  absl::Status Run(FunctionLibraryRuntime* flr,
                   FunctionLibraryRuntime::Handle handle,
                   FunctionLibraryRuntime::Options opts,
                   CallFrameInterface* frame, bool add_runner = true) {
    std::atomic<int32> call_count(0);
    std::function<void(std::function<void()>)> runner =
        [&call_count](std::function<void()> fn) {
          ++call_count;
          test::function::FunctionTestSchedClosure(fn);
        };
    if (add_runner) {
      opts.runner = &runner;
    } else {
      opts.runner = nullptr;
    }
    Notification done;
    absl::Status status;
    flr->Run(opts, handle, frame, [&status, &done](const absl::Status& s) {
      status = s;
      done.Notify();
    });
    done.WaitForNotification();
    if (!status.ok()) {
      return status;
    }

    if (add_runner) {
      EXPECT_GE(call_count, 1);  // Test runner is used.
    }

    return absl::OkStatus();
  }

  FunctionLibraryRuntime* flr0_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
};

TEST_F(FunctionLibraryRuntimeTest, DefaultThreadpool) {
  using test::function::blocking_op_state;
  using test::function::BlockingOpState;

  thread::ThreadPool* tp = new thread::ThreadPool(Env::Default(), "FLRTest", 1);
  Init({test::function::BlockingOpFn(), test::function::XTimesTwo()}, tp);

  auto x = test::AsScalar<float>(1.3);
  Tensor y;
  blocking_op_state = new BlockingOpState();

  thread::ThreadPool* tp1 = new thread::ThreadPool(Env::Default(), "tp1", 5);
  bool finished_running = false;
  tp1->Schedule([&x, &y, &finished_running, this]() {
    TF_CHECK_OK(InstantiateAndRun(flr0_, "BlockingOpFn", {}, {x}, {&y},
                                  false /* add_runner */));
    finished_running = true;
  });

  // InstantiateAndRun shouldn't finish because BlockingOpFn should be blocked.
  EXPECT_FALSE(finished_running);

  FunctionLibraryRuntime::Handle h;
  TF_CHECK_OK(Instantiate(flr0_, "XTimesTwo", {{"T", DT_FLOAT}}, &h));

  auto x1 = test::AsTensor<float>({1, 2, 3, 4});
  std::atomic<int32> num_done(0);
  FunctionLibraryRuntime::Options opts;
  for (int i = 0; i < 4; ++i) {
    tp1->Schedule([&h, &x1, &opts, &num_done, this]() {
      Tensor y1;
      TF_CHECK_OK(Run(flr0_, h, opts, {x1}, {&y1}, false /* add_runner */));
      num_done.fetch_add(1);
    });
  }
  // All the 4 Run() calls should be blocked because the runner is occupied.
  EXPECT_EQ(0, num_done.load());

  blocking_op_state->AwaitState(1);
  blocking_op_state->MoveToState(1, 2);
  // Now the runner should be unblocked and all the other Run() calls should
  // proceed.
  blocking_op_state->AwaitState(3);
  blocking_op_state->MoveToState(3, 0);
  delete tp1;
  EXPECT_TRUE(finished_running);
  EXPECT_EQ(4, num_done.load());

  delete blocking_op_state;
  blocking_op_state = nullptr;
  delete tp;
}

}  // namespace
}  // namespace tensorflow
