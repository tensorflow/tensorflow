/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/synchronization/notification.h"
#include "xla/tpu/tpu_api.h"
#include "xla/tpu/tpu_ops_c_api.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/tpu/kernels/tpu_ordinal_selector.h"

namespace tensorflow {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

int g_dequeue_count = 0;
int32_t g_last_dequeued_ordinal = -1;
int64_t g_last_dequeued_req_id = -1;

TfTpu_OpsApiFn* GetTestOpsApiFn() {
  static TfTpu_OpsApiFn ops_api_fn = []() {
    TfTpu_OpsApiFn fn = {};
    fn.TfTpu_GetTpuPartitionedCallParamsFn =
        [](TpuPartitionedCall_Params* params) {};
    fn.TfTpuOrdinalSelector_CreateFn =
        [](TfTpuOrdinalSelector** ordinal_selector, int num_cores_per_replica) {
          *ordinal_selector = reinterpret_cast<TfTpuOrdinalSelector*>(0x1);
        };
    fn.TfTpuOrdinalSelector_DestroyFn =
        [](TfTpuOrdinalSelector* ordinal_selector) {};
    fn.TfTpuOrdinalSelector_GetOrdinalFn =
        [](TfTpuOrdinalSelector* ordinal_selector, std::optional<uint64_t> key,
           int64_t* req_id, int64_t* ordinal) {
          *req_id = 7;
          *ordinal = 0;
        };
    fn.TfTpuOrdinalSelector_DequeueFromCoreSelectorFn =
        [](TfTpuOrdinalSelector* ordinal_selector, int32_t device_ordinal,
           int64_t req_id) {
          ++g_dequeue_count;
          g_last_dequeued_ordinal = device_ordinal;
          g_last_dequeued_req_id = req_id;
        };
    return fn;
  }();
  return &ops_api_fn;
}

class TpuFunctionalOpsTest : public OpsTestBase {
 protected:
  void SetUp() override {
    g_dequeue_count = 0;
    g_last_dequeued_ordinal = -1;
    g_last_dequeued_req_id = -1;
    stream_executor::tpu::SetOpsApiFn(GetTestOpsApiFn());
  }

  void InitPartitionedCallOp(const std::string& func_name) {
    NameAttrList func;
    func.set_name(func_name);
    TF_ASSERT_OK(NodeDefBuilder("tpu_partitioned_call", "TPUPartitionedCall")
                     .Input(FakeInput({DT_INT32}))
                     .Input(FakeInput(DT_INT32))
                     .Attr("Tin", {DT_INT32})
                     .Attr("Tout", {DT_INT32})
                     .Attr("f", func)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  absl::Status RunAsyncKernel(bool with_function_library = true) {
    CreateContext();
    if (!with_function_library) {
      params_->function_library = nullptr;
      context_ = std::make_unique<OpKernelContext>(params_.get());
    }
    absl::Notification done;
    AsyncOpKernel* async_kernel = kernel_->AsAsync();
    CHECK_NE(async_kernel, nullptr);
    device_->ComputeAsync(async_kernel, context_.get(),
                          [&done]() { done.Notify(); });
    done.WaitForNotification();
    return context_->status();
  }
};

TEST_F(TpuFunctionalOpsTest, NoFunctionLibraryPreservesInitStatusAcrossCalls) {
  InitPartitionedCallOp("IdentityFunc");
  AddInputFromArray<int32_t>(TensorShape({1}), {10});
  AddInputFromArray<int32_t>(TensorShape({}), {0});

  EXPECT_THAT(RunAsyncKernel(/*with_function_library=*/false),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("No function library is provided.")));
  // A second call on the same kernel instance must return the preserved
  // initialization error and invoke `done` rather than dereferencing a null
  // `flib_def_` or `library_runtime_`.
  EXPECT_THAT(RunAsyncKernel(/*with_function_library=*/false),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("No function library is provided.")));
}

TEST_F(TpuFunctionalOpsTest,
       MissingFunctionPreservesOrdinalSelectorInitStatusAcrossCalls) {
  InitPartitionedCallOp("NonExistentFunc");
  AddInputFromArray<int32_t>(TensorShape({1}), {10});
  AddInputFromArray<int32_t>(TensorShape({}), {0});

  EXPECT_THAT(
      RunAsyncKernel(),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("The TPUOrdinalSelector is not initialized: Function "
                         "NonExistentFunc is not defined.")));
  // A second call on the same kernel instance hits `OP_REQUIRES_OK_ASYNC(ctx,
  // init_status_, ...)` with the preserved `init_status_` and invokes `done`.
  EXPECT_THAT(RunAsyncKernel(),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Function NonExistentFunc is not defined.")));
}

TEST_F(TpuFunctionalOpsTest,
       SubgraphInstantiationFailureDoesNotPoisonCacheAndDequeuesCore) {
  AttrValue device_ordinal_attr;
  device_ordinal_attr.set_i(0);
  FunctionDef fdef = FunctionDefHelper::Create(
      "UnsupportedDeviceOrdinalFunc", {"x: int32"}, {"y: int32"}, {},
      {{{"id"},
        "Identity",
        {"x"},
        {{"T", DataType::DT_INT32}, {"device_ordinal", device_ordinal_attr}}}},
      {{"y", "id:output:0"}});
  TF_ASSERT_OK(flib_def_->AddFunctionDef(fdef));

  InitPartitionedCallOp("UnsupportedDeviceOrdinalFunc");
  AddInputFromArray<int32_t>(TensorShape({1}), {10});
  AddInputFromArray<int32_t>(TensorShape({}),
                             {tpu::kDeferredCoreSelectionReserved});

  EXPECT_THAT(RunAsyncKernel(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Node Identity is not yet supported.")));
  EXPECT_EQ(g_dequeue_count, 1);
  EXPECT_EQ(g_last_dequeued_ordinal, 0);
  EXPECT_EQ(g_last_dequeued_req_id, 7);

  // A subsequent call with the same cache hash must not hit a poisoned empty
  // `partition_cache_` entry (which would hang without calling `done`).
  EXPECT_THAT(RunAsyncKernel(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Node Identity is not yet supported.")));
  EXPECT_EQ(g_dequeue_count, 2);
}

TEST_F(TpuFunctionalOpsTest, ExecuteFunctionsSuccessAndCacheHit) {
  FunctionDef fdef = FunctionDefHelper::Create(
      "IdentityFunc", {"x: int32"}, {"y: int32"}, {},
      {{{"id"}, "Identity", {"x"}, {{"T", DataType::DT_INT32}}}},
      {{"y", "id:output:0"}});
  TF_ASSERT_OK(flib_def_->AddFunctionDef(fdef));

  InitPartitionedCallOp("IdentityFunc");
  AddInputFromArray<int32_t>(TensorShape({1}), {42});
  AddInputFromArray<int32_t>(TensorShape({}),
                             {tpu::kDeferredCoreSelectionReserved});

  // First call: cache miss -> InstantiateFunctionsFromSubgraphs ->
  // ExecuteFunctions.
  ASSERT_THAT(RunAsyncKernel(), IsOk());
  EXPECT_EQ(GetOutput(0)->flat<int32_t>()(0), 42);
  EXPECT_EQ(g_dequeue_count, 1);

  // Second call: cache hit -> ExecuteFunctions.
  ASSERT_THAT(RunAsyncKernel(), IsOk());
  EXPECT_EQ(GetOutput(0)->flat<int32_t>()(0), 42);
  EXPECT_EQ(g_dequeue_count, 2);
}

}  // namespace
}  // namespace tensorflow
