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
#include "tensorflow/core/tfrt/eager/op_cache.h"

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tfrt/eager/c_api_tfrt.h"
#include "tfrt/cpu/core_runtime/null_op_handler.h"  // from @tf_runtime

namespace tfrt {
namespace tf {
namespace {

constexpr char device_name[] = "/job:localhost/replica:0/task:0/device:CPU:0";
constexpr char op_name[] = "Add";
constexpr char dtype[] = "DT_INT8";

class OpCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up context.
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    TFE_ContextOptionsSetTfrt(opts, /*use_tfrt=*/true);
    tensorflow::AbstractContext* ctx_raw = nullptr;
    ctx_raw =
        tensorflow::unwrap(TF_NewEagerExecutionContext(opts, status.get()));
    tensorflow::Status s = tensorflow::StatusFromTF_Status(status.get());
    ASSERT_TRUE(s.ok());
    TFE_DeleteContextOptions(opts);
    ctx_.reset(ctx_raw);

    // Set up operation.
    auto op_interface_ptr =
        tensorflow::down_cast<::tfrt::tf::OperationInterface*>(
            ctx_->CreateOperation());
    op_interface_.reset(op_interface_ptr);
    ASSERT_TRUE(op_interface_->Reset(op_name, device_name).ok());
    ASSERT_TRUE(op_interface_->SetAttrType("T", tensorflow::DT_INT8).ok());
  }

  tensorflow::AbstractContextPtr ctx_;
  std::unique_ptr<::tfrt::tf::OperationInterface> op_interface_;
  ::tfrt::tf::OpCache cache_;
};

TEST_F(OpCacheTest, TestOpCacheInitiallyEmpty) {
  // Cache is empty initially.
  EXPECT_EQ(cache_.Size(), 0);
  EXPECT_FALSE(
      cache_.Contains(op_name, /*op_handler=*/nullptr, device_name, {dtype}));
}

TEST_F(OpCacheTest, TestOpCacheCacheHit) {
  auto expected_op =
      cache_.GetOrAddOp(op_name, /*op_handler=*/nullptr, device_name, {dtype},
                        op_interface_.get());
  // Inserts a new cache entry.
  EXPECT_TRUE((bool)expected_op) << StrCat(expected_op.takeError());

  // There's one entry in the cache.
  EXPECT_EQ(cache_.Size(), 1);
  EXPECT_TRUE(
      cache_.Contains(op_name, /*op_handler=*/nullptr, device_name, {dtype}));

  // This lookup is a cache hit.
  expected_op = cache_.GetOrAddOp(op_name, /*op_handler=*/nullptr, device_name,
                                  {dtype}, op_interface_.get());
  EXPECT_TRUE((bool)expected_op) << StrCat(expected_op.takeError());

  // Cache hit doesn't create new entry in the cache.
  EXPECT_EQ(cache_.Size(), 1);
}

TEST_F(OpCacheTest, TestOpCacheCacheDeviceNameNotSpecifiedAndCacheMiss) {
  // Inserts a new cache entry.
  auto expected_op =
      cache_.GetOrAddOp(op_name, /*op_handler=*/nullptr, device_name, {dtype},
                        op_interface_.get());
  EXPECT_TRUE((bool)expected_op) << StrCat(expected_op.takeError());

  // Inserts a op with empty device name. This incurs a cache miss.
  expected_op =
      cache_.GetOrAddOp(op_name, /*op_handler=*/nullptr, /*device_name=*/"",
                        {dtype}, op_interface_.get());
  EXPECT_TRUE((bool)expected_op) << StrCat(expected_op.takeError());

  // The eager placer (OpHandlerSelector) picks a device for the op.
  EXPECT_STREQ(expected_op.get()->DeviceName().str().c_str(), device_name);

  // This is a cache miss and will insert a new entry to the cache.
  EXPECT_EQ(cache_.Size(), 2);

  // Inserts a op with another dtype. This incurs a cache miss.
  expected_op =
      cache_.GetOrAddOp(op_name, /*op_handler=*/nullptr, /*device_name=*/"",
                        {"F64"}, op_interface_.get());
  EXPECT_TRUE((bool)expected_op) << StrCat(expected_op.takeError());

  // This is a cache miss and will insert a new entry to the cache.
  EXPECT_EQ(cache_.Size(), 3);
}

TEST_F(OpCacheTest, TestOpCacheAlreadyPlaced) {
  auto* op_handler =
      tensorflow::down_cast<::tfrt::tf::ContextInterface*>(ctx_.get())
          ->GetCoreRuntime()
          ->GetOpHandler(device_name);
  EXPECT_TRUE(op_handler != nullptr);
  // Inserts a new cache entry.
  auto expected_op = cache_.GetOrAddOp(op_name, op_handler, device_name,
                                       {dtype}, op_interface_.get());
  EXPECT_TRUE((bool)expected_op) << StrCat(expected_op.takeError());

  // The lookup is a cache hit.
  expected_op = cache_.GetOrAddOp(op_name, op_handler, device_name, {dtype},
                                  op_interface_.get());
  EXPECT_TRUE((bool)expected_op) << StrCat(expected_op.takeError());

  EXPECT_EQ(cache_.Size(), 1);
}

}  // namespace
}  // namespace tf
}  // namespace tfrt
