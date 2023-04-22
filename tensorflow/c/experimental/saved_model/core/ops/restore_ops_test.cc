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

#include "tensorflow/c/experimental/saved_model/core/ops/restore_ops.h"

#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/test_utils.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

std::string CheckpointPrefix(StringPiece saved_model_dir) {
  return io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model/testdata",
                      saved_model_dir, kSavedModelVariablesDirectory,
                      kSavedModelVariablesFilename);
}

class RestoreOpsTest : public ::testing::Test {
 public:
  RestoreOpsTest()
      : device_mgr_(testing::CreateTestingDeviceMgr()),
        ctx_(testing::CreateTestingEagerContext(device_mgr_.get())) {}

  EagerContext* context() { return ctx_.get(); }

 private:
  std::unique_ptr<StaticDeviceMgr> device_mgr_;
  EagerContextPtr ctx_;
};

// One way of obtaining the checkpointa checkpoint's tensor names is:
// bazel run //tensorflow/python/tools:inspect_checkpoint -- --all_tensors
// --file_name="$CKPT_PREFIX".
// Here are the values for VarsAndArithmeticObjectGraph:
// tensor: child/z/.ATTRIBUTES/VARIABLE_VALUE (float32) []
// 3.0
// tensor: x/.ATTRIBUTES/VARIABLE_VALUE (float32) []
// 1.0
// tensor: y/.ATTRIBUTES/VARIABLE_VALUE (float32) []
// 2.0

TEST_F(RestoreOpsTest, RestoreSuccessful) {
  ImmediateTensorHandlePtr x_handle;
  TF_EXPECT_OK(internal::SingleRestore(
      context(), CheckpointPrefix("VarsAndArithmeticObjectGraph"),
      "x/.ATTRIBUTES/VARIABLE_VALUE", DT_FLOAT, &x_handle));
  AbstractTensorPtr x = testing::TensorHandleToTensor(x_handle.get());
  EXPECT_EQ(x->Type(), DT_FLOAT);
  EXPECT_EQ(x->NumElements(), 1);
  EXPECT_EQ(x->NumDims(), 0);
  EXPECT_FLOAT_EQ(*reinterpret_cast<float*>(x->Data()), 1.0f);

  ImmediateTensorHandlePtr y_handle;
  TF_EXPECT_OK(internal::SingleRestore(
      context(), CheckpointPrefix("VarsAndArithmeticObjectGraph"),
      "y/.ATTRIBUTES/VARIABLE_VALUE", DT_FLOAT, &y_handle));
  AbstractTensorPtr y = testing::TensorHandleToTensor(y_handle.get());
  EXPECT_EQ(y->Type(), DT_FLOAT);
  EXPECT_EQ(y->NumElements(), 1);
  EXPECT_EQ(y->NumDims(), 0);
  EXPECT_FLOAT_EQ(*reinterpret_cast<float*>(y->Data()), 2.0f);

  ImmediateTensorHandlePtr z_handle;
  TF_EXPECT_OK(internal::SingleRestore(
      context(), CheckpointPrefix("VarsAndArithmeticObjectGraph"),
      "child/z/.ATTRIBUTES/VARIABLE_VALUE", DT_FLOAT, &z_handle));
  AbstractTensorPtr z = testing::TensorHandleToTensor(z_handle.get());
  EXPECT_EQ(z->Type(), DT_FLOAT);
  EXPECT_EQ(z->NumElements(), 1);
  EXPECT_EQ(z->NumDims(), 0);
  EXPECT_FLOAT_EQ(*reinterpret_cast<float*>(z->Data()), 3.0f);
}

TEST_F(RestoreOpsTest, BadCheckpointPrefixShouldFail) {
  ImmediateTensorHandlePtr x_handle;
  Status status = internal::SingleRestore(
      context(), CheckpointPrefix("unknown_bad_checkpoint_prefix"),
      "x/.ATTRIBUTES/VARIABLE_VALUE", DT_FLOAT, &x_handle);
  EXPECT_FALSE(status.ok()) << status.error_message();
}

TEST_F(RestoreOpsTest, BadCheckpointKeyShouldFail) {
  ImmediateTensorHandlePtr x_handle;
  Status status = internal::SingleRestore(
      context(), CheckpointPrefix("VarsAndArithmeticObjectGraph"),
      "bad_checkpoint_key", DT_FLOAT, &x_handle);
  EXPECT_FALSE(status.ok()) << status.error_message();
}

}  // namespace
}  // namespace tensorflow
