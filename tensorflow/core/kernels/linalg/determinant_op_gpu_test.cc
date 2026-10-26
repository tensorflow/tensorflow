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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/gpu_solvers.h"

namespace tensorflow {
namespace {

using ::absl_testing::StatusIs;
using ::testing::An;
using ::testing::NiceMock;
using ::testing::Return;

class DeterminantOpGpuTest : public OpsTestBase {
 protected:
  void TearDown() override {
    std::move(context_).reset();
    if (params_ != nullptr) {
      params_->op_device_context = nullptr;
    }
    gpu_dc_.reset();
  }

  void InitGpuOp(const char* op_name) {
    SetDevice(DEVICE_GPU, std::unique_ptr<Device>(DeviceFactory::NewDevice(
                              "GPU", {}, "/job:a/replica:0/task:0")));
    TF_ASSERT_OK(NodeDefBuilder("test_op", op_name)
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("T", DT_FLOAT)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void SetUpFailingMemcpyDeviceContext() {
    se::Stream* real_stream = device_->tensorflow_accelerator_device_info()
                                  ->default_context->stream();
    ON_CALL(mock_stream_, parent())
        .WillByDefault(Return(real_stream->parent()));
    ON_CALL(mock_stream_, platform_specific_handle())
        .WillByDefault(Return(real_stream->platform_specific_handle()));
    ON_CALL(mock_stream_, Memcpy(An<se::DeviceAddressBase*>(),
                                 An<const void*>(), An<uint64_t>()))
        .WillByDefault([real_stream](se::DeviceAddressBase* gpu_dst,
                                     const void* host_src, uint64_t size) {
          return real_stream->Memcpy(gpu_dst, host_src, size);
        });
    EXPECT_CALL(
        mock_stream_,
        Memcpy(An<void*>(), An<const se::DeviceAddressBase&>(), An<uint64_t>()))
        .WillOnce(Return(absl::InternalError("Simulated Memcpy failure")));
    EXPECT_CALL(mock_stream_, BlockHostUntilDone()).WillOnce([real_stream]() {
      return real_stream->BlockHostUntilDone();
    });

    gpu_dc_.reset(new GPUDeviceContext(
        /*stream_id=*/0, &mock_stream_,
#if TENSORFLOW_USE_ROCM
        /*nccl_stream=*/nullptr,
#endif
        /*host_to_device_stream=*/nullptr,
        /*device_to_host_stream=*/nullptr,
        /*device_to_device_stream=*/{},
        /*host_memory_allocator=*/nullptr));
    params_->op_device_context = gpu_dc_.get();
  }

  NiceMock<se::MockStream> mock_stream_;
  core::RefCountPtr<GPUDeviceContext> gpu_dc_;
};

TEST_F(DeterminantOpGpuTest,
       CheckLapackInfoAndDeleteSolverAsyncInvokesCallbackOnCopyToHostError) {
  InitGpuOp("MatrixDeterminant");
  AddInputFromArray<float>(TensorShape({2, 2}), {1.0f, 2.0f, 3.0f, 4.0f});
  CreateContext();
  SetUpFailingMemcpyDeviceContext();

  auto solver = std::make_unique<GpuSolver>(context_.get());
  std::vector<DeviceLapackInfo> dev_info;
  dev_info.push_back(solver->GetDeviceLapackInfo(1, "test_lapack_info"));

  bool callback_called = false;
  absl::Status callback_status = absl::OkStatus();
  GpuSolver::CheckLapackInfoAndDeleteSolverAsync(
      std::move(solver), dev_info,
      [&](const absl::Status& status,
          const std::vector<HostLapackInfo>& /*host_infos*/) {
        callback_called = true;
        callback_status = status;
      });

  EXPECT_TRUE(callback_called);
  EXPECT_THAT(callback_status, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(DeterminantOpGpuTest,
       CheckLapackInfoAndDeleteSolverAsyncDrainsStreamOnPartialCopyError) {
  InitGpuOp("MatrixDeterminant");
  AddInputFromArray<float>(TensorShape({2, 2}), {1.0f, 2.0f, 3.0f, 4.0f});
  CreateContext();

  se::Stream* real_stream =
      device_->tensorflow_accelerator_device_info()->default_context->stream();
  ON_CALL(mock_stream_, parent()).WillByDefault(Return(real_stream->parent()));
  ON_CALL(mock_stream_, platform_specific_handle())
      .WillByDefault(Return(real_stream->platform_specific_handle()));
  EXPECT_CALL(
      mock_stream_,
      Memcpy(An<void*>(), An<const se::DeviceAddressBase&>(), An<uint64_t>()))
      .WillOnce(Return(absl::OkStatus()))
      .WillOnce(Return(absl::InternalError("Simulated second Memcpy failure")));
  bool stream_blocked = false;
  EXPECT_CALL(mock_stream_, BlockHostUntilDone())
      .WillOnce([real_stream, &stream_blocked]() {
        stream_blocked = true;
        return real_stream->BlockHostUntilDone();
      });

  gpu_dc_.reset(new GPUDeviceContext(
      /*stream_id=*/0, &mock_stream_,
#if TENSORFLOW_USE_ROCM
      /*nccl_stream=*/nullptr,
#endif
      /*host_to_device_stream=*/nullptr,
      /*device_to_host_stream=*/nullptr,
      /*device_to_device_stream=*/{},
      /*host_memory_allocator=*/nullptr));
  params_->op_device_context = gpu_dc_.get();

  auto solver = std::make_unique<GpuSolver>(context_.get());
  std::vector<DeviceLapackInfo> dev_info;
  dev_info.push_back(solver->GetDeviceLapackInfo(1, "test_lapack_info_0"));
  dev_info.push_back(solver->GetDeviceLapackInfo(1, "test_lapack_info_1"));

  bool callback_called = false;
  absl::Status callback_status = absl::OkStatus();
  GpuSolver::CheckLapackInfoAndDeleteSolverAsync(
      std::move(solver), dev_info,
      [&](const absl::Status& status,
          const std::vector<HostLapackInfo>& host_infos) {
        EXPECT_TRUE(stream_blocked);
        EXPECT_EQ(host_infos.size(), 1u);
        callback_called = true;
        callback_status = status;
      });

  EXPECT_TRUE(callback_called);
  EXPECT_THAT(callback_status, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(
    DeterminantOpGpuTest,
    CheckLapackInfoAndDeleteSolverAsyncInvokesDoneCallbackOnCopyToHostError) {
  InitGpuOp("MatrixDeterminant");
  AddInputFromArray<float>(TensorShape({2, 2}), {1.0f, 2.0f, 3.0f, 4.0f});
  CreateContext();
  SetUpFailingMemcpyDeviceContext();

  auto solver = std::make_unique<GpuSolver>(context_.get());
  std::vector<DeviceLapackInfo> dev_info;
  dev_info.push_back(solver->GetDeviceLapackInfo(1, "test_lapack_info"));

  bool done_called = false;
  GpuSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                 [&]() { done_called = true; });

  EXPECT_TRUE(done_called);
  EXPECT_THAT(context_->status(), StatusIs(absl::StatusCode::kInternal));
}

TEST_F(DeterminantOpGpuTest,
       MatrixDeterminantGpuInvokesDoneAndPropagatesCopyToHostError) {
  InitGpuOp("MatrixDeterminant");
  AddInputFromArray<float>(TensorShape({2, 2}), {1.0f, 2.0f, 3.0f, 4.0f});
  CreateContext();
  SetUpFailingMemcpyDeviceContext();

  bool done_called = false;
  device_->ComputeAsync(op_kernel()->AsAsync(), context_.get(),
                        [&]() { done_called = true; });

  EXPECT_TRUE(done_called);
  EXPECT_THAT(context_->status(), StatusIs(absl::StatusCode::kInternal));
}

TEST_F(DeterminantOpGpuTest,
       LogMatrixDeterminantGpuInvokesDoneAndPropagatesCopyToHostError) {
  InitGpuOp("LogMatrixDeterminant");
  AddInputFromArray<float>(TensorShape({2, 2}), {1.0f, 2.0f, 3.0f, 4.0f});
  CreateContext();
  SetUpFailingMemcpyDeviceContext();

  bool done_called = false;
  device_->ComputeAsync(op_kernel()->AsAsync(), context_.get(),
                        [&]() { done_called = true; });

  EXPECT_TRUE(done_called);
  EXPECT_THAT(context_->status(), StatusIs(absl::StatusCode::kInternal));
}

TEST_F(DeterminantOpGpuTest,
       CholeskyGpuInvokesDoneAndPropagatesCopyToHostError) {
  InitGpuOp("Cholesky");
  AddInputFromArray<float>(TensorShape({2, 2}), {4.0f, 1.0f, 1.0f, 3.0f});
  CreateContext();
  SetUpFailingMemcpyDeviceContext();

  bool done_called = false;
  device_->ComputeAsync(op_kernel()->AsAsync(), context_.get(),
                        [&]() { done_called = true; });

  EXPECT_TRUE(done_called);
  EXPECT_THAT(context_->status(), StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
