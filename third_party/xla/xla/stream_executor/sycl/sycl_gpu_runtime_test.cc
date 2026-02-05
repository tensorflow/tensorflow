/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace stream_executor::sycl {
namespace {

class SyclGpuRuntimeTest : public ::testing::Test {
 public:
  std::vector<::sycl::device> sycl_devices_;

 protected:
  absl::StatusOr<void*> AllocateHostBuffer(int count) {
    TF_ASSIGN_OR_RETURN(
        void* buf, SyclMallocHost(kDefaultDeviceOrdinal, sizeof(int) * count));
    if (buf == nullptr) {
      return absl::InternalError(
          "SyclGpuRuntimeTest::AllocateHostBuffer: Failed to allocate host "
          "buffer.");
    }
    return buf;
  }

  absl::StatusOr<void*> AllocateDeviceBuffer(
      int count, int device_ordinal = kDefaultDeviceOrdinal) {
    TF_ASSIGN_OR_RETURN(void* buf,
                        SyclMallocDevice(device_ordinal, sizeof(int) * count));
    if (buf == nullptr) {
      return absl::InternalError(
          "SyclGpuRuntimeTest::AllocateDeviceBuffer: Failed to allocate "
          "device buffer.");
    }
    return buf;
  }

  void VerifyIntBuffer(void* buf, int count, int expected) {
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(static_cast<int*>(buf)[i], expected)
          << "Buffer mismatch at index " << i;
    }
  }

  absl::StatusOr<void*> AllocateAndInitHostBuffer(int count, int value) {
    TF_ASSIGN_OR_RETURN(void* buf, AllocateHostBuffer(count));
    for (int i = 0; i < count; ++i) {
      static_cast<int*>(buf)[i] = value;
    }
    return buf;
  }

  absl::StatusOr<void*> AllocateAndInitDeviceBuffer(
      int count, int value, int device_ordinal = kDefaultDeviceOrdinal) {
    TF_ASSIGN_OR_RETURN(void* buf, AllocateDeviceBuffer(count));
    TF_RETURN_IF_ERROR(
        SyclMemfillDevice(device_ordinal, buf, value, sizeof(int) * count));
    if (buf == nullptr) {
      return absl::InternalError(
          "SyclGpuRuntimeTest::AllocateAndInitDeviceBuffer: Failed to fill "
          "device buffer.");
    }
    return buf;
  }

  void FreeAndNullify(void*& ptr, int device_ordinal = kDefaultDeviceOrdinal) {
    if (ptr != nullptr) {
      EXPECT_THAT(SyclFree(device_ordinal, ptr), absl_testing::IsOk());
      EXPECT_EQ(ptr, nullptr);
    }
  }

 private:
  void SetUp() override {
    // Find the number of SYCL devices available. If there are none, skip the
    // test.
    TF_ASSERT_OK_AND_ASSIGN(int device_count, SyclDevicePool::GetDeviceCount());
    if (device_count <= 0) {
      GTEST_SKIP() << "No SYCL devices found.";
    } else {
      VLOG(2) << "Found " << device_count << " SYCL devices.";
    }

    // Initialize the device pool with available devices.
    for (int i = 0; i < device_count; ++i) {
      TF_ASSERT_OK_AND_ASSIGN(::sycl::device sycl_device,
                              SyclDevicePool::GetDevice(i));
      sycl_devices_.push_back(sycl_device);
    }
  }
};

TEST_F(SyclGpuRuntimeTest, GetDeviceCount) {
  EXPECT_THAT(SyclDevicePool::GetDeviceCount(),
              ::absl_testing::IsOkAndHolds(::testing::Gt(0)));
}

TEST_F(SyclGpuRuntimeTest, GetDeviceOrdinal) {
  TF_ASSERT_OK_AND_ASSIGN(::sycl::device sycl_device,
                          SyclDevicePool::GetDevice(kDefaultDeviceOrdinal));
  TF_ASSERT_OK_AND_ASSIGN(int device_ordinal,
                          SyclDevicePool::GetDeviceOrdinal(sycl_device));
  EXPECT_EQ(device_ordinal, kDefaultDeviceOrdinal);
}

TEST_F(SyclGpuRuntimeTest, TestStaticDeviceContext) {
  // Verify that GetDeviceContext returns the same context instance on multiple
  // calls.
  TF_ASSERT_OK_AND_ASSIGN(::sycl::context saved_sycl_context,
                          SyclDevicePool::GetDeviceContext());
  TF_ASSERT_OK_AND_ASSIGN(::sycl::context current_sycl_context,
                          SyclDevicePool::GetDeviceContext());
  EXPECT_EQ(saved_sycl_context, current_sycl_context);
}

TEST_F(SyclGpuRuntimeTest, TestDefaultStreamSynchronizeAndDestroy) {
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetDefaultStream(kDefaultDeviceOrdinal));
  ASSERT_NE(stream_handle, nullptr);

  TF_ASSERT_OK(SyclStreamPool::SynchronizeStreamPool(kDefaultDeviceOrdinal));

  TF_ASSERT_OK(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle));
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestCreateStreamSynchronizeAndDestroy) {
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  TF_ASSERT_OK(SyclStreamPool::SynchronizeStreamPool(kDefaultDeviceOrdinal));

  TF_ASSERT_OK(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle));
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestStreamPoolCreateAfterDestroy) {
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  TF_ASSERT_OK(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle));
  ASSERT_EQ(stream_handle, nullptr);

  // Verify that we can create a new stream after destroying the previous one.
  TF_ASSERT_OK_AND_ASSIGN(
      stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  // Clean up the stream after the test.
  TF_ASSERT_OK(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle));
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestStreamPoolCreate_Negative) {
  constexpr int kInvalidDeviceOrdinal = -1;
  EXPECT_THAT(
      SyclStreamPool::GetOrCreateStream(kInvalidDeviceOrdinal,
                                        /*enable_multiple_streams=*/false),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SyclGpuRuntimeTest, TestStreamPoolDestroy_Negative) {
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  TF_ASSERT_OK(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle));
  ASSERT_EQ(stream_handle, nullptr);

  // Try to destroy the stream again, which should be a no-op.
  EXPECT_THAT(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestMaxStreamsPerDevice) {
  // Ensure that the maximum number of streams per device is respected.
  constexpr int kMaxStreams = 8;
  std::vector<StreamPtr> streams(kMaxStreams);
  for (int i = 0; i < kMaxStreams - 1; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(streams[i], SyclStreamPool::GetOrCreateStream(
                                            kDefaultDeviceOrdinal,
                                            /*enable_multiple_streams=*/true));
    ASSERT_NE(streams[i], nullptr);
  }

  // Attempt to create one more stream, which should fail.
  EXPECT_THAT(
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/true),
      absl_testing::StatusIs(absl::StatusCode::kResourceExhausted));

  // Clean up the streams created.
  for (int i = 0; i < kMaxStreams - 1; ++i) {
    TF_ASSERT_OK(
        SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, streams[i]));
    EXPECT_EQ(streams[i], nullptr);
  }
}

TEST_F(SyclGpuRuntimeTest, TestGetTimerProperties) {
  TF_ASSERT_OK_AND_ASSIGN(SyclTimerProperties timer_props,
                          SyclGetTimerProperties(kDefaultDeviceOrdinal));
  EXPECT_GT(timer_props.frequency_hz, 0);
  EXPECT_GT(timer_props.timestamp_mask, 0);
}

TEST_F(SyclGpuRuntimeTest, TestSyclGetRecentEventFromStream) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  // Ensure there is an event associated with the stream by filling some memory
  // on the device.
  TF_ASSERT_OK_AND_ASSIGN(void* device_buf,
                          AllocateAndInitDeviceBuffer(kCount, 0xDEADC0DE));

  TF_ASSERT_OK(SyclStreamSynchronize(stream_handle.get()));

  TF_ASSERT_OK_AND_ASSIGN(std::optional<::sycl::event> event,
                          SyclGetRecentEventFromStream(stream_handle.get()));

  ASSERT_TRUE(event.has_value());

  // Expect the event to be in a valid state. The command_execution_status
  // should not be "unknown".
  EXPECT_NE(
      event.value().get_info<::sycl::info::event::command_execution_status>(),
      ::sycl::info::event_command_status::ext_oneapi_unknown);

  FreeAndNullify(device_buf);

  // Destroy the stream after use.
  TF_ASSERT_OK(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle));
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestSyclMemcopyAsync_DeviceToHost) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(void* src_device,
                          AllocateAndInitDeviceBuffer(kCount, 0xDEADC0DE));
  TF_ASSERT_OK_AND_ASSIGN(void* dst_host, AllocateHostBuffer(kCount));

  ASSERT_OK(SyclMemcpyAsync(stream_handle.get(), dst_host, src_device,
                            sizeof(int) * kCount,
                            SyclMemcpyKind::kSyclMemcpyDeviceToHost));

  // Synchronize the stream to ensure the copy is complete before checking
  // results.
  ASSERT_OK(SyclStreamSynchronize(stream_handle.get()));

  // Check the results after synchronization.
  VerifyIntBuffer(dst_host, kCount, 0xDEADC0DE);

  FreeAndNullify(src_device);
  FreeAndNullify(dst_host);

  // Destroy the stream after use.
  ASSERT_OK(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle));
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestSyclMemcopyAsync_HostToDeviceAndBack) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(void* src_host,
                          AllocateAndInitHostBuffer(kCount, 0xDEADC0DE));
  TF_ASSERT_OK_AND_ASSIGN(void* dst_device, AllocateDeviceBuffer(kCount));

  ASSERT_OK(SyclMemcpyAsync(stream_handle.get(), dst_device, src_host,
                            sizeof(int) * kCount,
                            SyclMemcpyKind::kSyclMemcpyHostToDevice));

  // Clear out the host buffer to ensure data is copied back correctly.
  // Before that, synchronize to ensure the first copy is done.
  ASSERT_OK(SyclStreamSynchronize(stream_handle.get()));
  for (int i = 0; i < kCount; ++i) {
    static_cast<int*>(src_host)[i] = 0;
  }

  ASSERT_OK(SyclMemcpyAsync(stream_handle.get(), src_host, dst_device,
                            sizeof(int) * kCount,
                            SyclMemcpyKind::kSyclMemcpyDeviceToHost));

  // Synchronize the stream to ensure the copy is complete before checking
  // results.
  ASSERT_OK(SyclStreamSynchronize(stream_handle.get()));

  // Check the results after synchronization.
  VerifyIntBuffer(src_host, kCount, 0xDEADC0DE);

  FreeAndNullify(src_host);
  FreeAndNullify(dst_device);

  // Destroy the stream after use.
  ASSERT_OK(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle));
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestSyclMemcopyAsync_DeviceToDeviceAndBackToHost) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  // Allocate device buffers that reside on the same device.
  TF_ASSERT_OK_AND_ASSIGN(void* src_device,
                          AllocateAndInitDeviceBuffer(kCount, 0xDEADC0DE));
  TF_ASSERT_OK_AND_ASSIGN(void* dst_device,
                          AllocateAndInitDeviceBuffer(kCount, 0));

  ASSERT_OK(SyclMemcpyAsync(stream_handle.get(), dst_device, src_device,
                            sizeof(int) * kCount,
                            SyclMemcpyKind::kSyclMemcpyDeviceToDevice));

  // Synchronize to ensure the device-to-device copy is done before copying
  // back to host.
  ASSERT_OK(SyclStreamSynchronize(stream_handle.get()));

  TF_ASSERT_OK_AND_ASSIGN(void* dst_host, AllocateAndInitHostBuffer(kCount, 0));

  // Verify the copy by reading back to host.
  ASSERT_OK(SyclMemcpyAsync(stream_handle.get(), dst_host, dst_device,
                            sizeof(int) * kCount,
                            SyclMemcpyKind::kSyclMemcpyDeviceToHost));

  // Synchronize the stream to ensure the copy is complete before checking
  ASSERT_OK(SyclStreamSynchronize(stream_handle.get()));

  // Check the results after synchronization.
  VerifyIntBuffer(dst_host, kCount, 0xDEADC0DE);

  FreeAndNullify(src_device);
  FreeAndNullify(dst_device);
  FreeAndNullify(dst_host);

  // Destroy the stream after use.
  ASSERT_OK(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle));
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestMemcopyDeviceToHost) {
  constexpr int kCount = 12;
  TF_ASSERT_OK_AND_ASSIGN(void* src_device,
                          AllocateAndInitDeviceBuffer(kCount, 0xDEADBEEF));
  TF_ASSERT_OK_AND_ASSIGN(void* dst_host, AllocateHostBuffer(kCount));

  TF_ASSERT_OK(SyclMemcpyDeviceToHost(kDefaultDeviceOrdinal, dst_host,
                                      src_device, sizeof(int) * kCount));

  VerifyIntBuffer(dst_host, kCount, 0xDEADBEEF);

  FreeAndNullify(src_device);
  FreeAndNullify(dst_host);
}

TEST_F(SyclGpuRuntimeTest, TestMemcopyHostToDeviceAndBack) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(void* src_host,
                          AllocateAndInitHostBuffer(kCount, 0xDEADC0DE));
  TF_ASSERT_OK_AND_ASSIGN(void* dst_device, AllocateDeviceBuffer(kCount));

  TF_ASSERT_OK(SyclMemcpyHostToDevice(kDefaultDeviceOrdinal, dst_device,
                                      src_host, sizeof(int) * kCount));

  // Clear out the host buffer to ensure data is copied back correctly.
  for (int i = 0; i < kCount; ++i) {
    static_cast<int*>(src_host)[i] = 0;
  }

  TF_ASSERT_OK(SyclMemcpyDeviceToHost(kDefaultDeviceOrdinal, src_host,
                                      dst_device, sizeof(int) * kCount));

  VerifyIntBuffer(src_host, kCount, 0xDEADC0DE);

  FreeAndNullify(src_host);
  FreeAndNullify(dst_device);
}

TEST_F(SyclGpuRuntimeTest, TestMemcopyDeviceToDevice_SameDevice) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(void* src_device, AllocateDeviceBuffer(kCount));
  TF_ASSERT_OK_AND_ASSIGN(void* dst_device, AllocateDeviceBuffer(kCount));

  // Test memcpy between two buffers within the same device.
  TF_ASSERT_OK(SyclMemcpyDeviceToDevice(kDefaultDeviceOrdinal, dst_device,
                                        src_device, sizeof(int) * kCount));

  FreeAndNullify(src_device);
  FreeAndNullify(dst_device);
}

TEST_F(SyclGpuRuntimeTest, TestMemcopyDeviceToHostAsync) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(void* src_device,
                          AllocateAndInitDeviceBuffer(kCount, 0xDEADBEEF));
  TF_ASSERT_OK_AND_ASSIGN(void* dst_host, AllocateHostBuffer(kCount));

  TF_ASSERT_OK(SyclMemcpyDeviceToHostAsync(stream_handle.get(), dst_host,
                                           src_device, sizeof(int) * kCount));

  // Synchronize the stream to ensure the copy is complete before checking
  // results.
  TF_ASSERT_OK(SyclStreamSynchronize(stream_handle.get()));

  // Check the results after synchronization.
  VerifyIntBuffer(dst_host, kCount, 0xDEADBEEF);

  FreeAndNullify(src_device);
  FreeAndNullify(dst_host);

  // Destroy the stream after use.
  TF_ASSERT_OK(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle));
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestMemcopyHostToDeviceAsync) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(void* src_host,
                          AllocateAndInitHostBuffer(kCount, 0xDEADC0DE));
  TF_ASSERT_OK_AND_ASSIGN(void* dst_device, AllocateDeviceBuffer(kCount));

  TF_ASSERT_OK(SyclMemcpyHostToDeviceAsync(stream_handle.get(), dst_device,
                                           src_host, sizeof(int) * kCount));

  // Synchronize the stream to ensure the copy is complete before checking
  // results.
  TF_ASSERT_OK(SyclStreamSynchronize(stream_handle.get()));

  // Verify the copy by reading back to host.
  // First, clear out the host buffer to ensure data is copied back correctly.
  for (int i = 0; i < kCount; ++i) {
    static_cast<int*>(src_host)[i] = 0;
  }

  TF_ASSERT_OK(SyclMemcpyDeviceToHost(kDefaultDeviceOrdinal, src_host,
                                      dst_device, sizeof(int) * kCount));

  VerifyIntBuffer(src_host, kCount, 0xDEADC0DE);

  FreeAndNullify(src_host);
  FreeAndNullify(dst_device);

  // Destroy the stream after use.
  TF_ASSERT_OK(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle));
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestMemsetDevice) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(
      void* src_device,
      SyclMallocDevice(kDefaultDeviceOrdinal, sizeof(char) * kCount));
  ASSERT_NE(src_device, nullptr);

  TF_ASSERT_OK(SyclMemsetDevice(kDefaultDeviceOrdinal, src_device, 'A',
                                sizeof(char) * kCount));

  TF_ASSERT_OK_AND_ASSIGN(void* dst_host, AllocateHostBuffer(kCount));

  TF_ASSERT_OK(SyclMemcpyDeviceToHost(kDefaultDeviceOrdinal, dst_host,
                                      src_device, sizeof(char) * kCount));

  for (int i = 0; i < kCount; ++i) {
    EXPECT_EQ(static_cast<char*>(dst_host)[i], 'A')
        << "Mismatch at index " << i;
  }

  FreeAndNullify(src_device);
  FreeAndNullify(dst_host);
}

TEST_F(SyclGpuRuntimeTest, TestMemsetDevice_Negative) {
  constexpr int kCount = 10;
  constexpr int kInvalidDeviceOrdinal = -1;

  TF_ASSERT_OK_AND_ASSIGN(void* src_device, AllocateDeviceBuffer(kCount));
  ASSERT_NE(src_device, nullptr);

  // Attempt to memset with an invalid device ordinal.
  EXPECT_THAT(SyclMemsetDevice(kInvalidDeviceOrdinal, src_device, 'A',
                               sizeof(char) * kCount),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Attempt to memset a null pointer.
  void* null_ptr = nullptr;
  EXPECT_THAT(SyclMemsetDevice(kDefaultDeviceOrdinal, null_ptr, 'A',
                               sizeof(char) * kCount),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  FreeAndNullify(src_device);
}

TEST_F(SyclGpuRuntimeTest, TestMemsetDeviceAsync) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(void* device_buf, AllocateDeviceBuffer(kCount));

  TF_ASSERT_OK(SyclMemsetDeviceAsync(stream_handle.get(), device_buf, 'B',
                                     sizeof(char) * kCount));

  // Synchronize the stream to ensure the memset is complete before checking
  // results.
  TF_ASSERT_OK(SyclStreamSynchronize(stream_handle.get()));

  TF_ASSERT_OK_AND_ASSIGN(void* host_buf, AllocateHostBuffer(kCount));

  TF_ASSERT_OK(SyclMemcpyDeviceToHost(kDefaultDeviceOrdinal, host_buf,
                                      device_buf, sizeof(char) * kCount));

  for (int i = 0; i < kCount; ++i) {
    EXPECT_EQ(static_cast<char*>(host_buf)[i], 'B')
        << "Mismatch at index " << i;
  }

  FreeAndNullify(device_buf);
  FreeAndNullify(host_buf);

  // Destroy the stream after use.
  TF_ASSERT_OK(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle));
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestMemfillDeviceAsync) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(void* device_buf, AllocateDeviceBuffer(kCount));

  TF_ASSERT_OK(SyclMemfillDeviceAsync(stream_handle.get(), device_buf,
                                      0xDEADC0DE, sizeof(int) * kCount));

  // Synchronize the stream to ensure the fill is complete before checking
  // results.
  TF_ASSERT_OK(SyclStreamSynchronize(stream_handle.get()));

  TF_ASSERT_OK_AND_ASSIGN(void* host_buf, AllocateHostBuffer(kCount));

  TF_ASSERT_OK(SyclMemcpyDeviceToHost(kDefaultDeviceOrdinal, host_buf,
                                      device_buf, sizeof(int) * kCount));

  VerifyIntBuffer(host_buf, kCount, 0xDEADC0DE);

  FreeAndNullify(device_buf);
  FreeAndNullify(host_buf);

  // Destroy the stream after use.
  TF_ASSERT_OK(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle));
  ASSERT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestMemfillDeviceAsync_Negative) {
  constexpr int kCount = 10;
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream_handle,
      SyclStreamPool::GetOrCreateStream(kDefaultDeviceOrdinal,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream_handle, nullptr);

  // Attempt to fill a null pointer.
  void* null_ptr = nullptr;
  EXPECT_THAT(SyclMemfillDeviceAsync(stream_handle.get(), null_ptr, 0xFEEDEAF,
                                     sizeof(int) * kCount),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Destroy the stream after use.
  TF_ASSERT_OK(
      SyclStreamPool::DestroyStream(kDefaultDeviceOrdinal, stream_handle));
  EXPECT_EQ(stream_handle, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestMultiDeviceAllocationAndSyncCopy) {
  // Skip if less than 2 devices are available.
  if (sycl_devices_.size() < 2) {
    GTEST_SKIP() << "Not enough SYCL devices available for this test.";
  }

  constexpr int kDevice0 = 0, kDevice1 = 1;
  constexpr int kCount = 16;

  // Allocate and initialize on device 0.
  TF_ASSERT_OK_AND_ASSIGN(void* device0_buf, AllocateAndInitDeviceBuffer(
                                                 kCount, 0x1234ABCD, kDevice0));
  // Allocate on device 1.
  TF_ASSERT_OK_AND_ASSIGN(void* device1_buf,
                          AllocateDeviceBuffer(kCount, kDevice1));

  // Try to copy from device 0 to device 1. It should work since cross-device
  // memcpy is supported.
  TF_ASSERT_OK(SyclMemcpyDeviceToDevice(kDevice0, device1_buf, device0_buf,
                                        sizeof(int) * kCount));

  // Verify the copy by reading back to host.
  TF_ASSERT_OK_AND_ASSIGN(void* host_buf, AllocateHostBuffer(kCount));

  TF_ASSERT_OK(SyclMemcpyDeviceToHost(kDevice1, host_buf, device1_buf,
                                      sizeof(int) * kCount));

  VerifyIntBuffer(host_buf, kCount, 0x1234ABCD);

  // Free the buffers.
  FreeAndNullify(device0_buf, kDevice0);
  FreeAndNullify(device1_buf, kDevice1);
  FreeAndNullify(host_buf, kDefaultDeviceOrdinal);
}

TEST_F(SyclGpuRuntimeTest, TestMultiDeviceAllocationAndAsyncCopy) {
  if (sycl_devices_.size() < 2) {
    GTEST_SKIP() << "Not enough SYCL devices available for this test.";
  }

  constexpr int kDevice0 = 0, kDevice1 = 1;
  constexpr int kCount = 10;

  // Create a stream for device-0.
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPtr stream0,
      SyclStreamPool::GetOrCreateStream(kDevice0,
                                        /*enable_multiple_streams=*/false));
  ASSERT_NE(stream0, nullptr);

  // Allocate and initialize on device-0.
  TF_ASSERT_OK_AND_ASSIGN(void* device0_buf, AllocateAndInitDeviceBuffer(
                                                 kCount, 0xDEADBEEF, kDevice0));

  // Allocate on device 1.
  TF_ASSERT_OK_AND_ASSIGN(void* device1_buf,
                          AllocateDeviceBuffer(kCount, kDevice1));

  // Copy from device-0 to device-1 using stream-0.
  TF_ASSERT_OK(SyclMemcpyDeviceToDeviceAsync(
      stream0.get(), device1_buf, device0_buf, sizeof(int) * kCount));

  // Synchronize the stream to ensure the copy is complete.
  TF_ASSERT_OK(SyclStreamSynchronize(stream0.get()));

  // Verify the copy by copying back to host.
  TF_ASSERT_OK_AND_ASSIGN(void* host_buf, AllocateHostBuffer(kCount));

  TF_ASSERT_OK(SyclMemcpyDeviceToHost(kDevice1, host_buf, device1_buf,
                                      sizeof(int) * kCount));

  VerifyIntBuffer(host_buf, kCount, 0xDEADBEEF);

  // Free the buffers.
  FreeAndNullify(device0_buf, kDevice0);
  FreeAndNullify(device1_buf, kDevice1);
  FreeAndNullify(host_buf, kDefaultDeviceOrdinal);

  // Destroy the stream after use.
  TF_ASSERT_OK(SyclStreamPool::DestroyStream(kDevice0, stream0));
  EXPECT_EQ(stream0, nullptr);
}

TEST_F(SyclGpuRuntimeTest, TestMallocAll_Positive) {
  TF_ASSERT_OK_AND_ASSIGN(void* host_ptr, AllocateHostBuffer(/*count=*/256));
  FreeAndNullify(host_ptr);

  TF_ASSERT_OK_AND_ASSIGN(void* device_ptr,
                          AllocateDeviceBuffer(/*count=*/256));
  FreeAndNullify(device_ptr);

  TF_ASSERT_OK_AND_ASSIGN(void* shared_ptr,
                          SyclMallocShared(kDefaultDeviceOrdinal,
                                           /*byte_count=*/1024));
  EXPECT_NE(shared_ptr, nullptr);
  FreeAndNullify(shared_ptr);
}

TEST_F(SyclGpuRuntimeTest, TestMallocAll_InvalidDeviceOrdinal) {
  constexpr int kInvalidDeviceOrdinal = -1;
  EXPECT_THAT(SyclMallocHost(kInvalidDeviceOrdinal, 10).status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SyclMallocDevice(kInvalidDeviceOrdinal, 20).status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SyclMallocShared(kInvalidDeviceOrdinal, 30).status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SyclGpuRuntimeTest, TestMallocAll_ZeroAllocation) {
  constexpr size_t kByteCount = 0;
  TF_ASSERT_OK_AND_ASSIGN(void* host_ptr,
                          SyclMallocHost(kDefaultDeviceOrdinal, kByteCount));
  EXPECT_EQ(host_ptr, nullptr)
      << "Expected nullptr for zero allocation on host memory.";
  FreeAndNullify(host_ptr);

  TF_ASSERT_OK_AND_ASSIGN(void* device_ptr,
                          SyclMallocDevice(kDefaultDeviceOrdinal, kByteCount));
  EXPECT_EQ(device_ptr, nullptr)
      << "Expected nullptr for zero allocation on device memory.";
  FreeAndNullify(device_ptr);

  TF_ASSERT_OK_AND_ASSIGN(void* shared_ptr,
                          SyclMallocShared(kDefaultDeviceOrdinal, kByteCount));
  EXPECT_EQ(shared_ptr, nullptr)
      << "Expected nullptr for zero allocation on shared memory.";
  FreeAndNullify(shared_ptr);
}

TEST_F(SyclGpuRuntimeTest, TestSyclFree_Negative) {
  constexpr int kInvalidDeviceOrdinal = -1;
  void* null_ptr = nullptr;  // Null pointer should not cause issues.

  // Attempt to free with an invalid device ordinal.
  EXPECT_THAT(SyclFree(kInvalidDeviceOrdinal, null_ptr),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Attempt to free a null pointer.
  EXPECT_THAT(SyclFree(kDefaultDeviceOrdinal, null_ptr),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument))
      << "Expected error when trying to free a null pointer.";
}

TEST_F(SyclGpuRuntimeTest, TestSyclFree_DoubleFree) {
  TF_ASSERT_OK_AND_ASSIGN(void* device_ptr, AllocateDeviceBuffer(10));
  TF_ASSERT_OK(SyclFree(kDefaultDeviceOrdinal, device_ptr));
  EXPECT_EQ(device_ptr, nullptr);

  // Try to free again, which should return an error.
  EXPECT_THAT(SyclFree(kDefaultDeviceOrdinal, device_ptr),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace stream_executor::sycl
