/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/common/pjrt_state.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace {

using tensorflow::PjRtState;
using ::testing::HasSubstr;

using ::tensorflow::testing::StatusIs;

class PjRtStateTestFixture : public testing::Test {
 protected:
  PjRtStateTestFixture() { pjrt_state_ = PjRtState::Create(); }
  ~PjRtStateTestFixture() override {
    tensorflow::core::ScopedUnref pjrt_state_ref(pjrt_state_);
  }
  PjRtState* pjrt_state_;
};

TEST_F(PjRtStateTestFixture, SetAndGetPjRtClient) {
  TF_ASSERT_OK(pjrt_state_->SetPjRtClient(
      tensorflow::DEVICE_CPU,
      xla::GetTfrtCpuClient(/*asynchronous=*/true, /*cpu_device_count=*/1)
          .value()));
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client,
                          pjrt_state_->GetPjRtClient(tensorflow::DEVICE_CPU));
  EXPECT_THAT(pjrt_client, testing::NotNull());
}

TEST_F(PjRtStateTestFixture, AddAlreadyExistsPjRtClient) {
  TF_ASSERT_OK(pjrt_state_->SetPjRtClient(
      tensorflow::DEVICE_CPU,
      xla::GetTfrtCpuClient(/*asynchronous=*/true, /*cpu_device_count=*/1)
          .value()));
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client_1,
                          pjrt_state_->GetPjRtClient(tensorflow::DEVICE_CPU));
  TF_ASSERT_OK(pjrt_state_->SetPjRtClient(
      tensorflow::DEVICE_CPU, xla::GetTfrtCpuClient(/*asynchronous=*/true,
                                                    /*cpu_device_count=*/1)
                                  .value()));
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client_2,
                          pjrt_state_->GetPjRtClient(tensorflow::DEVICE_CPU));

  EXPECT_NE(pjrt_client_1, pjrt_client_2);
}

TEST_F(PjRtStateTestFixture, GetNotExistPjRtClient) {
  EXPECT_THAT(pjrt_state_->GetPjRtClient(tensorflow::DEVICE_CPU),
              StatusIs(tensorflow::error::NOT_FOUND,
                       HasSubstr("PjRt client not found for device type")));
}

TEST_F(PjRtStateTestFixture, DeletePjRtClient) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_client,
      xla::GetTfrtCpuClient(/*asynchronous=*/true, /*cpu_device_count=*/1));
  xla::PjRtClient* pjrt_client_ptr = pjrt_client.get();
  TF_ASSERT_OK(pjrt_state_->SetPjRtClient(tensorflow::DEVICE_CPU,
                                          std::move(pjrt_client)));

  TF_ASSERT_OK(pjrt_state_->MovePjRtClientToUnused(tensorflow::DEVICE_CPU));

  EXPECT_THAT(pjrt_state_->GetPjRtClient(tensorflow::DEVICE_CPU),
              StatusIs(tensorflow::error::NOT_FOUND,
                       HasSubstr("PjRt client not found for device type")));
  // Verifies that the PJRT client is still alive.
  EXPECT_EQ(pjrt_client_ptr->platform_name(), "cpu");
}

TEST_F(PjRtStateTestFixture, DeleteNotExistPjRtClient) {
  EXPECT_THAT(pjrt_state_->MovePjRtClientToUnused(tensorflow::DEVICE_CPU),
              StatusIs(tensorflow::error::NOT_FOUND,
                       HasSubstr("PjRt client not found for device type")));
}

TEST_F(PjRtStateTestFixture, GetOrCreatePjRtClientExist) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_client,
      xla::GetTfrtCpuClient(/*asynchronous=*/true, /*cpu_device_count=*/1));
  auto pjrt_client_ptr = pjrt_client.get();
  TF_ASSERT_OK(pjrt_state_->SetPjRtClient(tensorflow::DEVICE_CPU,
                                          std::move(pjrt_client)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_client_get,
      pjrt_state_->GetOrCreatePjRtClient(tensorflow::DEVICE_CPU));
  EXPECT_THAT(pjrt_client_get, pjrt_client_ptr);
}

TEST_F(PjRtStateTestFixture, GetOrCreatePjRtClientNotExist) {
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, pjrt_state_->GetOrCreatePjRtClient(
                                                tensorflow::DEVICE_CPU));
  EXPECT_THAT(pjrt_client, testing::NotNull());
}

}  // namespace
