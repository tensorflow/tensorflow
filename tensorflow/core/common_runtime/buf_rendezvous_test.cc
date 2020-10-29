/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/buf_rendezvous.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class BufRendezvousTest : public ::testing::Test {
 protected:
  static std::unique_ptr<Device> NewDevice(const string& name,
                                           const string& type,
                                           const uint64 incarnation) {
    class FakeDevice : public Device {
     public:
      explicit FakeDevice(const DeviceAttributes& attrs)
          : Device(nullptr, attrs) {}
      Status Sync() override { return Status::OK(); }
      Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
    };
    DeviceAttributes attrs;
    attrs.set_name(name);
    attrs.set_device_type(type);
    attrs.set_incarnation(incarnation);
    return absl::make_unique<FakeDevice>(attrs);
  }

  void InitializeDevice(const string& device, const string& type,
                        const uint64 incarnation) {
    std::vector<std::unique_ptr<Device>> devices;
    devices.push_back(NewDevice(device, type, incarnation));
    dev_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
    br_ = absl::make_unique<BufRendezvous>(123, dev_mgr_.get());
  }

  BufRendezvousTest()
      : a_(Tensor(DT_FLOAT, TensorShape({24}))),
        b_(Tensor(DT_FLOAT, TensorShape({24}))),
        fake_device_context_(reinterpret_cast<DeviceContext*>(1024LLU)) {
    InitializeDevice(*kDefaultDeviceName, "CPU", kDefaultIncarnation);
    TF_CHECK_OK(dev_mgr_->LookupDevice(*kDefaultDeviceName, &default_device_));
  }

  Tensor a_;
  Tensor b_;
  AllocatorAttributes aa_;
  Device* default_device_;
  DeviceContext* fake_device_context_;
  std::unique_ptr<DeviceMgr> dev_mgr_;
  std::unique_ptr<BufRendezvous> br_;
  CancellationManager cm_;
  static const string* const kDefaultKey;
  static const string* const kDefaultDeviceName;
  static const uint64 kDefaultIncarnation;
};

const string* const BufRendezvousTest::kDefaultKey = new string("key0");
const string* const BufRendezvousTest::kDefaultDeviceName =
    new string("/device:CPU:0");
const uint64 BufRendezvousTest::kDefaultIncarnation = 12345;

TEST_F(BufRendezvousTest, CorrectUseProducerFirst) {
  Status prod_status;
  Status cons_status;
  bool prod_callback_called = false;
  bool cons_callback_called = false;
  Notification note;
  br_->ProvideBuf(
      *kDefaultKey, default_device_, fake_device_context_, &a_, aa_,
      [&note, &prod_status, &prod_callback_called](const Status& s) {
        prod_status = s;
        prod_callback_called = true;
        note.Notify();
      },
      &cm_);
  EXPECT_FALSE(prod_callback_called);
  br_->ConsumeBuf(
      *kDefaultKey, *kDefaultDeviceName, kDefaultIncarnation,
      [this, &cons_status, &cons_callback_called](const Status& s,
                                                  BufRendezvous::Hook* h) {
        cons_status = s;
        cons_callback_called = true;
        ASSERT_TRUE(h != nullptr);
        EXPECT_EQ(h->prod_dev, default_device_);
        EXPECT_EQ(h->prod_ctx, fake_device_context_);
        EXPECT_EQ(h->prod_value, &a_);
        br_->DoneWithHook(h);
      },
      &cm_);
  EXPECT_TRUE(cons_callback_called);
  note.WaitForNotification();
  EXPECT_TRUE(prod_callback_called);
  TF_EXPECT_OK(cons_status);
  TF_EXPECT_OK(prod_status);
}

TEST_F(BufRendezvousTest, CorrectUseConsumerFirst) {
  Status prod_status;
  Status cons_status;
  bool prod_callback_called = false;
  bool cons_callback_called = false;
  Notification note;
  br_->ConsumeBuf(
      *kDefaultKey, *kDefaultDeviceName, kDefaultIncarnation,
      [this, &cons_status, &cons_callback_called](const Status& s,
                                                  BufRendezvous::Hook* h) {
        cons_status = s;
        cons_callback_called = true;
        ASSERT_TRUE(h != nullptr);
        EXPECT_EQ(h->prod_dev, default_device_);
        EXPECT_EQ(h->prod_ctx, fake_device_context_);
        EXPECT_EQ(h->prod_value, &a_);
        br_->DoneWithHook(h);
      },
      &cm_);
  EXPECT_FALSE(cons_callback_called);
  br_->ProvideBuf(
      *kDefaultKey, default_device_, fake_device_context_, &a_, aa_,
      [&note, &prod_status, &prod_callback_called](const Status& s) {
        prod_status = s;
        prod_callback_called = true;
        note.Notify();
      },
      &cm_);
  EXPECT_TRUE(cons_callback_called);
  note.WaitForNotification();
  EXPECT_TRUE(prod_callback_called);
  TF_EXPECT_OK(cons_status);
  TF_EXPECT_OK(prod_status);
}

TEST_F(BufRendezvousTest, ErrorDuplicatePut) {
  bool prod_callback_called = false;
  br_->ProvideBuf(
      *kDefaultKey, default_device_, fake_device_context_, &a_, aa_,
      [&prod_callback_called](const Status& s) { prod_callback_called = true; },
      &cm_);
  Status bad_status;
  Notification note;
  br_->ProvideBuf(
      *kDefaultKey, default_device_, fake_device_context_, &a_, aa_,
      [&bad_status, &note](const Status& s) {
        bad_status = s;
        note.Notify();
      },
      &cm_);
  note.WaitForNotification();
  EXPECT_FALSE(bad_status.ok());
  EXPECT_EQ(absl::StrCat("BufRendezvous::ProvideBuf already called for key ",
                         *kDefaultKey),
            bad_status.error_message());
  EXPECT_FALSE(prod_callback_called);
  br_.reset();
}

TEST_F(BufRendezvousTest, ErrorDeleteNonEmpty) {
  Status cons_status;
  br_->ConsumeBuf(
      *kDefaultKey, *kDefaultDeviceName, kDefaultIncarnation,
      [&cons_status](const Status& s, BufRendezvous::Hook* h) {
        cons_status = s;
        EXPECT_EQ(h, nullptr);
      },
      &cm_);
  EXPECT_TRUE(cons_status.ok());
  br_.reset();
  EXPECT_FALSE(cons_status.ok());
  EXPECT_EQ("Delete called on non-empty BufRendezvous",
            cons_status.error_message());
}

TEST_F(BufRendezvousTest, AbortNonEmpty) {
  Status cons_status;
  Status prod_status;
  Notification prod_note;
  Notification cons_note;
  br_->ConsumeBuf(
      *kDefaultKey, *kDefaultDeviceName, kDefaultIncarnation,
      [&cons_note, &cons_status](const Status& s, BufRendezvous::Hook* h) {
        cons_status = s;
        cons_note.Notify();
      },
      &cm_);
  br_->ProvideBuf(
      "key1", default_device_, fake_device_context_, &a_, aa_,
      [&prod_note, &prod_status](const Status& s) {
        prod_status = s;
        prod_note.Notify();
      },
      &cm_);
  br_->StartAbort(errors::Internal("Falling sky detected"));
  prod_note.WaitForNotification();
  cons_note.WaitForNotification();
  EXPECT_FALSE(prod_status.ok());
  EXPECT_EQ(prod_status.error_message(), "Falling sky detected");
  EXPECT_FALSE(cons_status.ok());
  EXPECT_EQ(cons_status.error_message(), "Falling sky detected");
}

TEST_F(BufRendezvousTest, AbortEmpty) {
  br_->StartAbort(errors::Internal("Falling sky detected"));
}

TEST_F(BufRendezvousTest, UseAfterAbort) {
  br_->StartAbort(errors::Internal("Falling sky detected"));
  Status cons_status;
  Status prod_status;
  Notification prod_note;
  Notification cons_note;
  br_->ConsumeBuf(
      *kDefaultKey, *kDefaultDeviceName, kDefaultIncarnation,
      [&cons_note, &cons_status](const Status& s, BufRendezvous::Hook* h) {
        cons_status = s;
        cons_note.Notify();
      },
      &cm_);
  br_->ProvideBuf(
      "key1", default_device_, fake_device_context_, &a_, aa_,
      [&prod_note, &prod_status](const Status& s) {
        prod_status = s;
        prod_note.Notify();
      },
      &cm_);
  prod_note.WaitForNotification();
  cons_note.WaitForNotification();
  EXPECT_FALSE(prod_status.ok());
  EXPECT_NE(prod_status.error_message().find("Falling sky detected"),
            string::npos);
  EXPECT_FALSE(cons_status.ok());
  EXPECT_NE(cons_status.error_message().find("Falling sky detected"),
            string::npos);
}

TEST_F(BufRendezvousTest, DeviceIncarnationMismatch) {
  Status cons_status;
  Notification note;
  br_->ProvideBuf(
      *kDefaultKey, default_device_, fake_device_context_, &a_, aa_,
      [](const Status&) {}, /*cancellation_manager=*/nullptr);
  const uint64 incorrect_incarnation = 23456;
  br_->ConsumeBuf(
      *kDefaultKey, *kDefaultDeviceName, incorrect_incarnation,
      [&note, &cons_status](const Status& s, BufRendezvous::Hook* h) {
        cons_status = s;
        note.Notify();
      },
      /*cancellation_manager=*/nullptr);
  note.WaitForNotification();
  EXPECT_TRUE(errors::IsFailedPrecondition(cons_status));
}

TEST_F(BufRendezvousTest, ProvideThenCancel) {
  Status status;
  Notification note;
  br_->ProvideBuf(
      *kDefaultKey, default_device_, fake_device_context_, &a_, aa_,
      [&status, &note](const Status& s) {
        status = s;
        note.Notify();
      },
      &cm_);
  cm_.StartCancel();
  note.WaitForNotification();
  EXPECT_TRUE(errors::IsCancelled(status));
  EXPECT_NE(
      status.error_message().find(absl::StrCat(
          "Operation was cancelled for BufRendezvous key ", *kDefaultKey)),
      string::npos);
}

TEST_F(BufRendezvousTest, CancelThenProvide) {
  Status status;
  Notification note;
  cm_.StartCancel();
  br_->ProvideBuf(
      *kDefaultKey, default_device_, fake_device_context_, &a_, aa_,
      [&status, &note](const Status& s) {
        status = s;
        note.Notify();
      },
      &cm_);
  note.WaitForNotification();
  EXPECT_TRUE(errors::IsCancelled(status));
  EXPECT_NE(
      status.error_message().find(absl::StrCat(
          "Operation was cancelled for BufRendezvous key ", *kDefaultKey)),
      string::npos);
}

TEST_F(BufRendezvousTest, ConsumeThenCancel) {
  Status status;
  Notification note;
  br_->ConsumeBuf(
      *kDefaultKey, *kDefaultDeviceName, kDefaultIncarnation,
      [&status, &note](const Status& s, BufRendezvous::Hook* h) {
        status = s;
        note.Notify();
      },
      &cm_);
  cm_.StartCancel();
  note.WaitForNotification();
  EXPECT_TRUE(errors::IsCancelled(status));
  EXPECT_NE(
      status.error_message().find(absl::StrCat(
          "Operation was cancelled for BufRendezvous key ", *kDefaultKey)),
      string::npos);
}

TEST_F(BufRendezvousTest, CancelThenConsume) {
  Status status;
  Notification note;
  cm_.StartCancel();
  br_->ConsumeBuf(
      *kDefaultKey, *kDefaultDeviceName, kDefaultIncarnation,
      [&status, &note](const Status& s, BufRendezvous::Hook* h) {
        status = s;
        note.Notify();
      },
      &cm_);
  note.WaitForNotification();
  EXPECT_TRUE(errors::IsCancelled(status));
  EXPECT_NE(
      status.error_message().find(absl::StrCat(
          "Operation was cancelled for BufRendezvous key ", *kDefaultKey)),
      string::npos);
}

TEST_F(BufRendezvousTest, ProvideConsumeThenCancel) {
  Status prod_status;
  Status cons_status;
  bool prod_callback_called = false;
  bool cons_callback_called = false;
  Notification note;
  br_->ProvideBuf(
      *kDefaultKey, default_device_, fake_device_context_, &a_, aa_,
      [&note, &prod_status, &prod_callback_called](const Status& s) {
        prod_status = s;
        prod_callback_called = true;
        note.Notify();
      },
      &cm_);
  EXPECT_FALSE(prod_callback_called);
  br_->ConsumeBuf(
      *kDefaultKey, *kDefaultDeviceName, kDefaultIncarnation,
      [this, &cons_status, &cons_callback_called](const Status& s,
                                                  BufRendezvous::Hook* h) {
        cons_status = s;
        cons_callback_called = true;
        ASSERT_TRUE(h != nullptr);
        EXPECT_EQ(h->prod_dev, default_device_);
        EXPECT_EQ(h->prod_ctx, fake_device_context_);
        EXPECT_EQ(h->prod_value, &a_);
        br_->DoneWithHook(h);
      },
      &cm_);
  note.WaitForNotification();
  cm_.StartCancel();
  EXPECT_TRUE(cons_callback_called);
  EXPECT_TRUE(prod_callback_called);
  TF_EXPECT_OK(cons_status);
  TF_EXPECT_OK(prod_status);
}

TEST_F(BufRendezvousTest, CancelThenProvideConsume) {
  Status prod_status;
  Status cons_status;
  bool prod_callback_called = false;
  bool cons_callback_called = false;
  cm_.StartCancel();
  br_->ProvideBuf(
      *kDefaultKey, default_device_, fake_device_context_, &a_, aa_,
      [&prod_status, &prod_callback_called](const Status& s) {
        prod_status = s;
        EXPECT_TRUE(errors::IsCancelled(prod_status));
        prod_callback_called = true;
      },
      &cm_);
  EXPECT_TRUE(prod_callback_called);
  EXPECT_TRUE(errors::IsCancelled(prod_status));
  br_->ConsumeBuf(
      *kDefaultKey, *kDefaultDeviceName, kDefaultIncarnation,
      [&cons_status, &cons_callback_called](const Status& s,
                                            BufRendezvous::Hook* h) {
        cons_status = s;
        EXPECT_TRUE(errors::IsCancelled(cons_status));
        cons_callback_called = true;
      },
      &cm_);
  EXPECT_TRUE(cons_callback_called);
  EXPECT_TRUE(errors::IsCancelled(cons_status));
}

}  // namespace
}  // namespace tensorflow
