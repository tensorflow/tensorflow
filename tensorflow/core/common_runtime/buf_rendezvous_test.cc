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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

#define NUM_DEVS 3

class BufRendezvousTest : public ::testing::Test {
 protected:
  BufRendezvousTest() {
    br_.reset(new BufRendezvous(123));
    fake_dev_ptr_ = reinterpret_cast<Device*>(512LLU);
    fake_dev_ctx_ = reinterpret_cast<DeviceContext*>(1024LLU);
    a_ = Tensor(DT_FLOAT, TensorShape({24}));
    b_ = Tensor(DT_FLOAT, TensorShape({24}));
  }

  Device* fake_dev_ptr_ = nullptr;
  DeviceContext* fake_dev_ctx_ = nullptr;
  Tensor a_;
  Tensor b_;
  AllocatorAttributes aa_;
  std::unique_ptr<BufRendezvous> br_;
};

TEST_F(BufRendezvousTest, CorrectUseProducerFirst) {
  Status prod_status;
  Status cons_status;
  bool prod_callback_called = false;
  bool cons_callback_called = false;
  Notification note;
  br_->ProvideBuf(
      "key0", fake_dev_ptr_, fake_dev_ctx_, &a_, aa_,
      [&note, &prod_status, &prod_callback_called](const Status& s) {
        prod_status = s;
        prod_callback_called = true;
        note.Notify();
      });
  EXPECT_FALSE(prod_callback_called);
  br_->ConsumeBuf("key0", [this, &cons_status, &cons_callback_called](
                              const Status& s, BufRendezvous::Hook* h) {
    cons_status = s;
    cons_callback_called = true;
    ASSERT_TRUE(h != nullptr);
    EXPECT_EQ(h->prod_dev, fake_dev_ptr_);
    EXPECT_EQ(h->prod_ctx, fake_dev_ctx_);
    EXPECT_EQ(h->prod_value, &a_);
    br_->DoneWithHook(h);
  });
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
  br_->ConsumeBuf("key0", [this, &cons_status, &cons_callback_called](
                              const Status& s, BufRendezvous::Hook* h) {
    cons_status = s;
    cons_callback_called = true;
    ASSERT_TRUE(h != nullptr);
    EXPECT_EQ(h->prod_dev, fake_dev_ptr_);
    EXPECT_EQ(h->prod_ctx, fake_dev_ctx_);
    EXPECT_EQ(h->prod_value, &a_);
    br_->DoneWithHook(h);
  });
  EXPECT_FALSE(cons_callback_called);
  br_->ProvideBuf(
      "key0", fake_dev_ptr_, fake_dev_ctx_, &a_, aa_,
      [&note, &prod_status, &prod_callback_called](const Status& s) {
        prod_status = s;
        prod_callback_called = true;
        note.Notify();
      });
  EXPECT_TRUE(cons_callback_called);
  note.WaitForNotification();
  EXPECT_TRUE(prod_callback_called);
  TF_EXPECT_OK(cons_status);
  TF_EXPECT_OK(prod_status);
}

TEST_F(BufRendezvousTest, ErrorDuplicatePut) {
  bool prod_callback_called = false;
  br_->ProvideBuf("key0", fake_dev_ptr_, fake_dev_ctx_, &a_, aa_,
                  [this, &prod_callback_called](const Status& s) {
                    prod_callback_called = true;
                  });
  Status bad_status;
  Notification note;
  br_->ProvideBuf("key0", fake_dev_ptr_, fake_dev_ctx_, &a_, aa_,
                  [&bad_status, &note](const Status& s) {
                    bad_status = s;
                    note.Notify();
                  });
  note.WaitForNotification();
  EXPECT_FALSE(bad_status.ok());
  EXPECT_EQ("BufRendezvous::ProvideBuf already called for key key0",
            bad_status.error_message());
  EXPECT_FALSE(prod_callback_called);
  br_.reset();
}

TEST_F(BufRendezvousTest, ErrorDeleteNonEmpty) {
  Status cons_status;
  br_->ConsumeBuf(
      "key0", [this, &cons_status](const Status& s, BufRendezvous::Hook* h) {
        cons_status = s;
        EXPECT_EQ(h, nullptr);
      });
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
  br_->ConsumeBuf("key0", [this, &cons_note, &cons_status](
                              const Status& s, BufRendezvous::Hook* h) {
    cons_status = s;
    cons_note.Notify();
  });
  br_->ProvideBuf("key1", fake_dev_ptr_, fake_dev_ctx_, &a_, aa_,
                  [this, &prod_note, &prod_status](const Status& s) {
                    prod_status = s;
                    prod_note.Notify();
                  });
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
  br_->ConsumeBuf("key0", [this, &cons_note, &cons_status](
                              const Status& s, BufRendezvous::Hook* h) {
    cons_status = s;
    cons_note.Notify();
  });
  br_->ProvideBuf("key1", fake_dev_ptr_, fake_dev_ctx_, &a_, aa_,
                  [this, &prod_note, &prod_status](const Status& s) {
                    prod_status = s;
                    prod_note.Notify();
                  });
  prod_note.WaitForNotification();
  cons_note.WaitForNotification();
  EXPECT_FALSE(prod_status.ok());
  EXPECT_EQ(prod_status.error_message(), "Falling sky detected");
  EXPECT_FALSE(cons_status.ok());
  EXPECT_EQ(cons_status.error_message(), "Falling sky detected");
}

}  // namespace
}  // namespace tensorflow
