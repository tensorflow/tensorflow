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

#if GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"

#include <atomic>
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class TEST_EventMgrHelper {
 public:
  explicit TEST_EventMgrHelper(EventMgr* em) : em_(em) {
    // The polling loop can interfere with the measurements made here, and
    // isn't needed since the member PollEvents() always clears the queue.
    // The tested behavior is slightly different from what may occur in
    // ordinary execution.
    StopPollingLoop();
  }

  size_t queue_size() {
    mutex_lock l(em_->mu_);
    return em_->used_events_.size();
  }

  size_t free_size() {
    mutex_lock l(em_->mu_);
    return em_->free_events_.size();
  }

  void QueueTensors(se::Stream* stream, TensorReferenceVector* tensors) {
    mutex_lock l(em_->mu_);
    em_->QueueTensors(stream, tensors);
  }

  void PollEvents(bool is_dedicated_poller) {
    while (queue_size() > 0) {
      // For ordinary tensor frees, this function
      // should synchronously harvest all complete
      // events and execute the corresponding memory frees.
      EventMgr::ToFreeVector to_free;
      {
        mutex_lock l(em_->mu_);
        em_->PollEvents(is_dedicated_poller, &to_free);
      }
      em_->FreeMemory(to_free);
    }
  }

  void StopPollingLoop() { em_->StopPollingLoop(); }

  void StartPollingLoop() { em_->StartPollingLoop(); }

 private:
  EventMgr* em_;
};

static std::atomic_int_fast64_t live_tensor_bytes(0);

// A TensorBuffer that counts live memory usage for testing
class TestTensorBuffer : public TensorBuffer {
 public:
  explicit TestTensorBuffer(size_t bytes)
      : TensorBuffer(nullptr), bytes_(bytes) {
    live_tensor_bytes += bytes_;
  }
  ~TestTensorBuffer() override { live_tensor_bytes -= bytes_; }

  size_t size() const override { return bytes_; }

  // Not used in this test
  TensorBuffer* root_buffer() override { return nullptr; }
  void FillAllocationDescription(AllocationDescription* arg) const override {}

 private:
  size_t bytes_;
};

namespace {

TEST(EventMgr, Empty) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, th.queue_size());
  EXPECT_EQ(0, th.free_size());
}

static void AddTensorReference(TensorReferenceVector* v, int64 size) {
  TestTensorBuffer* buf = new TestTensorBuffer(size);
  v->push_back(TensorReference(buf));
  buf->Unref();
}

// Delaying polling until after several enqueings should grow the
// total number of allocated events.  Once we have enough events for
// the max simultaneously pending, we should not allocate any more.
TEST(EventMgr, DelayedPolling) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, th.queue_size());
  TensorReferenceVector* v = nullptr;
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  for (int i = 0; i < 5; ++i) {
    v = new TensorReferenceVector;
    AddTensorReference(v, 100 * 1048576);
    th.QueueTensors(stream.get(), v);
    EXPECT_EQ(i + 1, th.queue_size());
    EXPECT_EQ(0, th.free_size());
  }
  th.PollEvents(false);
  EXPECT_EQ(0, th.queue_size());
  EXPECT_EQ(5, th.free_size());
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 5; ++i) {
      v = new TensorReferenceVector;
      AddTensorReference(v, 100 * 1048576);
      th.QueueTensors(stream.get(), v);
      EXPECT_EQ(i + 1, th.queue_size());
      EXPECT_EQ(4 - i, th.free_size());
    }
    th.PollEvents(false);
    EXPECT_EQ(0, th.queue_size());
    EXPECT_EQ(5, th.free_size());
  }
}

TEST(EventMgr, FlushLargeTensorImmediately) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, live_tensor_bytes);
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  for (int i = 0; i < 5; ++i) {
    TensorReferenceVector v;
    AddTensorReference(&v, 100 * 1048576);
    em.ThenDeleteTensors(stream.get(), v);
    th.PollEvents(false);  // Ensure things get registered to be freed by Poll
    EXPECT_EQ(0, live_tensor_bytes);
  }
}

TEST(EventMgr, ManySmallTensorsFlushedImmediately) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, live_tensor_bytes);
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  for (int i = 0; i < 5; ++i) {
    TensorReferenceVector v;
    for (int i = 0; i < 1000; i++) {
      AddTensorReference(&v, 100 * 1024);
    }
    em.ThenDeleteTensors(stream.get(), v);
    th.PollEvents(false);  // Harvest the tensors ready to be freed.
    EXPECT_EQ(0, live_tensor_bytes);
  }
}

TEST(EventMgr, StreamSwitchingFlushesImmediately) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, live_tensor_bytes);
  std::unique_ptr<se::Stream> stream1(new se::Stream(stream_exec));
  std::unique_ptr<se::Stream> stream2(new se::Stream(stream_exec));
  stream1->Init();
  stream2->Init();
  TensorReferenceVector v1;
  AddTensorReference(&v1, 1024);
  em.ThenDeleteTensors(stream1.get(), v1);

  TensorReferenceVector v2;
  AddTensorReference(&v2, 1024);
  int64 initial_live_bytes = live_tensor_bytes;
  em.ThenDeleteTensors(stream2.get(), v2);
  th.PollEvents(false);  // Ensure things get registered to be freed by Poll
  // Different stream should cause first tensor to get deleted
  EXPECT_GT(initial_live_bytes, live_tensor_bytes);
}

TEST(EventMgr, ManySmallTensorsSeparateCallsFlushed) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, live_tensor_bytes);
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  for (int i = 0; i < 5; ++i) {
    for (int i = 0; i < 1000; i++) {
      TensorReferenceVector v;
      AddTensorReference(&v, 100 * 1024);
      em.ThenDeleteTensors(stream.get(), v);
    }
    th.PollEvents(false);  // Ensure things get registered to be freed by Poll
    // Some of the tensors at least should be flushed
    EXPECT_GT(1000 * 100 * 1024, live_tensor_bytes);
  }
}

// Deleting the EventMgr when events are still pending should shut
// down gracefully.
TEST(EventMgr, NonEmptyShutdown) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, th.queue_size());
  EXPECT_EQ(0, th.free_size());
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  for (int i = 0; i < 5; ++i) {
    TensorReferenceVector* v = new TensorReferenceVector;
    AddTensorReference(v, 100 * 1048576);
    th.QueueTensors(stream.get(), v);
    EXPECT_EQ(1 + i, th.queue_size());
    EXPECT_EQ(0, th.free_size());
  }
}

// Tests that WarnIfInCallback() triggers correctly.
TEST(EventMgr, WarnIfInCallback) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  bool hit = false;
  gpu_event_mgr::WarnIfInCallback([&hit] { hit = true; });
  EXPECT_FALSE(hit);
  Notification note;
  em.ThenExecute(stream.get(), [&hit, &note]() {
    gpu_event_mgr::WarnIfInCallback([&hit, &note] {
      hit = true;
      note.Notify();
    });
  });
  note.WaitForNotification();
  EXPECT_TRUE(hit);
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
