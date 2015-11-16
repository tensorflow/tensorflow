#if GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"

#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include <gtest/gtest.h>
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"

namespace gpu = ::perftools::gputools;

namespace tensorflow {

class TEST_EventMgrHelper {
 public:
  explicit TEST_EventMgrHelper(EventMgr* em) : em_(em) {}

  int queue_size() {
    mutex_lock l(em_->mu_);
    return em_->used_events_.size();
  }

  int free_size() {
    mutex_lock l(em_->mu_);
    return em_->free_events_.size();
  }

  void QueueTensors(perftools::gputools::Stream* stream,
                    std::vector<Tensor>* tensors) {
    mutex_lock l(em_->mu_);
    em_->QueueTensors(stream, tensors);
  }

  void PollEvents(bool is_dedicated_poller) {
    mutex_lock l(em_->mu_);
    em_->PollEvents(is_dedicated_poller);
  }

 private:
  EventMgr* em_;
};

namespace {

TEST(EventMgr, Empty) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec);
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, th.queue_size());
  EXPECT_EQ(0, th.free_size());
}

// Delaying polling until after several enqueings should grow the
// total number of allocated events.  Once we have enough events for
// the max simultaneously pending, we should not allocate any more.
TEST(EventMgr, DelayedPolling) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec);
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, th.queue_size());
  std::vector<Tensor>* v = nullptr;
  std::unique_ptr<gpu::Stream> stream(new gpu::Stream(stream_exec));
  CHECK(stream.get());
  stream->Init();
  for (int i = 0; i < 5; ++i) {
    v = new std::vector<Tensor>;
    th.QueueTensors(stream.get(), v);
    EXPECT_EQ(i + 1, th.queue_size());
    EXPECT_EQ(0, th.free_size());
  }
  th.PollEvents(false);
  EXPECT_EQ(0, th.queue_size());
  EXPECT_EQ(5, th.free_size());
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 5; ++i) {
      v = new std::vector<Tensor>;
      th.QueueTensors(stream.get(), v);
      EXPECT_EQ(i + 1, th.queue_size());
      EXPECT_EQ(4 - i, th.free_size());
    }
    th.PollEvents(false);
    EXPECT_EQ(0, th.queue_size());
    EXPECT_EQ(5, th.free_size());
  }
}

// Immediate polling should require only one event to be allocated.
TEST(EventMgr, ImmediatePolling) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec);
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, th.queue_size());
  EXPECT_EQ(0, th.free_size());
  std::vector<Tensor>* v = nullptr;
  std::unique_ptr<gpu::Stream> stream(new gpu::Stream(stream_exec));
  CHECK(stream.get());
  stream->Init();
  for (int i = 0; i < 5; ++i) {
    v = new std::vector<Tensor>;
    em.ThenDeleteTensors(stream.get(), v);
    EXPECT_EQ(0, th.queue_size());
    EXPECT_EQ(1, th.free_size());
  }
}

// If we delay polling by more than 1 second, the backup polling loop
// should clear the queue.
TEST(EventMgr, LongDelayedPolling) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec);
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, th.queue_size());
  EXPECT_EQ(0, th.free_size());
  std::vector<Tensor>* v = nullptr;
  std::unique_ptr<gpu::Stream> stream(new gpu::Stream(stream_exec));
  CHECK(stream.get());
  stream->Init();
  for (int i = 0; i < 5; ++i) {
    v = new std::vector<Tensor>;
    th.QueueTensors(stream.get(), v);
    EXPECT_EQ(1 + i, th.queue_size());
    EXPECT_EQ(0, th.free_size());
  }
  sleep(1);
  EXPECT_EQ(0, th.queue_size());
  EXPECT_EQ(5, th.free_size());
}

// Deleting the EventMgr when events are still pending should shut
// down gracefully.
TEST(EventMgr, NonEmptyShutdown) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec);
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, th.queue_size());
  EXPECT_EQ(0, th.free_size());
  std::vector<Tensor>* v = nullptr;
  std::unique_ptr<gpu::Stream> stream(new gpu::Stream(stream_exec));
  CHECK(stream.get());
  stream->Init();
  for (int i = 0; i < 5; ++i) {
    v = new std::vector<Tensor>;
    th.QueueTensors(stream.get(), v);
    EXPECT_EQ(1 + i, th.queue_size());
    EXPECT_EQ(0, th.free_size());
  }
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
