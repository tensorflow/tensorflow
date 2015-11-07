#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_

#include <deque>
#include <vector>
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/tensor.h"

namespace perftools {
namespace gputools {
class Event;
class Stream;
class StreamExecutor;
}  // namespace gputools
}  // namespace perftools

namespace tensorflow {

// An object to keep track of pending Events in the StreamExecutor streams
// and associated Tensors that cannot safely be deleted until the associated
// Events are recorded.
class EventMgr {
 public:
  explicit EventMgr(perftools::gputools::StreamExecutor* se);

  ~EventMgr();

  // Takes ownership of *tensors and deletes it as soon as all events
  // currently enqueued on *stream have completed.
  inline void ThenDeleteTensors(perftools::gputools::Stream* stream,
                                std::vector<Tensor>* tensors) {
    mutex_lock l(mu_);
    QueueTensors(stream, tensors);
    PollEvents(false);
  }

  struct BufRec {
    Allocator* alloc;
    void* buf;
  };

  // Takes ownership of *bufrec.buf and calls bufrec.alloc->DeallocateRaw()
  // on it as soon as all events currently enqueued on *stream have completed.
  inline void ThenDeleteBuffer(perftools::gputools::Stream* stream,
                               BufRec bufrec) {
    mutex_lock l(mu_);
    QueueBuffer(stream, bufrec);
    PollEvents(false);
  }

  inline void ThenExecute(perftools::gputools::Stream* stream,
                          std::function<void()> func) {
    mutex_lock l(mu_);
    QueueFunc(stream, func);
    PollEvents(false);
  }

 private:
  friend class TEST_EventMgrHelper;
  mutex mu_;
  perftools::gputools::StreamExecutor* exec_;

  struct InUse {
    perftools::gputools::Event* event;
    std::vector<Tensor>* mem;
    BufRec bufrec;
    std::function<void()> func;
  };

  // Stream-enqueue an unused Event and save with it a collection of
  // Tensors and/or a BufRec to be deleted only after the Event
  // records.
  void QueueInUse(perftools::gputools::Stream* stream, InUse in_use)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void QueueTensors(perftools::gputools::Stream* stream,
                    std::vector<Tensor>* tensors)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    QueueInUse(stream, {nullptr, tensors, BufRec(), nullptr});
  }

  void QueueBuffer(perftools::gputools::Stream* stream, BufRec bufrec)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    QueueInUse(stream, {nullptr, nullptr, bufrec, nullptr});
  }

  void QueueFunc(perftools::gputools::Stream* stream,
                 std::function<void()> func) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    QueueInUse(stream, {nullptr, nullptr, BufRec(), func});
  }

  // This function should be called at roughly the same tempo as
  // QueueTensors() to check whether pending events have recorded,
  // and then retire them.
  void PollEvents(bool is_dedicated_poller) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // An internal polling loop that runs at a low frequency to clear
  // straggler Events.
  void PollLoop();

  // A stack of unused events
  std::vector<perftools::gputools::Event*> free_events_ GUARDED_BY(mu_);

  // A FIFO queue of InUse events and associated tensors.
  std::deque<InUse> used_events_ GUARDED_BY(mu_);

  Notification stop_polling_;
  Notification polling_stopped_;

  // The main PollLoop for the event manager runs in this threadpool.
  thread::ThreadPool threadpool_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_
