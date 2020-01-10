#ifndef TENSORFLOW_LIB_CORE_BLOCKING_COUNTER_H_
#define TENSORFLOW_LIB_CORE_BLOCKING_COUNTER_H_

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

class BlockingCounter {
 public:
  BlockingCounter(int initial_count) : count_(initial_count) {
    CHECK_GE(count_, 0);
  }

  ~BlockingCounter() {}

  inline void DecrementCount() {
    mutex_lock l(mu_);
    --count_;
    CHECK(count_ >= 0);
    if (count_ == 0) {
      cond_var_.notify_all();
    }
  }

  inline void Wait() {
    mutex_lock l(mu_);
    while (count_ > 0) {
      cond_var_.wait(l);
    }
  }

 private:
  int count_;
  mutex mu_;
  condition_variable cond_var_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_CORE_BLOCKING_COUNTER_H_
