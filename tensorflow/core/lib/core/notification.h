#ifndef TENSORFLOW_UTIL_NOTIFICATION_H_
#define TENSORFLOW_UTIL_NOTIFICATION_H_

#include <assert.h>

#include "tensorflow/core/platform/port.h"

namespace tensorflow {

class Notification {
 public:
  Notification() : notified_(false) {}
  ~Notification() {}

  void Notify() {
    mutex_lock l(mu_);
    assert(!notified_);
    notified_ = true;
    cv_.notify_all();
  }

  bool HasBeenNotified() {
    mutex_lock l(mu_);
    return notified_;
  }

  void WaitForNotification() {
    mutex_lock l(mu_);
    while (!notified_) {
      cv_.wait(l);
    }
  }

 private:
  mutex mu_;
  condition_variable cv_;
  bool notified_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_NOTIFICATION_H_
