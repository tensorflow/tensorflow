/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/batching_util/periodic_function.h"

#include <algorithm>

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace serving {

PeriodicFunction::PeriodicFunction(const std::function<void()>& function,
                                   const int64 interval_micros,
                                   const Options& options)
    : function_(function),
      interval_micros_([interval_micros]() -> int64 {
        if (interval_micros < 0) {
          const string error = strings::StrCat(
              " The value of 'interval_micros' should be >= 0: ",
              interval_micros, ". ");
          DCHECK(false) << error;
          LOG(WARNING) << error << "Resetting it to 0.";
          return 0;
        }
        return interval_micros;
      }()),
      options_(options) {
  thread_.reset(options_.env->StartThread(
      options_.thread_options, options_.thread_name_prefix, [this]() {
        // Record the starting time here instead of in RunLoop.  That way, if
        // there is a delay starting RunLoop, that does not affect the timing
        // of
        // the first function.  (Such a delay can often happen in tests where
        // the test simulates a large time delay immediately after calling
        // Start.)
        RunLoop(options_.env->NowMicros());
      }));
}

PeriodicFunction::~PeriodicFunction() {
  NotifyStop();

  // Waits for thread_ to complete and clean up.
  thread_.reset();
}

void PeriodicFunction::NotifyStop() {
  if (!stop_thread_.HasBeenNotified()) {
    stop_thread_.Notify();
  }
}

void PeriodicFunction::RunLoop(const int64 start) {
  {
    if (options_.startup_delay_micros > 0) {
      const int64 deadline = start + options_.startup_delay_micros;
      options_.env->SleepForMicroseconds(deadline - start);
    }

    while (!stop_thread_.HasBeenNotified()) {
      VLOG(3) << "Running function.";
      const int64 begin = options_.env->NowMicros();
      function_();

      // Take the max() here to guard against time going backwards which
      // sometimes happens in multiproc machines.
      const int64 end =
          std::max(static_cast<int64>(options_.env->NowMicros()), begin);

      // The deadline is relative to when the last function started.
      const int64 deadline = begin + interval_micros_;

      // We want to sleep until 'deadline'.
      if (deadline > end) {
        if (end > begin) {
          VLOG(3) << "Reducing interval_micros from " << interval_micros_
                  << " to " << (deadline - end);
        }
        options_.env->SleepForMicroseconds(deadline - end);
      } else {
        VLOG(3) << "Function took longer than interval_micros, so not sleeping";
      }
    }
  }
}

}  // namespace serving
}  // namespace tensorflow
