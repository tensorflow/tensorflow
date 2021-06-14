/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/system_setup.h"

#ifndef TF_LITE_STRIP_ERROR_STRINGS
#include "q6sim_timer.h"  // NOLINT
#endif                    // TF_LITE_STRIP_ERROR_STRINGS

#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/micro_time.h"

namespace tflite {

// Calling this method enables a timer that runs for eternity.
void InitializeTarget() {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  hexagon_sim_init_timer();
  hexagon_sim_start_timer();
#endif  // TF_LITE_STRIP_ERROR_STRINGS
}

int32_t ticks_per_second() { return 1000000; }

int32_t GetCurrentTimeTicks() {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  return hexagon_sim_read_cycles();
#else
  return 0;
#endif  // TF_LITE_STRIP_ERROR_STRINGS
}

}  // namespace tflite
