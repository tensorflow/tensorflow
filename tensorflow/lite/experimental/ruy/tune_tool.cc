/* Copyright 2019 Google LLC. All Rights Reserved.

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

// Self-contained tool used to tune the tune code --- see the
// threshold ratios used in tune.cc.

#include <chrono>  // NOLINT(build/c++11)
#include <cstdio>
#include <thread>  // NOLINT(build/c++11)

#include "tensorflow/lite/experimental/ruy/tune.h"

#ifdef _WIN32
#define getpid() 0
#else
#include <unistd.h>
#endif

namespace ruy {

class TuneTool {
 public:
  static void Query(float* eval, float* threshold) {
    TuningResolver resolver;
    *eval = resolver.EvalRatio();
    *threshold = resolver.ThresholdRatio();
  }
};

}  // namespace ruy

int main() {
  // Infinite loop: the user can hit Ctrl-C
  while (true) {
    float eval;
    float threshold;
    ruy::TuneTool::Query(&eval, &threshold);
    printf("[%d] eval=%.3f %c threshold=%.3f  ==> probably %s...\n", getpid(),
           eval, eval < threshold ? '<' : '>', threshold,
           eval < threshold ? "in-order" : "out-of-order");
    fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}
