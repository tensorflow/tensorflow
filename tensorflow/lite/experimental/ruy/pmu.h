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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PMU_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PMU_H_

#include <cstdint>

namespace ruy {

class PmuEventsPrivate;

class PmuEvents {
 public:
  PmuEvents();
  ~PmuEvents();
  void StartRecording();
  void StopRecording();
  float L1AccessCount() const;
  float L1RefillCount() const;
  float L2RefillCount() const;
  float L3RefillCount() const;
  float BranchMispredictionRate() const;
  float FrontendStallRate() const;
  float BackendStallRate() const;

 private:
  PmuEventsPrivate* priv = nullptr;
};

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PMU_H_
