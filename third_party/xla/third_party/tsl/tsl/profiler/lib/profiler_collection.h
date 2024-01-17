/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_TSL_PROFILER_LIB_PROFILER_COLLECTION_H_
#define TENSORFLOW_TSL_PROFILER_LIB_PROFILER_COLLECTION_H_

#include <memory>
#include <vector>

#include "tsl/platform/status.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

// ProfilerCollection multiplexes ProfilerInterface calls into a collection of
// profilers.
class ProfilerCollection : public ProfilerInterface {
 public:
  explicit ProfilerCollection(
      std::vector<std::unique_ptr<ProfilerInterface>> profilers);

  Status Start() override;

  Status Stop() override;

  Status CollectData(tensorflow::profiler::XSpace* space) override;

 private:
  std::vector<std::unique_ptr<ProfilerInterface>> profilers_;
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_PROFILER_COLLECTION_H_
