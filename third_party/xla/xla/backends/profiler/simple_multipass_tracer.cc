/* Copyright 2026 The OpenXLA Authors.

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

// Simple multipass tracer wrapping all registered single-pass profilers.
#include "xla/backends/profiler/simple_multipass_tracer.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/profiler/lib/profiler_collection.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using tensorflow::ProfileOptions;
using tensorflow::profiler::XSpace;
using tsl::profiler::MultiPassProfilerInterface;
using tsl::profiler::ProfilerCollection;

class SimpleMultiPassTracer : public MultiPassProfilerInterface {
 public:
  explicit SimpleMultiPassTracer(const ProfileOptions& options)
      : options_(options) {
    VLOG(3) << "SimpleMultiPassTracer created for device "
            << options_.device_type();
  }
  ~SimpleMultiPassTracer() override {
    VLOG(3) << "SimpleMultiPassTracer destroyed.";
  }

  // MultiPassProfilerInterface methods
  bool NeedMorePasses() override {
    VLOG(3) << "SimpleMultiPassTracer::NeedMorePasses. pass_count="
            << pass_count_;
    return pass_count_ < 2;
  }

  absl::Status StartPass() override {
    VLOG(3) << "SimpleMultiPassTracer::StartPass. pass_count=" << pass_count_;
    if (IsRealPass()) {
      if (profilers_ != nullptr) {
        VLOG(3) << "Starting real profilers for Pass " << pass_count_;
        return profilers_->Start();
      }
      LOG(WARNING) << "Real profilers are null, cannot start them.";
    }
    return absl::OkStatus();
  }

  absl::Status PushRange(absl::string_view name) override {
    VLOG(3) << "SimpleMultiPassTracer::PushRange: " << name;
    return absl::OkStatus();
  }

  absl::Status PopRange() override {
    VLOG(3) << "SimpleMultiPassTracer::PopRange";
    return absl::OkStatus();
  }

  absl::Status StopPass() override {
    VLOG(3) << "SimpleMultiPassTracer::StopPass. pass_count=" << pass_count_;
    absl::Status status = absl::OkStatus();
    if (IsRealPass()) {
      if (profilers_ != nullptr) {
        VLOG(3) << "Stopping real profilers for Pass " << pass_count_;
        status = profilers_->Stop();
      }
    }
    pass_count_++;
    return status;
  }

  // ProfilerInterface methods
  absl::Status Start() override {
    VLOG(3) << "SimpleMultiPassTracer::Start";
    pass_count_ = 0;

    ProfileOptions options = options_;
    options.set_enable_multipass(false);

    profilers_ = std::make_unique<ProfilerCollection>(
        tsl::profiler::CreateProfilers(options));

    return absl::OkStatus();
  }

  absl::Status Stop() override {
    VLOG(3) << "SimpleMultiPassTracer::Stop";
    return absl::OkStatus();
  }

  absl::Status CollectData(XSpace* space) override {
    VLOG(3) << "SimpleMultiPassTracer::CollectData";
    if (profilers_ != nullptr) {
      return profilers_->CollectData(space);
    }
    return absl::OkStatus();
  }

 private:
  bool IsRealPass() const {
    if (options_.device_type() == ProfileOptions::GPU) {
      // GPU: Pass 0 is dummy (warmup), Pass 1 is real.
      return pass_count_ == 1;
    }
    // TPU and others: Pass 0 is real, Pass 1 is dummy.
    return pass_count_ == 0;
  }

  ProfileOptions options_;
  int pass_count_ = 0;
  std::unique_ptr<ProfilerCollection> profilers_;
};

std::unique_ptr<MultiPassProfilerInterface> CreateSimpleMultiPassTracer(
    const ProfileOptions& options) {
  VLOG(3) << "CreateSimpleMultiPassTracer called. device_type="
          << options.device_type()
          << ", enable_multipass=" << options.enable_multipass();
  if (!options.enable_multipass()) {
    VLOG(3) << "enable_multipass is false, not creating SimpleMultiPassTracer";
    return nullptr;
  }
  // We support GPU, TPU, and UNSPECIFIED.
  if (options.device_type() == ProfileOptions::GPU ||
      options.device_type() == ProfileOptions::TPU ||
      options.device_type() == ProfileOptions::UNSPECIFIED) {
    return std::make_unique<SimpleMultiPassTracer>(options);
  }
  VLOG(3) << "Device is not supported, not creating SimpleMultiPassTracer";
  return nullptr;
}

}  // namespace

void RegisterSimpleMultiPassTracer() {
  VLOG(3) << "Registering SimpleMultiPassTracer factory.";
  tsl::profiler::RegisterMultiPassProfilerFactory(&CreateSimpleMultiPassTracer);
}

}  // namespace profiler
}  // namespace xla
