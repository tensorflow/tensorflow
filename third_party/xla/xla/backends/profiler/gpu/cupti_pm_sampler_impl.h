/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_IMPL_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_IMPL_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"

// This class requires the CUPTI PM sampling APIs to be defined and available.
// This means this cannot build with CUDA < 12.6.  Build is controled through
// bazel.
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_pmsampling.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_profiler_host.h"
#include "xla/backends/profiler/gpu/cupti_pm_sampler.h"

namespace xla {
namespace profiler {

// Information related to a decode counters pass over a single device

// Maps to CUpti_PmSampling_DecodeStopReason without requiring CUPTI headers
// Only include the enum values that are used in the code
enum class CuptiPmSamplingDecodeStopReason {
  kOther,            // CUPTI_PM_SAMPLING_DECODE_STOP_REASON_OTHER
  kCounterDataFull,  // CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL
  kEndOfRecords,     // CUPTI_PM_SAMPLING_DECODE_STOP_REASON_END_OF_RECORDS
  kCount             // CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNT
};

struct CuptiPmSamplerDecodeInfo {
  CuptiPmSamplingDecodeStopReason decode_stop_reason =
      CuptiPmSamplingDecodeStopReason::kCount;
  uint8_t overflow = 0;
  size_t num_samples = 0;
  size_t num_completed = 0;
  size_t num_populated = 0;
  int device_id;
  std::vector<SamplerRange> sampler_ranges;
};

// Container class for all CUPTI pm sampling infrastructure
// - Configuration
// - Enablement / disablement
// - Buffer creation
// - Worker thread creation / control
// Decoding of counter data is done in CuptiPmSamplerDecodeThread class
class CuptiPmSamplerDevice {
 public:
  // Device information
  int device_id_;

  // Creates host and sampler objects, all images
  absl::Status CreateConfig();

  // Requires config image, pm sampler object
  absl::Status SetConfig();

  // Requires pm sampler object
  absl::Status StartSampling();

  // Requires pm sampler object
  absl::Status StopSampling();

  // Requires pm sampler object, destroys pm sampler object
  absl::Status DisableSampling();

  // Collect sampling data
  // Requires pm sampler object, counter data image, fetches data from hw
  // buffer into counter data image
  absl::Status FillCounterDataImage(CuptiPmSamplerDecodeInfo& decode_info);

  // Requires counter data image
  absl::Status GetSampleCounts(CuptiPmSamplerDecodeInfo& decode_info);

  // Requires host object, pm sampler object, counter data image, metric
  // names, returns sample time, metric values
  absl::Status GetSample(SamplerRange& sample, size_t index);

  // Requires pm sampler object, counter data image, (re)initializes it
  absl::Status InitializeCounterDataImage();

  // Restores image from backup (faster than re-initializing)
  absl::Status RestoreCounterDataImage();

  // Simple warning, needed in multiple spots
  void WarnPmSamplingMetrics();

  // Constructor provides all configuration needed to set up sampling on a
  // single device
  CuptiPmSamplerDevice(int device_id, const CuptiPmSamplerOptions& options);

  // Destructor cleans up all images and objects
  ~CuptiPmSamplerDevice();

  // Return reference to enabled metrics
  const std::vector<std::string>& GetEnabledMetrics() const {
    return enabled_metrics_;
  }

 private:
  // Internal state
  size_t max_samples_;
  size_t hw_buf_size_;
  size_t sample_interval_ns_;

  // CUPTI PM sampling objects
  // Declared roughly in order of initialization
  std::string chip_name_;
  std::vector<std::string> config_metrics_;   // Local copy of metrics strings
  std::vector<std::string> enabled_metrics_;  // Metrics enabled in host object
  std::vector<const char*> c_metrics_;        // CUPTI needs C string pointers
  std::vector<uint8_t> counter_availability_image_;
  CUpti_Profiler_Host_Object* host_obj_ = nullptr;
  std::vector<uint8_t> config_image_;
  CUpti_PmSampling_Object* sampling_obj_ = nullptr;
  std::vector<uint8_t> counter_data_image_;
  std::vector<uint8_t> counter_data_image_backup_;

  // XLA interface to CUPTI
  // Needed both to call PM sampling APIs and stringify CUPTI errors
  CuptiInterface* cupti_interface_;

  // Configuration calls
  absl::Status GetChipName();
  absl::Status DeviceSupported();
  absl::Status CreateCounterAvailabilityImage();

  // Requires counter availability image
  absl::Status CreateProfilerHostObj();

  // Requires profiler host object
  absl::Status CreateConfigImage();

  // Set metrics for host object
  CUptiResult AddMetricsToHostObj(std::vector<const char*> metrics);

  // Requires config image
  absl::Status NumPasses(size_t* passes);

  absl::Status InitializeProfilerAPIs();
  absl::Status CreatePmSamplerObject();

  // Requires pm sampler object
  absl::Status CreateCounterDataImage();

  // Clean up
  void DestroyCounterAvailabilityImage();
  void DestroyConfigImage();
  void DestroyCounterDataImage();
  void DestroyProfilerHostObj();
  void DestroyPmSamplerObject();
};

// Container for PM sampling decode thread
// Responsible for fetching PM sampling data from device and providing it to
// handler or other external container
class CuptiPmSamplerDecodeThread {
 public:
  CuptiPmSamplerDecodeThread(
      std::vector<std::shared_ptr<CuptiPmSamplerDevice>> devs,
      const CuptiPmSamplerOptions& options);

  // Signal thread to exit; join thread
  ~CuptiPmSamplerDecodeThread() {
    next_state_ = ThreadState::kExiting;
    thread_->join();
  }

  // Transitions to disabled
  void Initialize() { ChangeState(ThreadState::kInitialized); }
  void AwaitInitialization() { AwaitState(ThreadState::kInitialized); }

  // Straightforward state transitions
  void Enable() { ChangeState(ThreadState::kEnabled); }
  void AwaitEnablement() { AwaitState(ThreadState::kEnabled); }
  void Disable() { ChangeState(ThreadState::kDisabled); }
  void AwaitDisablement() { AwaitState(ThreadState::kDisabled); }
  void Exit() { ChangeState(ThreadState::kExiting); }
  void AwaitExit() { AwaitState(ThreadState::kExiting); }

 private:
  // Spin wait sleep period, set to the min of this and all device periods
  // Space to asynchronously initialize this class and the thread it spawns
  absl::Duration decode_period_ = absl::Seconds(1);

  std::function<void(PmSamples* samples)> process_samples_;

  // Guard state change with mutexes
  absl::Mutex state_mutex_;

  // Efficient notifier for state changes
  absl::CondVar state_change_notifier_;

  // Thread state.  Initialization goes straight to disabled, hence they are
  // equivalent.
  enum class ThreadState {
    // Thread is starting, not yet ready to be enabled
    kUninitialized,
    // Thread is ready for enablement but decoding has not yet been triggered
    kInitialized,
    // Thread is disabled but could be re-enabled
    kDisabled = kInitialized,
    // Thread is enabled, polling for metrics from all devices
    kEnabled,
    // Thread is finishing and guaranteed to return, allowing join
    kExiting
  };

  // Current state of the thread
  ThreadState current_state_ ABSL_GUARDED_BY(state_mutex_) =
      ThreadState::kUninitialized;

  // State thread should transition to
  ThreadState next_state_ ABSL_GUARDED_BY(state_mutex_) =
      ThreadState::kInitialized;

  // Tell thread to change state
  void ChangeState(ThreadState state) ABSL_LOCKS_EXCLUDED(state_mutex_) {
    absl::WriterMutexLock lock(state_mutex_);
    next_state_ = state;
    state_change_notifier_.SignalAll();
  }

  // Internal state change
  void StateIs(ThreadState state) ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mutex_) {
    current_state_ = state;
    state_change_notifier_.SignalAll();
  }

  // Compare state
  bool CurrentStateIs(ThreadState state) ABSL_LOCKS_EXCLUDED(state_mutex_) {
    absl::ReaderMutexLock lock(state_mutex_);
    return current_state_ == state;
  }

  bool NextStateIs(ThreadState state) ABSL_LOCKS_EXCLUDED(state_mutex_) {
    absl::ReaderMutexLock lock(state_mutex_);
    return next_state_ == state;
  }

  void AwaitState(ThreadState state) ABSL_LOCKS_EXCLUDED(state_mutex_) {
    absl::ReaderMutexLock lock(state_mutex_);
    auto equals = [this, state] {
      state_mutex_.AssertReaderHeld();
      return current_state_ == state;
    };
    state_mutex_.Await(absl::Condition(&equals));
  }

  // Thread handle
  std::unique_ptr<std::thread> thread_;

  // Function run by thread_
  void MainFunc();

  // Isolate the main decode loop
  void DecodeUntilDisabled();

  // Devices to decode by this thread
  std::vector<std::shared_ptr<CuptiPmSamplerDevice>> devs_;
};

// Full implementation of CuptiPmSampler
class CuptiPmSamplerImpl : public CuptiPmSampler {
 public:
  static absl::StatusOr<std::unique_ptr<CuptiPmSamplerImpl>> Create(
      size_t num_gpus, const CuptiPmSamplerOptions& options);

  // Start sampling and decoding
  absl::Status StartSampler() override;

  // Stop sampling and decoding
  absl::Status StopSampler() override;

  // Deinitialize the PM sampler
  absl::Status Deinitialize() override;

 private:
  CuptiPmSamplerImpl() = default;

  // Initialize the PM sampler, but do not start sampling or decoding
  absl::Status Initialize(size_t num_gpus,
                          const CuptiPmSamplerOptions& options);

  // Interface is (at least, partially) initialized
  bool initialized_ = false;
  // Interface is (at least, partially) enabled
  bool enabled_ = false;
  // All PM sampler per-device objects
  std::vector<std::shared_ptr<CuptiPmSamplerDevice>> devices_;
  // All PM sampler per-decode-thread objects
  std::vector<std::unique_ptr<CuptiPmSamplerDecodeThread>> threads_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_IMPL_H_
