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

#include "xla/backends/profiler/gpu/cupti_pm_sampler_impl.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_pmsampling.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_profiler_target.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_target.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "xla/backends/profiler/gpu/cupti_pm_sampler.h"
#include "xla/backends/profiler/gpu/cupti_status.h"
#include "xla/backends/profiler/gpu/cupti_utils.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace profiler {

// Full implementation of CuptiPmSampler
// This class is responsible for managing the PM sampling process, and
// requires the CUPTI PM sampling APIs to be defined and available.
// This means this cannot build with CUDA < 12.6.

// Constructor provides all configuration needed to set up sampling on a
// single device
CuptiPmSamplerDevice::CuptiPmSamplerDevice(int device_id,
                                           const CuptiPmSamplerOptions& options)
    : device_id_(device_id), cupti_interface_(GetCuptiInterface()) {
  // Save local copy of configured metrics strings
  config_metrics_ = options.metrics;
  enabled_metrics_ = options.metrics;

  // Build list of local c string pointers (required by CUPTI calls)
  for (const auto& metric : config_metrics_) {
    c_metrics_.push_back(metric.c_str());
  }

  // Save other values provided by the options struct
  max_samples_ = options.max_samples;
  hw_buf_size_ = options.hw_buf_size;
  sample_interval_ns_ = options.sample_interval_ns;
}

// Destructor cleans up all images and objects
CuptiPmSamplerDevice::~CuptiPmSamplerDevice() {
  DestroyCounterAvailabilityImage();
  DestroyConfigImage();
  DestroyCounterDataImage();
  DestroyPmSamplerObject();
  DestroyProfilerHostObj();
}

// CUPTI params struct definitions are very long, macro it for convenience
// They all have a struct_size field which must be set to type_STRUCT_SIZE
// These strucs also have a pPriv field which must be null, ie:
// CUpti_Struct_Type var = { CUpti_Struct_Type_STRUCT_SIZE, .pPriv = nullptr }
#define DEF_SIZED_PRIV_STRUCT(type, name) \
  type name = {.structSize = type##_STRUCT_SIZE, .pPriv = nullptr}

// Fetch chip name for this device
absl::Status CuptiPmSamplerDevice::GetChipName() {
  DEF_SIZED_PRIV_STRUCT(CUpti_Device_GetChipName_Params, p);
  p.deviceIndex = device_id_;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->DeviceGetChipName(&p)));

  chip_name_ = p.pChipName;

  return absl::OkStatus();
}

// Test for device support for PM sampling
absl::Status CuptiPmSamplerDevice::DeviceSupported() {
  CUdevice cuDevice;
  TF_RETURN_IF_ERROR(
      stream_executor::cuda::ToStatus(cuDeviceGet(&cuDevice, device_id_)));

  // CUPTI call to validate configuration
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_DeviceSupported_Params, p);
  p.cuDevice = cuDevice;
  p.api = CUPTI_PROFILER_PM_SAMPLING;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->ProfilerDeviceSupported(&p)));

  if (p.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
    return absl::FailedPreconditionError("Device does not support pm sampling");
  }

  return absl::OkStatus();
}

// Get counter availability image size, set the image to that size,
// then initialize it
absl::Status CuptiPmSamplerDevice::CreateCounterAvailabilityImage() {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_GetCounterAvailability_Params, p);
  p.deviceIndex = device_id_;
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->PmSamplingGetCounterAvailability(&p)));

  counter_availability_image_.clear();
  counter_availability_image_.resize(p.counterAvailabilityImageSize);

  p.pCounterAvailabilityImage = counter_availability_image_.data();
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->PmSamplingGetCounterAvailability(&p)));

  return absl::OkStatus();
}

// Create profiler host object
absl::Status CuptiPmSamplerDevice::CreateProfilerHostObj() {
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_Initialize_Params, p);
  p.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
  p.pChipName = chip_name_.c_str();
  p.pCounterAvailabilityImage = counter_availability_image_.data();
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->ProfilerHostInitialize(&p)));

  host_obj_ = p.pHostObject;

  return absl::OkStatus();
}

void CuptiPmSamplerDevice::WarnPmSamplingMetrics() {
  static absl::once_flag once;
  absl::call_once(once, [&]() {
    LOG(WARNING) << "(Profiling::PM Sampling)  Valid metrics can by queried "
                 << "using Nsight Compute:";
    LOG(WARNING) << "(Profiling::PM Sampling)   ncu --query-metrics-collection "
                 << "pmsampling --chip " << chip_name_;
    LOG(WARNING) << "(Profiling::PM Sampling)   Note that some metrics may not "
                 << "be available on other devices, and that Triage<> named "
                 << "metrics should be available in a single pass.  Other "
                 << "combinations of metrics may not be valid if they require "
                 << "more than a single pass to collect.  Nsight Compute can "
                 << "be run remotely and list metrics for any device.";
  });
}

// Add a metrics list to the host object (destructive, requires new host obj)
CUptiResult CuptiPmSamplerDevice::AddMetricsToHostObj(
    std::vector<const char*> metrics) {
  // Add metrics to the host obj
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_ConfigAddMetrics_Params, p);
  p.pHostObject = host_obj_;
  p.ppMetricNames = metrics.data();
  p.numMetrics = metrics.size();
  return cupti_interface_->ProfilerHostConfigAddMetrics(&p);
}

// Register metrics, resize config image, and initialize it
absl::Status CuptiPmSamplerDevice::CreateConfigImage() {
  // Add metrics to the host obj, requires initializing host obj
  TF_RETURN_IF_ERROR(CreateProfilerHostObj());
  CUptiResult status = AddMetricsToHostObj(c_metrics_);

  // If the metric name is invalid, we need to log it and remove it from the
  // list of metrics.  Attempt to recover from invalid metric configuration
  if (status == CUPTI_ERROR_INVALID_PARAMETER ||
      status == CUPTI_ERROR_INVALID_METRIC_NAME) {
    // INVALID_PARAMETER may mean invalid metric, INVALID_METRIC_NAME definitely
    // does mean this
    if (status == CUPTI_ERROR_INVALID_PARAMETER) {
      LOG(WARNING) << "(Profiling::PM Sampling) Possible invalid metric name";
    } else {
      LOG(WARNING) << "(Profiling::PM Sampling) Invalid metric name";
    }

    // Log current device number
    LOG(WARNING) << "(Profiling::PM Sampling)  Device number: " << device_id_;
    // Log current device name
    LOG(WARNING) << "(Profiling::PM Sampling)  Device name: " << chip_name_;
    // Log current metric set
    LOG(WARNING) << "(Profiling::PM Sampling)  Specified metric set: ";
    for (const auto& metric : c_metrics_) {
      LOG(WARNING) << "   " << metric;
    }

    WarnPmSamplingMetrics();

    std::vector<const char*> valid_metrics;

    // Use ConfigAddMetrics to test each metric, log invalid ones, add valid
    // ones to valid vector
    for (auto& metric : c_metrics_) {
      // Reset host object and add a single metric
      TF_RETURN_IF_ERROR(CreateProfilerHostObj());
      status = AddMetricsToHostObj({metric});

      // Test validity
      if (status == CUPTI_ERROR_INVALID_PARAMETER ||
          status == CUPTI_ERROR_INVALID_METRIC_NAME) {
        LOG(WARNING) << "(Profiling::PM Sampling)   Invalid metric name: "
                     << metric;
      } else if (status == CUPTI_SUCCESS) {
        valid_metrics.push_back(metric);
      } else {
        LOG(WARNING) << "(Profiling::PM Sampling)   Unknown error for metric "
                     << metric;
        const char* errstr = "";
        cuptiGetResultString(status, &errstr);
        return absl::UnknownError(
            absl::StrCat("CUPTI error ", errstr, " for metric ", metric));
      }
    }

    // If no valid metrics, return error
    if (valid_metrics.size() == 0) {
      return absl::FailedPreconditionError("No valid metrics for PM sampling");
    }

    // Reset metrics_ to valid ones
    c_metrics_ = valid_metrics;

    // Recreate host object with valid metrics
    TF_RETURN_IF_ERROR(CreateProfilerHostObj());
    status = AddMetricsToHostObj(c_metrics_);
    if (status != CUPTI_SUCCESS) {
      LOG(WARNING) << "(Profiling::PM Sampling)   Failed to add valid metrics";
      const char* errstr = "";
      cuptiGetResultString(status, &errstr);
      return absl::UnknownError(
          absl::StrCat("CUPTI error ", errstr, " for metric set"));
    }
  } else if (status != CUPTI_SUCCESS) {
    // If we get here, we have an unknown error
    LOG(WARNING) << "(Profiling::PM Sampling)   Unknown error for metric set";
    const char* errstr = "";
    cuptiGetResultString(status, &errstr);
    return absl::UnknownError(
        absl::StrCat("CUPTI error ", errstr, " for metric set"));
  }

  // Resize and create config image
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_GetConfigImageSize_Params, ps);
  ps.pHostObject = host_obj_;
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->ProfilerHostGetConfigImageSize(&ps)));

  config_image_.clear();
  config_image_.resize(ps.configImageSize);

  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_GetConfigImage_Params, p);
  p.pHostObject = host_obj_;
  p.pConfigImage = config_image_.data();
  p.configImageSize = config_image_.size();
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->ProfilerHostGetConfigImage(&p)));

  return absl::OkStatus();
}

// Return number of passes
absl::Status CuptiPmSamplerDevice::NumPasses(size_t* passes) {
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_GetNumOfPasses_Params, p);
  p.pConfigImage = config_image_.data();
  p.configImageSize = config_image_.size();

  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->ProfilerHostGetNumOfPasses(&p)));

  *passes = p.numOfPasses;

  return absl::OkStatus();
}

// Initialize profiler APIs - required before PM sampler specific calls.
// No visible side effects.
// FIXME: Remove?
absl::Status CuptiPmSamplerDevice::InitializeProfilerAPIs() {
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Initialize_Params, p);
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->ProfilerInitialize(&p)));

  return absl::OkStatus();
}

// Create pm sampling object (initializes pm sampling APIs)
absl::Status CuptiPmSamplerDevice::CreatePmSamplerObject() {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_Enable_Params, p);
  p.deviceIndex = device_id_;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->PmSamplingEnable(&p)));

  sampling_obj_ = p.pPmSamplingObject;

  return absl::OkStatus();
}

// Resize and initialize counter data image
absl::Status CuptiPmSamplerDevice::CreateCounterDataImage() {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_GetCounterDataSize_Params, p);
  p.pPmSamplingObject = sampling_obj_;
  p.numMetrics = c_metrics_.size();
  p.pMetricNames = c_metrics_.data();
  p.maxSamples = max_samples_;
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->PmSamplingGetCounterDataSize(&p)));

  counter_data_image_.resize(p.counterDataSize);

  return InitializeCounterDataImage();
}

// Sets several pm sampling configuration items
absl::Status CuptiPmSamplerDevice::SetConfig() {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_SetConfig_Params, p);
  p.pPmSamplingObject = sampling_obj_;
  p.configSize = config_image_.size();
  p.pConfig = config_image_.data();
  p.hardwareBufferSize = hw_buf_size_;
  p.samplingInterval = sample_interval_ns_;
  p.triggerMode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->PmSamplingSetConfig(&p)));

  return absl::OkStatus();
}

// Start recording pm sampling data
absl::Status CuptiPmSamplerDevice::StartSampling() {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_Start_Params, p);
  p.pPmSamplingObject = sampling_obj_;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->PmSamplingStart(&p)));

  return absl::OkStatus();
}

// Stop recording pm sampling data
absl::Status CuptiPmSamplerDevice::StopSampling() {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_Stop_Params, p);
  p.pPmSamplingObject = sampling_obj_;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->PmSamplingStop(&p)));

  return absl::OkStatus();
}

// Disable pm sampling and destroy the pm sampling object
absl::Status CuptiPmSamplerDevice::DisableSampling() {
  // Note: currently, disabling pm sampling object finalizes all of
  // CUPTI, so do not disable here
  // TODO: Add CUPTI version test and disable once ordering is changed
  //
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_Disable_Params, p);
  p.pPmSamplingObject = sampling_obj_;
  TF_RETURN_IF_ERROR(ToStatus(cupti_interface_->PmSamplingDisable(&p)));

  sampling_obj_ = nullptr;

  return absl::OkStatus();
}

// Fetches data from hw buffer, fills in counter data image
absl::Status CuptiPmSamplerDevice::FillCounterDataImage(
    CuptiPmSamplerDecodeInfo& decode_info) {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_DecodeData_Params, p);
  p.pPmSamplingObject = sampling_obj_;
  p.pCounterDataImage = counter_data_image_.data();
  p.counterDataImageSize = counter_data_image_.size();
  CUptiResult ret = cupti_interface_->PmSamplingDecodeData(&p);

  // If this is CUPTI_ERROR_OUT_OF_MEMORY, hardware buffer is full
  // and session needs to be restarted
  if (ret == CUPTI_ERROR_OUT_OF_MEMORY) {
    LOG(WARNING) << "Profiling::PM Sampling - hardware buffer overflow, must "
                 << "restart session.  Decrease sample rate or increase decode "
                 << "rate to avoid this.";
  }

  if (ret != CUPTI_SUCCESS) {
    return absl::InternalError("CUPTI error during cuptiPmSamplingDecodeData");
  }

  // Map decode stop reason to enum class
  if (p.decodeStopReason == CUPTI_PM_SAMPLING_DECODE_STOP_REASON_OTHER) {
    decode_info.decode_stop_reason = CuptiPmSamplingDecodeStopReason::kOther;
  } else if (p.decodeStopReason ==
             CUPTI_PM_SAMPLING_DECODE_STOP_REASON_END_OF_RECORDS) {
    decode_info.decode_stop_reason =
        CuptiPmSamplingDecodeStopReason::kEndOfRecords;
  } else if (p.decodeStopReason ==
             CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL) {
    decode_info.decode_stop_reason =
        CuptiPmSamplingDecodeStopReason::kCounterDataFull;
  } else {
    LOG(WARNING) << "Profiling::PM Sampling - decode stopped for unhandled "
                    "reason: "
                 << p.decodeStopReason;
  }

  decode_info.overflow = p.overflow;

  return absl::OkStatus();
}

// Gets count of samples decoded into counter data image
absl::Status CuptiPmSamplerDevice::GetSampleCounts(
    CuptiPmSamplerDecodeInfo& decode_info) {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_GetCounterDataInfo_Params, p);
  p.pCounterDataImage = counter_data_image_.data();
  p.counterDataImageSize = counter_data_image_.size();
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->PmSamplingGetCounterDataInfo(&p)));

  decode_info.num_samples = p.numTotalSamples;
  decode_info.num_populated = p.numPopulatedSamples;
  decode_info.num_completed = p.numCompletedSamples;

  return absl::OkStatus();
}

// Fill in a single pm sampling record
absl::Status CuptiPmSamplerDevice::GetSample(SamplerRange& sample,
                                             size_t index) {
  // First, get the start and end times
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_CounterData_GetSampleInfo_Params, ps);
  ps.pPmSamplingObject = sampling_obj_;
  ps.pCounterDataImage = counter_data_image_.data();
  ps.counterDataImageSize = counter_data_image_.size();
  ps.sampleIndex = index;
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->PmSamplingCounterDataGetSampleInfo(&ps)));

  sample.range_index = index;
  sample.start_timestamp_ns = ps.startTimestamp;
  sample.end_timestamp_ns = ps.endTimestamp;
  sample.metric_values.resize(c_metrics_.size());

  // Second, get the final metric values
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_EvaluateToGpuValues_Params, p);
  p.pHostObject = host_obj_;
  p.pCounterDataImage = counter_data_image_.data();
  p.counterDataImageSize = counter_data_image_.size();
  p.ppMetricNames = c_metrics_.data();
  p.numMetrics = c_metrics_.size();
  p.rangeIndex = index;
  p.pMetricValues = sample.metric_values.data();
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->ProfilerHostEvaluateToGpuValues(&p)));

  return absl::OkStatus();
}

// Initializes image, then copies this to the backup counter data image
absl::Status CuptiPmSamplerDevice::InitializeCounterDataImage() {
  DEF_SIZED_PRIV_STRUCT(CUpti_PmSampling_CounterDataImage_Initialize_Params, p);
  p.pPmSamplingObject = sampling_obj_;
  p.counterDataSize = counter_data_image_.size();
  p.pCounterData = counter_data_image_.data();
  TF_RETURN_IF_ERROR(
      ToStatus(cupti_interface_->PmSamplingCounterDataImageInitialize(&p)));

  // Stash this in a vector so it can be restored with copy semantics
  counter_data_image_backup_ = std::vector<uint8_t>(counter_data_image_);

  return absl::OkStatus();
}

// Restores image from backup (faster than re-initializing)
absl::Status CuptiPmSamplerDevice::RestoreCounterDataImage() {
  // Will use copy semantics
  counter_data_image_ = counter_data_image_backup_;

  return absl::OkStatus();
}

void CuptiPmSamplerDevice::DestroyCounterAvailabilityImage() {
  counter_availability_image_.clear();
}

// Deinitialize and destroy the profiler host object
// Must be done after decode has stopped
void CuptiPmSamplerDevice::DestroyProfilerHostObj() {
  DEF_SIZED_PRIV_STRUCT(CUpti_Profiler_Host_Deinitialize_Params, p);
  p.pHostObject = host_obj_;
  cupti_interface_->ProfilerHostDeinitialize(&p);

  host_obj_ = nullptr;
}

void CuptiPmSamplerDevice::DestroyConfigImage() { config_image_.clear(); }

// Disable sampling and destroy the pm sampling object
// Must be done after decode has stopped
void CuptiPmSamplerDevice::DestroyPmSamplerObject() {
  DisableSampling().IgnoreError();
}

void CuptiPmSamplerDevice::DestroyCounterDataImage() {
  counter_data_image_.clear();
  counter_data_image_backup_.clear();
}

absl::Status CuptiPmSamplerDevice::CreateConfig() {
  // Get chip name string
  TF_RETURN_IF_ERROR(GetChipName());

  // Test whether the hardware supports pm sampling
  TF_RETURN_IF_ERROR(DeviceSupported());

  // Create counter availability image
  TF_RETURN_IF_ERROR(CreateCounterAvailabilityImage());

  // Attempt to handle invalid metrics and multiple passes
  size_t passes;
  do {
    // Create a host object and add current set of metrics; create config image
    TF_RETURN_IF_ERROR(CreateConfigImage());

    // Test config image for number of passes
    TF_RETURN_IF_ERROR(NumPasses(&passes));

    if (passes > 1) {
      // If on last metric, return error
      if (c_metrics_.size() == 1) {
        LOG(WARNING) << "(Profiling::PM Sampling) Device " << device_id_ << " "
                     << "requires more than one pass even for the first "
                     << "metric, " << c_metrics_.back() << ", and cannot be "
                     << "configured";
        return absl::InvalidArgumentError(
            "Primary metric requires more than one pass");
      }
      // Remove last metric from list and try again
      LOG(WARNING) << "(Profiling::PM Sampling) Device " << device_id_
                   << " metrics configuration requires more than one pass, "
                   << "removing last metric " << c_metrics_.back() << " and "
                   << "trying to configure again";

      WarnPmSamplingMetrics();

      c_metrics_.pop_back();
    } else if (passes == 0) {
      LOG(WARNING) << "(Profiling::PM Sampling) Device " << device_id_
                   << " metrics configuration is invalid, cannot configure";
      return absl::InvalidArgumentError(
          "Invalid metric configuration for PM sampling");
    }
  } while (passes != 1);

  // Update enabled_metrics_ with the current metrics
  enabled_metrics_.clear();
  for (const auto& metric : c_metrics_) {
    enabled_metrics_.emplace_back(metric);
  }

  // Create PM sampler object
  TF_RETURN_IF_ERROR(CreatePmSamplerObject());

  // Create counter data image
  TF_RETURN_IF_ERROR(CreateCounterDataImage());

  return absl::OkStatus();
}

// Constructor, creates worker thread
CuptiPmSamplerDecodeThread::CuptiPmSamplerDecodeThread(
    std::vector<std::shared_ptr<CuptiPmSamplerDevice>> devs,
    const CuptiPmSamplerOptions& options) {
  devs_ = devs;

  decode_period_ = options.decode_period;

  if (!options.process_samples) {
    process_samples_ = [](PmSamples* info) {
      LOG(WARNING) << "(Profiling::PM Sampling) No decode handler specified, "
                   << "discarding " << info->GetSamplerRanges().size()
                   << " samples";
      return;
    };
  } else {
    process_samples_ = options.process_samples;
  }

  thread_ = std::make_unique<std::thread>(&CuptiPmSamplerDecodeThread::MainFunc,
                                          this);
}

void CuptiPmSamplerDecodeThread::DecodeUntilDisabled() {
  // Test for exit condition on all devices
  bool all_devs_end_of_records = false;
  bool final_pass = false;
  bool disabling = false;

  // When enabled, loop over each device, decoding
  // Run until disabled, and then do one extra pass
  while (!disabling) {
    VLOG(2) << "(Profiling::PM Sampling) Top of decode loop";

    // If next state is not enabled, do one more pass
    if (!NextStateIs(ThreadState::kEnabled)) {
      if (!final_pass) {
        final_pass = true;
      } else {
        disabling = true;
      }
    }

    all_devs_end_of_records = true;
    absl::Time begin = absl::Now();

    size_t decoded_samples = 0;

    // Each decode period, decode all devices assigned to it
    for (const auto& dev : devs_) {
      VLOG(2) << "(Profiling::PM Sampling)  Beginning decode for device "
              << dev->device_id_;

      absl::Time start_time = absl::Now();
      absl::Time fill_time = start_time;
      absl::Time get_count_time = start_time;
      absl::Time get_samples_time = start_time;
      absl::Time process_samples_time = start_time;
      absl::Time initialize_image_time = start_time;

      CuptiPmSamplerDecodeInfo info;

      if (!dev->FillCounterDataImage(info).ok()) {
        continue;
      }
      fill_time = absl::Now();

      if (!dev->GetSampleCounts(info).ok()) {
        continue;
      }
      get_count_time = absl::Now();

      // Track whether this device reached end of records
      if (info.decode_stop_reason !=
          CuptiPmSamplingDecodeStopReason::kEndOfRecords) {
        all_devs_end_of_records = false;
      } else {
        VLOG(2) << "(Profiling::PM Sampling)   End of records for device "
                << dev->device_id_;
      }

      if (info.overflow) {
        LOG(WARNING) << "(Profiling::PM Sampling) hardware buffer overflow on "
                     << "device " << dev->device_id_
                     << ", sample data has been lost";
      }

      if (info.decode_stop_reason ==
          CuptiPmSamplingDecodeStopReason::kCounterDataFull) {
        LOG(WARNING) << "(Profiling::PM Sampling) ran out of host buffer space "
                     << "before decoding all records from the device buffer on "
                     << "device " << dev->device_id_;
      }

      if (info.num_completed == 0) {
        VLOG(3) << "(Profiling::PM Sampling)   FillCounterDataImage took "
                << (fill_time - start_time);
        VLOG(3) << "(Profiling::PM Sampling)   GetSampleCounts took "
                << (get_count_time - fill_time);
        continue;
      }

      decoded_samples += info.num_completed;

      info.sampler_ranges.resize(info.num_completed);

      // Set each sample's info, reset samples that error
      // (should not happen)
      for (size_t i = 0; i < info.num_completed; i++) {
        if (!dev->GetSample(info.sampler_ranges[i], i).ok()) {
          LOG(WARNING) << "(Profiling::PM Sampling) Error decoding pm sample";
          info.sampler_ranges[i].range_index = 0;
          info.sampler_ranges[i].start_timestamp_ns = 0;
          info.sampler_ranges[i].end_timestamp_ns = 0;
          info.sampler_ranges[i].metric_values.clear();
        } else {
          if (VLOG_IS_ON(4)) {
            for (int j = 0; j < dev->GetEnabledMetrics().size(); j++) {
              LOG(INFO) << "            " << dev->GetEnabledMetrics()[j] << "["
                        << i
                        << "] = " << info.sampler_ranges[i].metric_values[j];
            }
          }
        }
      }

      get_samples_time = absl::Now();

      // info now contains a list of samples and metrics,
      // hand off to process or store elsewhere
      if (process_samples_) {
        PmSamples samples(dev->GetEnabledMetrics(), info.sampler_ranges,
                          dev->device_id_);
        process_samples_(&samples);
      }

      process_samples_time = absl::Now();

      if (!dev->RestoreCounterDataImage().ok()) {
        LOG(WARNING) << "(Profiling::PM Sampling) Error resetting counter data "
                     << "image";
      }

      initialize_image_time = absl::Now();

      VLOG(3) << "(Profiling::PM Sampling)   FillCounterDataImage took "
              << (fill_time - start_time);
      VLOG(3) << "(Profiling::PM Sampling)   GetSampleCounts took "
              << (get_count_time - fill_time);
      VLOG(3) << "(Profiling::PM Sampling)   vector resize & getSample for "
              << info.num_completed << " samples took "
              << (get_samples_time - get_count_time);
      VLOG(3) << "(Profiling::PM Sampling)   external processing of samples "
              << "took " << (process_samples_time - get_samples_time);
      VLOG(3) << "(Profiling::PM Sampling)   RestoreCounterDataImage took "
              << (initialize_image_time - process_samples_time);
    }

    // Sleep until start of next period,
    // warning if decode took longer than allocated time
    absl::Time end = absl::Now();
    absl::Duration elapsed = end - begin;
    if (elapsed < decode_period_) {
      VLOG(2) << "(Profiling::PM Sampling)   decoded " << decoded_samples
              << ", took " << elapsed << ", sleeping for "
              << (decode_period_ - elapsed);
      absl::SleepFor(decode_period_ - elapsed);
    } else {
      VLOG(2) << "(Profiling::PM Sampling)   decoded " << decoded_samples
              << ", took " << elapsed << ", decode period is "
              << decode_period_;
      LOG(WARNING) << "(Profiling::PM Sampling) decode thread took longer than "
                   << "configured period to complete a single decode pass.  "
                   << "When this happens, hardware buffer may overflow and "
                   << "lose sample data.  Reduce number of devices per decode "
                   << "thread, reduce the number of metrics gathered, reduce "
                   << "the sample rate, or ensure decode threads have "
                   << "sufficient cpu resources to maintain decode faster than "
                   << "metric sampling.  Elapsed time: " << elapsed << ", "
                   << "decode period: " << decode_period_;
    }
  }

  VLOG(2) << "(Profiling::PM Sampling) Exited decode loop";

  if (!all_devs_end_of_records) {
    // If not all devices reached end of records, warn the user
    LOG(WARNING) << "(Profiling::PM Sampling) Not all devices reached end of "
                 << "records, some data may be lost.  This is expected if "
                 << "sampling rate is too high or decode thread is not "
                 << "sufficiently prioritized, or if decode thread is disabled "
                 << "while sampling is enabled.";
  }
}

// Entry function for decode thread
void CuptiPmSamplerDecodeThread::MainFunc() {
  // RAII lock to ensure mutex is released when thread exits
  absl::MutexLock lock(state_mutex_);

  // Control loop for decode thread
  do {
    // Wait for signal to change state.  Releases lock during wait
    auto stateChanged = [this] {
      state_mutex_.AssertReaderHeld();
      return current_state_ != next_state_;
    };
    state_mutex_.Await(absl::Condition(&stateChanged));

    switch (next_state_) {
      case ThreadState::kInitialized:
        // Space for thread initialization if needed
        // ...
        // Initialization done, transition to disabled state
        StateIs(ThreadState::kDisabled);
        break;
      case ThreadState::kEnabled:
        StateIs(ThreadState::kEnabled);
        {
          // Release lock for expensive decode call, but regain before returning
          // to control loop
          state_mutex_.unlock();
          DecodeUntilDisabled();
          state_mutex_.lock();
        }
        // Returns when Disabled has been requested
        StateIs(ThreadState::kDisabled);
        break;
      case ThreadState::kUninitialized:
        // Initially both current and next state should be uninitialized so we
        // should never get here
        LOG(WARNING) << "(Profiling::PM Sampling) Decode thread transitioned "
                     << "to uninitialized state";
        StateIs(ThreadState::kUninitialized);
        break;
      case ThreadState::kExiting:
        // Space for thread teardown if needed
        // ...
        // Thread is exiting, so return to allow joining
        StateIs(ThreadState::kExiting);
        return;
    }
  } while (true);
}

absl::Status CuptiPmSamplerImpl::Initialize(
    size_t num_gpus, const CuptiPmSamplerOptions& options) {
  // Ensure not already initialized
  if (initialized_) {
    return absl::AlreadyExistsError("PM sampler already initialized");
  }

  // Use absl cleanup to clear allocated memory on error
  // (Cancel before successful return)
  absl::Cleanup cleanup([this]() {
    threads_.clear();
    devices_.clear();
  });

  // PM sampling has to be enabled on individual devices
  for (int dev_idx = 0; dev_idx < num_gpus; dev_idx++) {
    // Create a new PM sampling instance for this device
    // This makes a copy of the relevant options passed in including a copy of
    // the metrics vector, so can free after this point
    std::shared_ptr<CuptiPmSamplerDevice> dev =
        std::make_shared<CuptiPmSamplerDevice>(dev_idx, options);

    // Create all configuration needed for this device, or error out
    TF_RETURN_IF_ERROR(dev->CreateConfig());

    // Set configuration or error out
    TF_RETURN_IF_ERROR(dev->SetConfig());

    // Device is fully configured but PM sampling not yet started - push to list
    // of PM sampling devices
    devices_.push_back(std::move(dev));
  }

  // Create decode thread(s)
  for (int i = 0; i < devices_.size(); i += options.devs_per_decode_thd) {
    // Slice iterators
    auto begin = devices_.begin() + i;
    auto end = begin + options.devs_per_decode_thd;
    // Don't go past end of vector
    end = std::min(end, devices_.end());

    // Slice for this decode thread
    std::vector<std::shared_ptr<CuptiPmSamplerDevice>> slice(begin, end);

    // Create worker thread for this slice
    auto thd =
        std::make_unique<CuptiPmSamplerDecodeThread>(std::move(slice), options);
    threads_.push_back(std::move(thd));
  }

  // Request and wait for signal that all threads are ready
  for (auto& thd : threads_) {
    thd->Initialize();
  }
  for (auto& thd : threads_) {
    thd->AwaitInitialization();
  }

  // Cancel the cleanup, as we are now initialized
  std::move(cleanup).Cancel();

  initialized_ = true;

  return absl::OkStatus();
}

absl::Status CuptiPmSamplerImpl::StartSampler() {
  if (enabled_) {
    return absl::AlreadyExistsError("Already started");
  }

  // Start sampling on all devices
  for (auto& dev : devices_) {
    if (!dev->StartSampling().ok()) {
      LOG(WARNING) << "Profiling::PM Sampling - failed to start on device "
                   << dev->device_id_;
      // TODO: What is appropriate behavior if start thread fails?
      // Most likely should delete the sampler for this device but this would
      // need to be communicated to the decoder thread.  Should be safe to do
      // nothing, as there will just be no data to decode
    }
  }

  // Signal threads should be enabled
  for (auto& thd : threads_) {
    thd->Enable();
  }

  // Wait for signal that decode thread is enabled
  for (auto& thd : threads_) {
    thd->AwaitEnablement();
  }

  enabled_ = true;

  return absl::OkStatus();
}

absl::Status CuptiPmSamplerImpl::StopSampler() {
  if (!enabled_) {
    return absl::FailedPreconditionError(
        "StopSampler called before StartSampler, or failure during "
        "StartSampler");
  }

  // Stop sampling on all devices
  for (auto& dev : devices_) {
    if (!dev->StopSampling().ok()) {
      LOG(WARNING) << "Profiling::PM Sampling - failed to stop on device "
                   << dev->device_id_;
    }
  }

  // Signal threads should be disabled
  for (auto& thd : threads_) {
    thd->Disable();
  }

  // Wait for signal that decode thread is disabled
  for (const auto& thd : threads_) {
    thd->AwaitDisablement();
  }

  enabled_ = false;

  return absl::OkStatus();
}

absl::Status CuptiPmSamplerImpl::Deinitialize() {
  if (enabled_) {
    StopSampler().IgnoreError();
  }
  if (!initialized_) {
    return absl::FailedPreconditionError(
        "Deinitialize called before Initialize, or failure during Initialize");
  }

  // Tell threads to exit
  for (auto& thd : threads_) {
    thd->Exit();
  }

  // Threads will soon exit, ready to join
  for (auto& thd : threads_) {
    thd->AwaitExit();

    // Destroy decode thread (joins thread)
    thd.reset();
  }

  // Destroy all decode threads
  threads_.clear();

  // Destroy all devices
  devices_.clear();

  initialized_ = false;

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<CuptiPmSamplerImpl>> CuptiPmSamplerImpl::Create(
    size_t num_gpus, const CuptiPmSamplerOptions& options) {
  std::unique_ptr<CuptiPmSamplerImpl> sampler(new CuptiPmSamplerImpl());

  if (num_gpus < 1) {
    // Return an uninitialized sampler if no gpus are present
    return sampler;
  }

  TF_RETURN_IF_ERROR(sampler->Initialize(num_gpus, options));
  return sampler;
}

}  // namespace profiler
}  // namespace xla
