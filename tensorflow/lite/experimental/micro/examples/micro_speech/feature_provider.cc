/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.h"

#include "tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/timer.h"

namespace {
// Stores the timestamp for the previous fetch of audio data, so that we can
// avoid recalculating all the features from scratch if some earlier timeslices
// are still present.
int32_t g_last_time_in_ms = 0;
// Make sure we don't try to use cached information if this is the first call
// into the provider.
bool g_is_first_run = true;
}  // namespace

FeatureProvider::FeatureProvider(int feature_size, uint8_t* feature_data)
    : feature_size_(feature_size), feature_data_(feature_data) {
  // Initialize the feature data to default values.
  for (int n = 0; n < feature_size_; ++n) {
    feature_data_[n] = 0;
  }
}

FeatureProvider::~FeatureProvider() {}

TfLiteStatus FeatureProvider::PopulateFeatureData(
    tflite::ErrorReporter* error_reporter, int* how_many_new_slices) {
  if (feature_size_ != kFeatureElementCount) {
    error_reporter->Report("Requested feature_data_ size %d doesn't match %d",
                           feature_size_, kFeatureElementCount);
    return kTfLiteError;
  }

  const int32_t time_in_ms = TimeInMilliseconds();
  // Quantize the time into steps as long as each window stride, so we can
  // figure out which audio data we need to fetch.
  const int last_step = (g_last_time_in_ms / kFeatureSliceStrideMs);
  const int current_step = (time_in_ms / kFeatureSliceStrideMs);
  g_last_time_in_ms = time_in_ms;

  int slices_needed = current_step - last_step;
  // If this is the first call, make sure we don't use any cached information.
  if (g_is_first_run) {
    g_is_first_run = false;
    slices_needed = kFeatureSliceCount;
  }
  if (slices_needed > kFeatureSliceCount) {
    slices_needed = kFeatureSliceCount;
  }
  *how_many_new_slices = slices_needed;

  const int slices_to_keep = kFeatureSliceCount - slices_needed;
  const int slices_to_drop = kFeatureSliceCount - slices_to_keep;
  // If we can avoid recalculating some slices, just move the existing data
  // up in the spectrogram, to perform something like this:
  // last time = 80ms          current time = 120ms
  // +-----------+             +-----------+
  // | data@20ms |         --> | data@60ms |
  // +-----------+       --    +-----------+
  // | data@40ms |     --  --> | data@80ms |
  // +-----------+   --  --    +-----------+
  // | data@60ms | --  --      |  <empty>  |
  // +-----------+   --        +-----------+
  // | data@80ms | --          |  <empty>  |
  // +-----------+             +-----------+
  if (slices_to_keep > 0) {
    for (int dest_slice = 0; dest_slice < slices_to_keep; ++dest_slice) {
      uint8_t* dest_slice_data =
          feature_data_ + (dest_slice * kFeatureSliceSize);
      const int src_slice = dest_slice + slices_to_drop;
      const uint8_t* src_slice_data =
          feature_data_ + (src_slice * kFeatureSliceSize);
      for (int i = 0; i < kFeatureSliceSize; ++i) {
        dest_slice_data[i] = src_slice_data[i];
      }
    }
  }
  // Any slices that need to be filled in with feature data have their
  // appropriate audio data pulled, and features calculated for that slice.
  if (slices_needed > 0) {
    for (int new_slice = slices_to_keep; new_slice < kFeatureSliceCount;
         ++new_slice) {
      const int new_step = (current_step - kFeatureSliceCount + 1) + new_slice;
      const int32_t slice_start_ms = (new_step * kFeatureSliceStrideMs);
      int16_t* audio_samples = nullptr;
      int audio_samples_size = 0;
      GetAudioSamples(error_reporter, slice_start_ms, kFeatureSliceDurationMs,
                      &audio_samples_size, &audio_samples);
      if (audio_samples_size < kMaxAudioSampleSize) {
        error_reporter->Report("Audio data size %d  too small, want %d",
                               audio_samples_size, kMaxAudioSampleSize);
        return kTfLiteError;
      }
      uint8_t* new_slice_data = feature_data_ + (new_slice * kFeatureSliceSize);
      TfLiteStatus preprocess_status =
          Preprocess(error_reporter, audio_samples, audio_samples_size,
                     kFeatureSliceSize, new_slice_data);
      if (preprocess_status != kTfLiteOk) {
        return preprocess_status;
      }
    }
  }
  return kTfLiteOk;
}
