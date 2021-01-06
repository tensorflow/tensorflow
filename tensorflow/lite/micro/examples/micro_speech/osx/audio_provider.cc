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

#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"

#include <AudioToolbox/AudioToolbox.h>

#include "tensorflow/lite/micro/examples/micro_speech/simple_features/simple_model_settings.h"

namespace {

constexpr int kNumberRecordBuffers = 3;
bool g_is_audio_initialized = false;
constexpr int kAudioCaptureBufferSize = kAudioSampleFrequency * 0.5;
int16_t g_audio_capture_buffer[kAudioCaptureBufferSize];
int16_t g_audio_output_buffer[kMaxAudioSampleSize];
int32_t g_latest_audio_timestamp = 0;

// Checks for MacOS errors, prints information and returns a TF Lite version.
#define RETURN_IF_OS_ERROR(error, error_reporter)                           \
  do {                                                                      \
    if (error != noErr) {                                                   \
      TF_LITE_REPORT_ERROR(error_reporter, "Error: %s:%d (%d)\n", __FILE__, \
                           __LINE__, error);                                \
      return kTfLiteError;                                                  \
    }                                                                       \
  } while (0);

// Called when an audio input buffer has been filled.
void OnAudioBufferFilledCallback(
    void* user_data, AudioQueueRef queue, AudioQueueBufferRef buffer,
    const AudioTimeStamp* start_time, UInt32 num_packets,
    const AudioStreamPacketDescription* packet_description) {
  const int sample_size = buffer->mAudioDataByteSize / sizeof(float);
  const int64_t sample_offset = start_time->mSampleTime;
  const int32_t time_in_ms =
      (sample_offset + sample_size) / (kAudioSampleFrequency / 1000);
  const float* float_samples = static_cast<const float*>(buffer->mAudioData);
  for (int i = 0; i < sample_size; ++i) {
    const int capture_index = (sample_offset + i) % kAudioCaptureBufferSize;
    g_audio_capture_buffer[capture_index] = float_samples[i] * ((1 << 15) - 1);
  }
  // This is how we let the outside world know that new audio data has arrived.
  g_latest_audio_timestamp = time_in_ms;
  AudioQueueEnqueueBuffer(queue, buffer, 0, nullptr);
}

// Set up everything we need to capture audio samples from the default recording
// device on MacOS.
TfLiteStatus InitAudioRecording(tflite::ErrorReporter* error_reporter) {
  // Set up the format of the audio - single channel, 32-bit float at 16KHz.
  AudioStreamBasicDescription recordFormat = {};
  recordFormat.mSampleRate = kAudioSampleFrequency;
  recordFormat.mFormatID = kAudioFormatLinearPCM;
  recordFormat.mFormatFlags =
      kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
  recordFormat.mBitsPerChannel = 8 * sizeof(float);
  recordFormat.mChannelsPerFrame = 1;
  recordFormat.mBytesPerFrame = sizeof(float) * recordFormat.mChannelsPerFrame;
  recordFormat.mFramesPerPacket = 1;
  recordFormat.mBytesPerPacket =
      recordFormat.mBytesPerFrame * recordFormat.mFramesPerPacket;
  recordFormat.mReserved = 0;

  UInt32 propSize = sizeof(recordFormat);
  RETURN_IF_OS_ERROR(AudioFormatGetProperty(kAudioFormatProperty_FormatInfo, 0,
                                            NULL, &propSize, &recordFormat),
                     error_reporter);

  // Create a recording queue.
  AudioQueueRef queue;
  RETURN_IF_OS_ERROR(
      AudioQueueNewInput(&recordFormat, OnAudioBufferFilledCallback,
                         error_reporter, nullptr, nullptr, 0, &queue),
      error_reporter);

  // Set up the buffers we'll need.
  int buffer_bytes = 512 * sizeof(float);
  for (int i = 0; i < kNumberRecordBuffers; ++i) {
    AudioQueueBufferRef buffer;
    RETURN_IF_OS_ERROR(AudioQueueAllocateBuffer(queue, buffer_bytes, &buffer),
                       error_reporter);
    RETURN_IF_OS_ERROR(AudioQueueEnqueueBuffer(queue, buffer, 0, nullptr),
                       error_reporter);
  }

  // Start capturing audio.
  RETURN_IF_OS_ERROR(AudioQueueStart(queue, nullptr), error_reporter);

  return kTfLiteOk;
}

}  // namespace

TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    for (int i = 0; i < kMaxAudioSampleSize; ++i) {
      g_audio_output_buffer[i] = 0;
    }
    g_is_audio_initialized = true;
  }
  // This should only be called when the main thread notices that the latest
  // audio sample data timestamp has changed, so that there's new data in the
  // capture ring buffer. The ring buffer will eventually wrap around and
  // overwrite the data, but the assumption is that the main thread is checking
  // often enough and the buffer is large enough that this call will be made
  // before that happens.
  const int start_offset = start_ms * (kAudioSampleFrequency / 1000);
  const int duration_sample_count =
      duration_ms * (kAudioSampleFrequency / 1000);
  for (int i = 0; i < duration_sample_count; ++i) {
    const int capture_index = (start_offset + i) % kAudioCaptureBufferSize;
    g_audio_output_buffer[i] = g_audio_capture_buffer[capture_index];
  }

  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() { return g_latest_audio_timestamp; }
