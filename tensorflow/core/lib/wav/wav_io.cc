/* Copyright 2016 Google Inc. All Rights Reserved.

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

// Functions to write audio in WAV format.

#include <math.h>
#include <string.h>
#include <algorithm>

#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/wav/wav_io.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace wav {
namespace {

struct TF_PACKED RiffChunk {
  char chunk_id[4];
  char chunk_data_size[4];
  char riff_type[4];
};
static_assert(sizeof(RiffChunk) == 12, "TF_PACKED does not work.");

struct TF_PACKED FormatChunk {
  char chunk_id[4];
  char chunk_data_size[4];
  char compression_code[2];
  char channel_numbers[2];
  char sample_rate[4];
  char bytes_per_second[4];
  char bytes_per_frame[2];
  char bits_per_sample[2];
};
static_assert(sizeof(FormatChunk) == 24, "TF_PACKED does not work.");

struct TF_PACKED DataChunk {
  char chunk_id[4];
  char chunk_data_size[4];
};
static_assert(sizeof(DataChunk) == 8, "TF_PACKED does not work.");

struct TF_PACKED WavHeader {
  RiffChunk riff_chunk;
  FormatChunk format_chunk;
  DataChunk data_chunk;
};
static_assert(sizeof(WavHeader) ==
                  sizeof(RiffChunk) + sizeof(FormatChunk) + sizeof(DataChunk),
              "TF_PACKED does not work.");

inline int16 FloatToInt16Sample(float data) {
  constexpr float kMultiplier = 1.0f * (1 << 15);
  return std::min<float>(std::max<float>(roundf(data * kMultiplier), kint16min),
                         kint16max);
}

}  // namespace

Status EncodeAudioAsS16LEWav(const float* audio, size_t sample_rate,
                             size_t num_channels, size_t num_frames,
                             string* wav_string) {
  constexpr char kRiffChunkId[] = "RIFF";
  constexpr char kRiffType[] = "WAVE";
  constexpr char kFormatChunkId[] = "fmt ";
  constexpr char kDataChunkId[] = "data";
  constexpr size_t kFormatChunkSize = 16;
  constexpr size_t kCompressionCodePcm = 1;
  constexpr size_t kBitsPerSample = 16;
  constexpr size_t kBytesPerSample = kBitsPerSample / 8;
  constexpr size_t kHeaderSize = sizeof(WavHeader);

  if (audio == nullptr) {
    return errors::InvalidArgument("audio is null");
  }
  if (wav_string == nullptr) {
    return errors::InvalidArgument("wav_string is null");
  }
  if (sample_rate == 0 || sample_rate > kuint32max) {
    return errors::InvalidArgument("sample_rate must be in (0, 2^32), got: ",
                                   sample_rate);
  }
  if (num_channels == 0 || num_channels > kuint16max) {
    return errors::InvalidArgument("num_channels must be in (0, 2^16), got: ",
                                   num_channels);
  }
  if (num_frames == 0) {
    return errors::InvalidArgument("num_frames must be positive.");
  }

  const size_t bytes_per_second = sample_rate * kBytesPerSample;
  const size_t num_samples = num_frames * num_channels;
  const size_t data_size = num_samples * kBytesPerSample;
  const size_t file_size = kHeaderSize + num_samples * kBytesPerSample;
  const size_t bytes_per_frame = kBytesPerSample * num_channels;

  // WAV represents the length of the file as a uint32 so file_size cannot
  // exceed kuint32max.
  if (file_size > kuint32max) {
    return errors::InvalidArgument(
        "Provided channels and frames cannot be encoded as a WAV.");
  }

  wav_string->resize(file_size);
  char* data = &wav_string->at(0);
  WavHeader* header = bit_cast<WavHeader*>(data);

  // Fill RIFF chunk.
  auto* riff_chunk = &header->riff_chunk;
  memcpy(riff_chunk->chunk_id, kRiffChunkId, 4);
  core::EncodeFixed32(riff_chunk->chunk_data_size, file_size - 8);
  memcpy(riff_chunk->riff_type, kRiffType, 4);

  // Fill format chunk.
  auto* format_chunk = &header->format_chunk;
  memcpy(format_chunk->chunk_id, kFormatChunkId, 4);
  core::EncodeFixed32(format_chunk->chunk_data_size, kFormatChunkSize);
  core::EncodeFixed16(format_chunk->compression_code, kCompressionCodePcm);
  core::EncodeFixed16(format_chunk->channel_numbers, num_channels);
  core::EncodeFixed32(format_chunk->sample_rate, sample_rate);
  core::EncodeFixed32(format_chunk->bytes_per_second, bytes_per_second);
  core::EncodeFixed16(format_chunk->bytes_per_frame, bytes_per_frame);
  core::EncodeFixed16(format_chunk->bits_per_sample, kBitsPerSample);

  // Fill data chunk.
  auto* data_chunk = &header->data_chunk;
  memcpy(data_chunk->chunk_id, kDataChunkId, 4);
  core::EncodeFixed32(data_chunk->chunk_data_size, data_size);

  // Write the audio.
  data += kHeaderSize;
  for (size_t i = 0; i < num_samples; ++i) {
    int16 sample = FloatToInt16Sample(audio[i]);
    core::EncodeFixed16(&data[i * kBytesPerSample],
                        static_cast<uint16>(sample));
  }
  return Status::OK();
}

}  // namespace wav
}  // namespace tensorflow
