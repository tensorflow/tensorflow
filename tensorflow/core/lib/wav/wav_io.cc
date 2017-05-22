/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/platform/cpu_info.h"
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

constexpr char kRiffChunkId[] = "RIFF";
constexpr char kRiffType[] = "WAVE";
constexpr char kFormatChunkId[] = "fmt ";
constexpr char kDataChunkId[] = "data";

inline int16 FloatToInt16Sample(float data) {
  constexpr float kMultiplier = 1.0f * (1 << 15);
  return std::min<float>(std::max<float>(roundf(data * kMultiplier), kint16min),
                         kint16max);
}

inline float Int16SampleToFloat(int16 data) {
  constexpr float kMultiplier = 1.0f / (1 << 15);
  return data * kMultiplier;
}

Status ExpectText(const string& data, const string& expected_text,
                  int* offset) {
  const int new_offset = *offset + expected_text.size();
  if (new_offset > data.size()) {
    return errors::InvalidArgument("Data too short when trying to read ",
                                   expected_text);
  }
  const string found_text(data.begin() + *offset, data.begin() + new_offset);
  if (found_text != expected_text) {
    return errors::InvalidArgument("Header mismatch: Expected ", expected_text,
                                   " but found ", found_text);
  }
  *offset = new_offset;
  return Status::OK();
}

template <class T>
Status ReadValue(const string& data, T* value, int* offset) {
  const int new_offset = *offset + sizeof(T);
  if (new_offset > data.size()) {
    return errors::InvalidArgument("Data too short when trying to read value");
  }
  if (port::kLittleEndian) {
    memcpy(value, data.data() + *offset, sizeof(T));
  } else {
    *value = 0;
    const uint8* data_buf =
        reinterpret_cast<const uint8*>(data.data() + *offset);
    int shift = 0;
    for (int i = 0; i < sizeof(T); ++i, shift += 8) {
      *value = *value | (data_buf[i] >> shift);
    }
  }
  *offset = new_offset;
  return Status::OK();
}

}  // namespace

Status EncodeAudioAsS16LEWav(const float* audio, size_t sample_rate,
                             size_t num_channels, size_t num_frames,
                             string* wav_string) {
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

Status DecodeLin16WaveAsFloatVector(const string& wav_string,
                                    std::vector<float>* float_values,
                                    uint32* sample_count, uint16* channel_count,
                                    uint32* sample_rate) {
  int offset = 0;
  TF_RETURN_IF_ERROR(ExpectText(wav_string, kRiffChunkId, &offset));
  uint32 total_file_size;
  TF_RETURN_IF_ERROR(ReadValue<uint32>(wav_string, &total_file_size, &offset));
  TF_RETURN_IF_ERROR(ExpectText(wav_string, kRiffType, &offset));
  TF_RETURN_IF_ERROR(ExpectText(wav_string, kFormatChunkId, &offset));
  uint32 format_chunk_size;
  TF_RETURN_IF_ERROR(
      ReadValue<uint32>(wav_string, &format_chunk_size, &offset));
  if ((format_chunk_size != 16) && (format_chunk_size != 18)) {
    return errors::InvalidArgument(
        "Bad file size for WAV: Expected 16 or 18, but got", format_chunk_size);
  }
  uint16 audio_format;
  TF_RETURN_IF_ERROR(ReadValue<uint16>(wav_string, &audio_format, &offset));
  if (audio_format != 1) {
    return errors::InvalidArgument(
        "Bad audio format for WAV: Expected 1 (PCM), but got", audio_format);
  }
  TF_RETURN_IF_ERROR(ReadValue<uint16>(wav_string, channel_count, &offset));
  TF_RETURN_IF_ERROR(ReadValue<uint32>(wav_string, sample_rate, &offset));
  uint32 bytes_per_second;
  TF_RETURN_IF_ERROR(ReadValue<uint32>(wav_string, &bytes_per_second, &offset));
  uint16 bytes_per_sample;
  TF_RETURN_IF_ERROR(ReadValue<uint16>(wav_string, &bytes_per_sample, &offset));
  // Confusingly, bits per sample is defined as holding the number of bits for
  // one channel, unlike the definition of sample used elsewhere in the WAV
  // spec. For example, bytes per sample is the memory needed for all channels
  // for one point in time.
  uint16 bits_per_sample;
  TF_RETURN_IF_ERROR(ReadValue<uint16>(wav_string, &bits_per_sample, &offset));
  if (bits_per_sample != 16) {
    return errors::InvalidArgument(
        "Can only read 16-bit WAV files, but received ", bits_per_sample);
  }
  const uint32 expected_bytes_per_sample =
      ((bits_per_sample * *channel_count) + 7) / 8;
  if (bytes_per_sample != expected_bytes_per_sample) {
    return errors::InvalidArgument(
        "Bad bytes per sample in WAV header: Expected ",
        expected_bytes_per_sample, " but got ", bytes_per_sample);
  }
  const uint32 expected_bytes_per_second =
      (bytes_per_sample * (*sample_rate)) / *channel_count;
  if (bytes_per_second != expected_bytes_per_second) {
    return errors::InvalidArgument(
        "Bad bytes per second in WAV header: Expected ",
        expected_bytes_per_second, " but got ", bytes_per_second,
        " (sample_rate=", *sample_rate, ", bytes_per_sample=", bytes_per_sample,
        ")");
  }
  if (format_chunk_size == 18) {
    // Skip over this unused section.
    offset += 2;
  }
  TF_RETURN_IF_ERROR(ExpectText(wav_string, kDataChunkId, &offset));
  uint32 data_size;
  TF_RETURN_IF_ERROR(ReadValue<uint32>(wav_string, &data_size, &offset));
  *sample_count = data_size / bytes_per_sample;
  const uint32 data_count = *sample_count * *channel_count;
  float_values->resize(data_count);
  for (int i = 0; i < data_count; ++i) {
    int16 single_channel_value = 0;
    TF_RETURN_IF_ERROR(
        ReadValue<int16>(wav_string, &single_channel_value, &offset));
    (*float_values)[i] = Int16SampleToFloat(single_channel_value);
  }
  return Status::OK();
}

}  // namespace wav
}  // namespace tensorflow
