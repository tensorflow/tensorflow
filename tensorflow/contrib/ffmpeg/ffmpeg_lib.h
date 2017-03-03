// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_FFMPEG_FFMPEG_LIB_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_FFMPEG_FFMPEG_LIB_H_

#include <string>
#include <vector>

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace ffmpeg {

// Gets a temp filename in an appropriate location.
string GetTempFilename(const string& extension);

// Reads an audio file using ffmpeg and converts it into an array of samples in
// [-1.0, 1.0]. If there are multiple channels in the audio then each frame will
// contain a separate sample for each channel. Frames are ordered by time.
Status ReadAudioFile(const string& filename,
                     const string& audio_format_id,
                     int32 samples_per_second,
                     int32 channel_count,
                     std::vector<float>* output_samples);

// Creates an audio file using ffmpeg in a specific format. The samples are in
// [-1.0, 1.0]. If there are multiple channels in the audio then each frame will
// contain a separate sample for each channel. Frames are ordered by time.
// Currently, the implementation only supports wav files, and ffmpeg is not used
// to create them.
Status CreateAudioFile(const string& audio_format_id, int32 bits_per_second,
                       int32 samples_per_second, int32 channel_count,
                       const std::vector<float>& samples, string* output_data);

}  // namespace ffmpeg
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_FFMPEG_DEFAULT_FFMPEG_LIB_H_
