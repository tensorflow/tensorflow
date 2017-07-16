/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/spectrogram_test_utils.h"

#include <math.h>
#include <stddef.h>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/wav/wav_io.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

bool ReadWaveFileToVector(const string& file_name, std::vector<double>* data) {
  string wav_data;
  if (!ReadFileToString(Env::Default(), file_name, &wav_data).ok()) {
    LOG(ERROR) << "Wave file read failed for " << file_name;
    return false;
  }
  std::vector<float> decoded_data;
  uint32 decoded_sample_count;
  uint16 decoded_channel_count;
  uint32 decoded_sample_rate;
  if (!wav::DecodeLin16WaveAsFloatVector(
           wav_data, &decoded_data, &decoded_sample_count,
           &decoded_channel_count, &decoded_sample_rate)
           .ok()) {
    return false;
  }
  // Convert from float to double for the output value.
  data->resize(decoded_data.size());
  for (int i = 0; i < decoded_data.size(); ++i) {
    (*data)[i] = decoded_data[i];
  }
  return true;
}

bool ReadRawFloatFileToComplexVector(
    const string& file_name, int row_length,
    std::vector<std::vector<std::complex<double> > >* data) {
  data->clear();
  string data_string;
  if (!ReadFileToString(Env::Default(), file_name, &data_string).ok()) {
    LOG(ERROR) << "Failed to open file " << file_name;
    return false;
  }
  float real_out;
  float imag_out;
  const int kBytesPerValue = 4;
  CHECK_EQ(sizeof(real_out), kBytesPerValue);
  std::vector<std::complex<double> > data_row;
  int row_counter = 0;
  int offset = 0;
  const int end = data_string.size();
  while (offset < end) {
    memcpy(&real_out, data_string.data() + offset, kBytesPerValue);
    offset += kBytesPerValue;
    memcpy(&imag_out, data_string.data() + offset, kBytesPerValue);
    offset += kBytesPerValue;
    if (row_counter >= row_length) {
      data->push_back(data_row);
      data_row.clear();
      row_counter = 0;
    }
    data_row.push_back(std::complex<double>(real_out, imag_out));
    ++row_counter;
  }
  if (row_counter >= row_length) {
    data->push_back(data_row);
  }
  return true;
}

void ReadCSVFileToComplexVectorOrDie(
    const string& file_name,
    std::vector<std::vector<std::complex<double> > >* data) {
  data->clear();
  string data_string;
  if (!ReadFileToString(Env::Default(), file_name, &data_string).ok()) {
    LOG(FATAL) << "Failed to open file " << file_name;
    return;
  }
  std::vector<string> lines = str_util::Split(data_string, '\n');
  for (const string& line : lines) {
    if (line.empty()) {
      continue;
    }
    std::vector<std::complex<double> > data_line;
    std::vector<string> values = str_util::Split(line, ',');
    for (std::vector<string>::const_iterator i = values.begin();
         i != values.end(); ++i) {
      // each element of values may be in the form:
      // 0.001+0.002i, 0.001, 0.001i, -1.2i, -1.2-3.2i, 1.5, 1.5e-03+21.0i
      std::vector<string> parts;
      // Find the first instance of + or - after the second character
      // in the string, that does not immediately follow an 'e'.
      size_t operator_index = i->find_first_of("+-", 2);
      if (operator_index < i->size() &&
          i->substr(operator_index - 1, 1) == "e") {
        operator_index = i->find_first_of("+-", operator_index + 1);
      }
      parts.push_back(i->substr(0, operator_index));
      if (operator_index < i->size()) {
        parts.push_back(i->substr(operator_index, string::npos));
      }

      double real_part = 0.0;
      double imaginary_part = 0.0;
      for (std::vector<string>::const_iterator j = parts.begin();
           j != parts.end(); ++j) {
        if (j->find_first_of("ij") != string::npos) {
          strings::safe_strtod((*j).c_str(), &imaginary_part);
        } else {
          strings::safe_strtod((*j).c_str(), &real_part);
        }
      }
      data_line.push_back(std::complex<double>(real_part, imaginary_part));
    }
    data->push_back(data_line);
  }
}

void ReadCSVFileToArrayOrDie(const string& filename,
                             std::vector<std::vector<float> >* array) {
  string contents;
  TF_CHECK_OK(ReadFileToString(Env::Default(), filename, &contents));
  std::vector<string> lines = str_util::Split(contents, '\n');
  contents.clear();

  array->clear();
  std::vector<float> values;
  for (int l = 0; l < lines.size(); ++l) {
    values.clear();
    CHECK(str_util::SplitAndParseAsFloats(lines[l], ',', &values));
    array->push_back(values);
  }
}

bool WriteDoubleVectorToFile(const string& file_name,
                             const std::vector<double>& data) {
  std::unique_ptr<WritableFile> file;
  if (!Env::Default()->NewWritableFile(file_name, &file).ok()) {
    LOG(ERROR) << "Failed to open file " << file_name;
    return false;
  }
  for (int i = 0; i < data.size(); ++i) {
    if (!file->Append(StringPiece(reinterpret_cast<const char*>(&(data[i])),
                                  sizeof(data[i])))
             .ok()) {
      LOG(ERROR) << "Failed to append to file " << file_name;
      return false;
    }
  }
  if (!file->Close().ok()) {
    LOG(ERROR) << "Failed to close file " << file_name;
    return false;
  }
  return true;
}

bool WriteFloatVectorToFile(const string& file_name,
                            const std::vector<float>& data) {
  std::unique_ptr<WritableFile> file;
  if (!Env::Default()->NewWritableFile(file_name, &file).ok()) {
    LOG(ERROR) << "Failed to open file " << file_name;
    return false;
  }
  for (int i = 0; i < data.size(); ++i) {
    if (!file->Append(StringPiece(reinterpret_cast<const char*>(&(data[i])),
                                  sizeof(data[i])))
             .ok()) {
      LOG(ERROR) << "Failed to append to file " << file_name;
      return false;
    }
  }
  if (!file->Close().ok()) {
    LOG(ERROR) << "Failed to close file " << file_name;
    return false;
  }
  return true;
}

bool WriteDoubleArrayToFile(const string& file_name, int size,
                            const double* data) {
  std::unique_ptr<WritableFile> file;
  if (!Env::Default()->NewWritableFile(file_name, &file).ok()) {
    LOG(ERROR) << "Failed to open file " << file_name;
    return false;
  }
  for (int i = 0; i < size; ++i) {
    if (!file->Append(StringPiece(reinterpret_cast<const char*>(&(data[i])),
                                  sizeof(data[i])))
             .ok()) {
      LOG(ERROR) << "Failed to append to file " << file_name;
      return false;
    }
  }
  if (!file->Close().ok()) {
    LOG(ERROR) << "Failed to close file " << file_name;
    return false;
  }
  return true;
}

bool WriteFloatArrayToFile(const string& file_name, int size,
                           const float* data) {
  std::unique_ptr<WritableFile> file;
  if (!Env::Default()->NewWritableFile(file_name, &file).ok()) {
    LOG(ERROR) << "Failed to open file " << file_name;
    return false;
  }
  for (int i = 0; i < size; ++i) {
    if (!file->Append(StringPiece(reinterpret_cast<const char*>(&(data[i])),
                                  sizeof(data[i])))
             .ok()) {
      LOG(ERROR) << "Failed to append to file " << file_name;
      return false;
    }
  }
  if (!file->Close().ok()) {
    LOG(ERROR) << "Failed to close file " << file_name;
    return false;
  }
  return true;
}

bool WriteComplexVectorToRawFloatFile(
    const string& file_name,
    const std::vector<std::vector<std::complex<double> > >& data) {
  std::unique_ptr<WritableFile> file;
  if (!Env::Default()->NewWritableFile(file_name, &file).ok()) {
    LOG(ERROR) << "Failed to open file " << file_name;
    return false;
  }
  for (int i = 0; i < data.size(); ++i) {
    for (int j = 0; j < data[i].size(); ++j) {
      const float real_part(real(data[i][j]));
      if (!file->Append(StringPiece(reinterpret_cast<const char*>(&real_part),
                                    sizeof(real_part)))
               .ok()) {
        LOG(ERROR) << "Failed to append to file " << file_name;
        return false;
      }

      const float imag_part(imag(data[i][j]));
      if (!file->Append(StringPiece(reinterpret_cast<const char*>(&imag_part),
                                    sizeof(imag_part)))
               .ok()) {
        LOG(ERROR) << "Failed to append to file " << file_name;
        return false;
      }
    }
  }
  if (!file->Close().ok()) {
    LOG(ERROR) << "Failed to close file " << file_name;
    return false;
  }
  return true;
}

void SineWave(int sample_rate, float frequency, float duration_seconds,
              std::vector<double>* data) {
  data->clear();
  for (int i = 0; i < static_cast<int>(sample_rate * duration_seconds); ++i) {
    data->push_back(
        sin(2.0 * M_PI * i * frequency / static_cast<double>(sample_rate)));
  }
}

}  // namespace tensorflow
