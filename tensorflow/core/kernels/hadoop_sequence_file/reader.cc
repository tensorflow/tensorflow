/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/hadoop_sequence_file/reader.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace io {

namespace {
const int kSyncMarkerSize = 16;
// The value class name in the header must match this, i.e., the values are
// raw bytes.
const char kBytesWritableClassName[] = "org.apache.hadoop.io.BytesWritable";
}  // namespace

// static
const SequenceFileReaderOptions& SequenceFileReaderOptions::Defaults() {
  static const SequenceFileReaderOptions* options = [](){
    auto* opt = new SequenceFileReaderOptions;
    opt->buffer_size_bytes = 1024 * 1024;    // 1M buffer by default.
    return opt;
  }();
  return *options;
}

SequenceFileReader::SequenceFileReader(
    RandomAccessFile* file, const SequenceFileReaderOptions& options)
  : options_(options),
    stream_(new BufferedInputStream(file, options_.buffer_size_bytes)) {
  TF_CHECK_OK(stream_->ReadNBytes(3, &buf_));
  CHECK_EQ(buf_, "SEQ") << "Invalid hadoop sequence file, beginning with "
                        << buf_ << ", expecting SEQ.";
  TF_CHECK_OK(stream_->ReadNBytes(1, &buf_));
  CHECK_GE(buf_[0], 5) << "Sequence file ver " << buf_[0]
                       << " < 5 unsupported.";
  // We do not care about the key encoding.
  ReadHadoopVarLenStringOrDie();
  ReadHadoopVarLenStringOrDie();
  CHECK_EQ(buf_, kBytesWritableClassName)
      << "Sequence file value must be raw bytes.";
  // Compression unsupported.
  TF_CHECK_OK(stream_->ReadNBytes(2, &buf_));
  CHECK_EQ(buf_[1], 0) << "Sequence file with block compression unsupported.";
  CHECK_EQ(buf_[0], 0) << "Sequence file with record compression unsupported.";
  // Discard all the metadata.
  ConsumeMetadataOrDie();
  TF_CHECK_OK(stream_->ReadNBytes(kSyncMarkerSize, &sync_marker_));
}

Status SequenceFileReader::ReadRecord(string* value) {
  CHECK(value != nullptr);
  int32 total_len = 0;
  TF_RETURN_IF_ERROR(ReadBigEndianInt32(&total_len));
  if (total_len == -1) {  // sync marker.
    TF_RETURN_IF_ERROR(stream_->ReadNBytes(kSyncMarkerSize, &buf_));
    if (buf_ != sync_marker_) {
      return errors::DataLoss(
          "Sync marker [", buf_, "] != [", sync_marker_, "] in header");
    }
    return ReadRecord(value);
  }
  if (total_len < 8) {
    return errors::DataLoss("Invalid record length ", total_len, " < 8");
  }
  int32 key_len = 0;
  TF_RETURN_IF_ERROR(ReadBigEndianInt32(&key_len));
  if (key_len < 4) {
    return errors::DataLoss("Invalid key length ", key_len, " < 4");
  }
  const int32 val_len = total_len - key_len;
  if (val_len < 4) {
    return errors::DataLoss("Raw byte value length ", val_len, " < 4");
  }
  // Ignore key.
  TF_RETURN_IF_ERROR(stream_->ReadNBytes(key_len, &buf_));
  TF_RETURN_IF_ERROR(stream_->ReadNBytes(val_len, &buf_));
  // For raw bytes, hadoop add 4 bytes in front before writing to sequence file.
  *value = buf_.substr(4);
  return Status::OK();
}

// NOTE(yi.sun): This is the most moronic integer encoding I have ever seen.
int64 SequenceFileReader::ReadHadoopVarIntOrDie() {
  TF_CHECK_OK(stream_->ReadNBytes(1, &buf_));
  const int8 len = buf_[0];
  int remaining_bytes = 0;
  bool negative = false;
  if (len >= -112) {
    return int64(len);
  }
  if (len >= -120) {
    remaining_bytes = int(-112 - len);
  } else {
    remaining_bytes = int(-120 - len);
    negative = true;
  }

  uint64 result = 0;
  TF_CHECK_OK(stream_->ReadNBytes(remaining_bytes, &buf_));
  for (uint8 b : buf_) {
    result = ((result << 8) | uint64(b));
  }
  if (negative) {
    result = ~result;
  }
  return int64(result);
}

void SequenceFileReader::ReadHadoopVarLenStringOrDie() {
  const int64 len = ReadHadoopVarIntOrDie();
  TF_CHECK_OK(stream_->ReadNBytes(len, &buf_));
}

Status SequenceFileReader::ReadBigEndianUint32(uint32* out) {
  TF_RETURN_IF_ERROR(stream_->ReadNBytes(4, &buf_));
  *out = (uint8(buf_[0]) << 24) + (uint8(buf_[1]) << 16) +
         (uint8(buf_[2]) << 8) + uint8(buf_[3]);
  return Status::OK();
}

Status SequenceFileReader::ReadBigEndianInt32(int32* out) {
  uint32 u;
  TF_RETURN_IF_ERROR(ReadBigEndianUint32(&u));
  *out = static_cast<int32>(u);
  return Status::OK();
}

void SequenceFileReader::ConsumeMetadataOrDie() {
  uint32 num_key_values = 0;
  TF_CHECK_OK(ReadBigEndianUint32(&num_key_values));
  CHECK_LE(num_key_values, 1024)
      << "Meta data contains " << num_key_values << " > 1024 entries.";
  // We are not interested in the actual content of the metadata.
  for (int i = 0; i < num_key_values; i++) {
    ReadHadoopVarLenStringOrDie();  // key
    ReadHadoopVarLenStringOrDie();  // value
  }
}

}  // namespace io
}  // namespace tensorflow
