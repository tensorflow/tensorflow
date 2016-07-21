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

#include "tensorflow/examples/android/jni/jni_utils.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <jni.h>
#include <stdlib.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message_lite.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/examples/android/jni/limiting_file_input_stream.h"

static const char* const ASSET_PREFIX = "file:///android_asset/";

namespace {
class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
 public:
  explicit IfstreamInputStream(const std::string& file_name)
      : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
  ~IfstreamInputStream() { ifs_.close(); }

  int Read(void* buffer, int size) {
    if (!ifs_) {
      return -1;
    }
    ifs_.read(static_cast<char*>(buffer), size);
    return ifs_.gcount();
  }

 private:
  std::ifstream ifs_;
};

}  // namespace

bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto) {
  ::google::protobuf::io::CopyingInputStreamAdaptor stream(
      new IfstreamInputStream(file_name));
  stream.SetOwnsCopyingStream(true);
  // TODO(jiayq): the following coded stream is for debugging purposes to allow
  // one to parse arbitrarily large messages for MessageLite. One most likely
  // doesn't want to put protobufs larger than 64MB on Android, so we should
  // eventually remove this and quit loud when a large protobuf is passed in.
  ::google::protobuf::io::CodedInputStream coded_stream(&stream);
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

bool IsAsset(const char* const filename) {
  return strstr(filename, ASSET_PREFIX) == filename;
}

void ReadFileToProto(AAssetManager* const asset_manager,
                     const char* const filename,
                     google::protobuf::MessageLite* message) {
  if (!IsAsset(filename)) {
    VLOG(0) << "Opening file: " << filename;
    CHECK(PortableReadFileToProto(filename, message));
    return;
  }

  CHECK_NOTNULL(asset_manager);

  const char* const asset_filename = filename + strlen(ASSET_PREFIX);
  AAsset* asset =
      AAssetManager_open(asset_manager, asset_filename, AASSET_MODE_STREAMING);
  CHECK_NOTNULL(asset);

  off_t start;
  off_t length;
  const int fd = AAsset_openFileDescriptor(asset, &start, &length);

  if (fd >= 0) {
    ::tensorflow::android::LimitingFileInputStream is(fd, start + length);
    google::protobuf::io::CopyingInputStreamAdaptor adaptor(&is);
    // If the file is smaller than protobuf's default limit, avoid copies.
    if (length < (64 * 1024 * 1024)) {
      // If it has a file descriptor that means it can be memmapped directly
      // from the APK.
      VLOG(0) << "Opening asset " << asset_filename
              << " from disk with zero-copy.";
      adaptor.Skip(start);
      CHECK(message->ParseFromZeroCopyStream(&adaptor));
    } else {
      ::google::protobuf::io::CodedInputStream coded_stream(&adaptor);
      // Total bytes hard limit / warning limit are set to 1GB and 512MB
      // respectively.
      coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
      coded_stream.Skip(start);
      CHECK(message->ParseFromCodedStream(&coded_stream));
    }
  } else {
    // It may be compressed, in which case we have to uncompress
    // it to memory first.
    VLOG(0) << "Opening asset " << asset_filename << " from disk with copy.";
    const off_t data_size = AAsset_getLength(asset);
    const void* const memory = AAsset_getBuffer(asset);
    CHECK(message->ParseFromArray(memory, data_size));
  }
  AAsset_close(asset);
}

void ReadFileToString(AAssetManager* const asset_manager,
                      const char* const filename, std::string* str) {
  if (!IsAsset(filename)) {
    VLOG(0) << "Opening file: " << filename;
    std::ifstream t(filename);
    std::string tmp((std::istreambuf_iterator<char>(t)),
                    std::istreambuf_iterator<char>());
    tmp.swap(*str);
    t.close();
    return;
  }

  CHECK_NOTNULL(asset_manager);
  const char* const asset_filename = filename + strlen(ASSET_PREFIX);
  AAsset* asset =
      AAssetManager_open(asset_manager, asset_filename, AASSET_MODE_STREAMING);
  CHECK_NOTNULL(asset);
  VLOG(0) << "Opening asset " << asset_filename << " from disk with copy.";
  const off_t data_size = AAsset_getLength(asset);
  const char* memory = reinterpret_cast<const char*>(AAsset_getBuffer(asset));

  std::string tmp(memory, memory + data_size);
  tmp.swap(*str);
  AAsset_close(asset);
}

void ReadFileToVector(AAssetManager* const asset_manager,
                      const char* const filename,
                      std::vector<std::string>* str_vector) {
  std::string labels_string;
  ReadFileToString(asset_manager, filename, &labels_string);
  std::istringstream ifs(labels_string);
  str_vector->clear();
  std::string label;
  while (std::getline(ifs, label)) {
    str_vector->push_back(label);
  }
  VLOG(0) << "Read " << str_vector->size() << " values from " << filename;
}

void WriteProtoToFile(const char* const filename,
                      const google::protobuf::MessageLite& message) {
  std::fstream outfile;
  outfile.open(filename, std::fstream::binary | std::fstream::out);
  std::string serialized;
  message.SerializeToString(&serialized);
  outfile.write(serialized.c_str(), serialized.size());
  outfile.close();
  if (outfile.fail()) {
    LOG(WARNING) << "Failed to write proto to " << filename;
    return;
  }
  VLOG(0) << "Wrote proto to " << filename;
}
