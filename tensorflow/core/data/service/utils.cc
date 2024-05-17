/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/utils.h"

#include <memory>
#include <string>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"

namespace tensorflow {
namespace data {

Status WriteDatasetDef(const std::string& path, const DatasetDef& dataset_def) {
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(Env::Default()->NewWritableFile(path, &file));
  io::RecordWriter writer(file.get());
  TF_RETURN_IF_ERROR(writer.WriteRecord(dataset_def.SerializeAsString()));
  return absl::OkStatus();
}

Status ReadDatasetDef(const std::string& path, DatasetDef& dataset_def) {
  if (path.empty()) {
    return errors::InvalidArgument("Path is empty");
  }
  TF_RETURN_IF_ERROR(Env::Default()->FileExists(path));
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(Env::Default()->NewRandomAccessFile(path, &file));
  io::RecordReader reader(file.get());
  uint64 offset = 0;
  tstring record;
  TF_RETURN_IF_ERROR(reader.ReadRecord(&offset, &record));
  if (!dataset_def.ParseFromString(record)) {
    return errors::DataLoss("Failed to parse dataset definition");
  }
  return absl::OkStatus();
}

}  // namespace data
}  // namespace tensorflow
