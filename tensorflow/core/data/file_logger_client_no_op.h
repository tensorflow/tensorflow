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
#ifndef TENSORFLOW_CORE_DATA_FILE_LOGGER_CLIENT_NO_OP_H_
#define TENSORFLOW_CORE_DATA_FILE_LOGGER_CLIENT_NO_OP_H_

#include <string>
#include <vector>

#include "tensorflow/core/data/file_logger_client_interface.h"

namespace tensorflow::data {

// Implementation of the abstract class FileLoggerClientInterface, which does
// nothing. It does not allocate any resources and immediately returns in
// LogFilesAsync.3rd This is used in 3rd party version of the tf.data library.
class FileLoggerClientNoOp : public FileLoggerClientInterface {
 public:
  // Default constructor
  FileLoggerClientNoOp() = default;

  // Does not do anything
  void LogFilesAsync(std::vector<std::string> files) override{};

  // Default destructor
  ~FileLoggerClientNoOp() override = default;
};
}  // namespace tensorflow::data

#endif  // TENSORFLOW_CORE_DATA_FILE_LOGGER_CLIENT_NO_OP_H_
