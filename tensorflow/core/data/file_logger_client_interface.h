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
#ifndef TENSORFLOW_CORE_DATA_FILE_LOGGER_CLIENT_INTERFACE_H_
#define TENSORFLOW_CORE_DATA_FILE_LOGGER_CLIENT_INTERFACE_H_

#include <string>
#include <vector>

namespace tensorflow::data {

// An abstract class to provides an easy and thread safe api to make
// asynchronous calls to the TFDataLoggerService.
// LogFilesAsync is guaranteed to be non blocking.
// The destructor however might be blocking.
class FileLoggerClientInterface {
 public:
  // Default constructor
  FileLoggerClientInterface() = default;

  // Sends file names in `files` to the TFDataLoggerService. Asynchronously.
  virtual void LogFilesAsync(std::vector<std::string> files) = 0;

  // Default destructor. May block depending on implementation of the derived
  // class.
  virtual ~FileLoggerClientInterface() = default;
};
}  // namespace tensorflow::data

#endif  // TENSORFLOW_CORE_DATA_FILE_LOGGER_CLIENT_INTERFACE_H_
