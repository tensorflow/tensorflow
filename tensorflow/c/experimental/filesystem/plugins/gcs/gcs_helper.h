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
#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_GCS_HELPER_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_GCS_HELPER_H_

#include <fstream>
#include <ios>
#include <string>

class TempFile : public std::fstream {
 public:
  // We should specify openmode each time we call TempFile.
  TempFile(const std::string& temp_file_name, std::ios::openmode mode);
  TempFile(TempFile&& rhs);
  ~TempFile() override;
  const std::string getName() const;
  bool truncate();

 private:
  const std::string name_;
};

#endif  // TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_GCS_HELPER_H_
