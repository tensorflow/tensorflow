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

%include "tensorflow/python/lib/core/strings.i"
%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/match.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/util/tf_status_helper.h"
%}

%{
inline bool FileExists(const string& filename) {
  return tensorflow::Env::Default()->FileExists(filename);
}

inline bool FileExists(const tensorflow::StringPiece& filename) {
  return tensorflow::Env::Default()->FileExists(filename.ToString());
}

inline void DeleteFile(const string& filename, TF_Status* out_status) {
  tensorflow::Status status = tensorflow::Env::Default()->DeleteFile(filename);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }
}

string ReadFileToString(const string& filename, TF_Status* out_status) {
  string file_content;
  tensorflow::Status status = ReadFileToString(tensorflow::Env::Default(),
      filename, &file_content);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }
  return file_content;
}

void WriteStringToFile(const string& filename, const string& file_content,
                       TF_Status* out_status) {
  tensorflow::Status status = WriteStringToFile(tensorflow::Env::Default(),
      filename, file_content);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }
}

std::vector<string> GetMatchingFiles(const string& filename,
                                     TF_Status* out_status) {
  std::vector<string> results;
  tensorflow::Status status =
      tensorflow::io::GetMatchingFiles(tensorflow::Env::Default(), filename,
          &results);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }
  return results;
}

void CreateDir(const string& dirname, TF_Status* out_status) {
  tensorflow::Status status = tensorflow::Env::Default()->CreateDir(dirname);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }
}
%}

// Wrap the above functions.
inline bool FileExists(const string& filename);
inline void DeleteFile(const string& filename, TF_Status* out_status);
string ReadFileToString(const string& filename, TF_Status* out_status);
void WriteStringToFile(const string& filename, const string& file_content,
                       TF_Status* out_status);
std::vector<string> GetMatchingFiles(const string& filename,
                                     TF_Status* out_status);
void CreateDir(const string& dirname, TF_Status* out_status);

%ignoreall

%unignore tensorflow;

%insert("python") %{
  def file_exists(filename):
    from tensorflow.python.util import compat
    return FileExists(compat.as_bytes(filename))

  def delete_file(filename):
    from tensorflow.python.framework import errors
    with errors.raise_exception_on_not_ok_status() as status:
      from tensorflow.python.util import compat
      DeleteFile(compat.as_bytes(filename), status)

  def read_file_to_string(filename):
    from tensorflow.python.framework import errors
    with errors.raise_exception_on_not_ok_status() as status:
      from tensorflow.python.util import compat
      return ReadFileToString(compat.as_bytes(filename), status);

  def write_string_to_file(filename, file_content):
    from tensorflow.python.framework import errors
    with errors.raise_exception_on_not_ok_status() as status:
      from tensorflow.python.util import compat
      WriteStringToFile(compat.as_bytes(filename),
          compat.as_bytes(file_content), status)

  def get_matching_files(filename):
    from tensorflow.python.framework import errors
    with errors.raise_exception_on_not_ok_status() as status:
      from tensorflow.python.util import compat
      return GetMatchingFiles(compat.as_bytes(filename), status)

  def create_dir(dirname):
    from tensorflow.python.framework import errors
    with errors.raise_exception_on_not_ok_status() as status:
      from tensorflow.python.util import compat
      CreateDir(compat.as_bytes(partial_dir), status)

  def recursive_create_dir(dirname):
    from tensorflow.python.framework import errors
    with errors.raise_exception_on_not_ok_status() as status:
      from tensorflow.python.util import compat
      dirs = dirname.split('/')
      for i in range(len(dirs)):
        partial_dir = '/'.join(dirs[0:i+1])
        if partial_dir and not file_exists(partial_dir):
          CreateDir(compat.as_bytes(partial_dir), status)
%}

%unignoreall
