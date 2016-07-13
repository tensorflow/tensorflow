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
  if (!status.ok() && status.code() != tensorflow::error::ALREADY_EXISTS) {
    Set_TF_Status_from_Status(out_status, status);
  }
}

void CopyFile(const string& oldpath, const string& newpath, bool overwrite,
              TF_Status* out_status) {
  // If overwrite is false and the newpath file exists then its an error.
  if (!overwrite && FileExists(newpath)) {
    TF_SetStatus(out_status, TF_ALREADY_EXISTS, "file already exists");
    return;
  }
  string file_content;
  tensorflow::Status status = ReadFileToString(tensorflow::Env::Default(),
      oldpath, &file_content);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
    return;
  }
  status = WriteStringToFile(tensorflow::Env::Default(), newpath, file_content);
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
void CopyFile(const string& oldpath, const string& newpath, bool overwrite,
              TF_Status* out_status);
