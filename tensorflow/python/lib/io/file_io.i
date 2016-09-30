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
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_statistics.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
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
  tensorflow::Status status = tensorflow::Env::Default()->GetMatchingPaths(
      filename, &results);
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

void RecursivelyCreateDir(const string& dirname, TF_Status* out_status) {
  tensorflow::Status status = tensorflow::Env::Default()->RecursivelyCreateDir(
      dirname);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }
}

void CopyFile(const string& oldpath, const string& newpath, bool overwrite,
              TF_Status* out_status) {
  // If overwrite is false and the newpath file exists then it's an error.
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

void RenameFile(const string& src, const string& target, bool overwrite,
                TF_Status* out_status) {
  // If overwrite is false and the target file exists then its an error.
  if (!overwrite && FileExists(target)) {
    TF_SetStatus(out_status, TF_ALREADY_EXISTS, "file already exists");
    return;
  }
  tensorflow::Status status = tensorflow::Env::Default()->RenameFile(src,
                                                                     target);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }
}

using tensorflow::int64;

void DeleteRecursively(const string& dirname, TF_Status* out_status) {
  int64 undeleted_files, undeleted_dirs;
  tensorflow::Status status = tensorflow::Env::Default()->DeleteRecursively(
      dirname, &undeleted_files, &undeleted_dirs);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
    return;
  }
  if (undeleted_files > 0 || undeleted_dirs > 0) {
    TF_SetStatus(out_status, TF_PERMISSION_DENIED,
                 "could not fully delete dir");
    return;
  }
}

bool IsDirectory(const string& dirname, TF_Status* out_status) {
  tensorflow::Status status = tensorflow::Env::Default()->IsDirectory(dirname);
  if (status.ok()) {
    return true;
  }
  // FAILED_PRECONDITION Status response means path exists but isn't a dir.
  if (status.code() != tensorflow::error::FAILED_PRECONDITION) {
    Set_TF_Status_from_Status(out_status, status);
  }
  return false;
}

using tensorflow::FileStatistics;

void Stat(const string& filename, FileStatistics* stats,
          TF_Status* out_status) {
  tensorflow::Status status = tensorflow::Env::Default()->Stat(filename,
                                                               stats);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }
}

tensorflow::io::BufferedInputStream* CreateBufferedInputStream(
    const string& filename, size_t buffer_size, TF_Status* out_status) {
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  tensorflow::Status status =
      tensorflow::Env::Default()->NewRandomAccessFile(filename, &file);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
    return nullptr;
  }
  std::unique_ptr<tensorflow::io::RandomAccessInputStream> input_stream(
      new tensorflow::io::RandomAccessInputStream(file.release()));
  std::unique_ptr<tensorflow::io::BufferedInputStream> buffered_input_stream(
      new tensorflow::io::BufferedInputStream(input_stream.release(),
                                              buffer_size));
  return buffered_input_stream.release();
}

tensorflow::WritableFile* CreateWritableFile(
    const string& filename, TF_Status* out_status) {
  std::unique_ptr<tensorflow::WritableFile> file;
  tensorflow::Status status =
      tensorflow::Env::Default()->NewWritableFile(filename, &file);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
    return nullptr;
  }
  return file.release();
}

void AppendToFile(const string& file_content, tensorflow::WritableFile* file,
                  TF_Status* out_status) {
  tensorflow::Status status = file->Append(file_content);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }
}

void FlushWritableFile(tensorflow::WritableFile* file, TF_Status* out_status) {
  tensorflow::Status status = file->Flush();
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }
}

string ReadFromStream(tensorflow::io::BufferedInputStream* stream,
                      size_t bytes,
                      TF_Status* out_status) {
  string result;
  tensorflow::Status status = stream->ReadNBytes(bytes, &result);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
    result.clear();
  }
  return result;
}

void SeekInStream(tensorflow::io::BufferedInputStream* stream, int64 position,
                  TF_Status* out_status) {
  tensorflow::Status status = stream->Seek(position);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }
}

%}

// Ensure that the returned object is destroyed when its wrapper is
// garbage collected.
%newobject CreateBufferedInputStream;
%newobject CreateWritableFile;

// Wrap the above functions.
inline bool FileExists(const string& filename);
inline void DeleteFile(const string& filename, TF_Status* out_status);
string ReadFileToString(const string& filename, TF_Status* out_status);
void WriteStringToFile(const string& filename, const string& file_content,
                       TF_Status* out_status);
std::vector<string> GetMatchingFiles(const string& filename,
                                     TF_Status* out_status);
void CreateDir(const string& dirname, TF_Status* out_status);
void RecursivelyCreateDir(const string& dirname, TF_Status* out_status);
void CopyFile(const string& oldpath, const string& newpath, bool overwrite,
              TF_Status* out_status);
void RenameFile(const string& oldname, const string& newname, bool overwrite,
                TF_Status* out_status);
void DeleteRecursively(const string& dirname, TF_Status* out_status);
bool IsDirectory(const string& dirname, TF_Status* out_status);
void Stat(const string& filename, tensorflow::FileStatistics* stats,
          TF_Status* out_status);
tensorflow::io::BufferedInputStream* CreateBufferedInputStream(
    const string& filename, size_t buffer_size, TF_Status* out_status);
tensorflow::WritableFile* CreateWritableFile(const string& filename,
                                             TF_Status* out_status);
void AppendToFile(const string& file_content, tensorflow::WritableFile* file,
                  TF_Status* out_status);
void FlushWritableFile(tensorflow::WritableFile* file, TF_Status* out_status);
string ReadFromStream(tensorflow::io::BufferedInputStream* stream,
                      size_t bytes,
                      TF_Status* out_status);
void SeekInStream(tensorflow::io::BufferedInputStream* stream, int64 position,
                  TF_Status* out_status);

%ignoreall
%unignore tensorflow::io::BufferedInputStream;
%unignore tensorflow::io::BufferedInputStream::~BufferedInputStream;
%unignore tensorflow::io::BufferedInputStream::ReadLineAsString;
%unignore tensorflow::io::BufferedInputStream::Tell;
%unignore tensorflow::WritableFile;
%unignore tensorflow::WritableFile::~WritableFile;
%include "tensorflow/core/platform/file_system.h"
%include "tensorflow/core/lib/io/inputstream_interface.h"
%include "tensorflow/core/lib/io/buffered_inputstream.h"
%unignoreall

%include "tensorflow/core/lib/io/path.h"
%include "tensorflow/core/platform/file_statistics.h"
