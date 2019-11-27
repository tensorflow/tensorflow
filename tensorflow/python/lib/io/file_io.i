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
inline void FileExists(const string& filename, TF_Status* status) {
  tensorflow::Status s = tensorflow::Env::Default()->FileExists(filename);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
  }
}

inline void FileExists(const tensorflow::StringPiece& filename,
    TF_Status* status) {
  tensorflow::Status s =
      tensorflow::Env::Default()->FileExists(string(filename));
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
  }
}

inline void DeleteFile(const string& filename, TF_Status* status) {
  tensorflow::Status s = tensorflow::Env::Default()->DeleteFile(filename);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
  }
}

string ReadFileToString(const string& filename, TF_Status* status) {
  string file_content;
  tensorflow::Status s = ReadFileToString(tensorflow::Env::Default(),
      filename, &file_content);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
  }
  return file_content;
}

void WriteStringToFile(const string& filename, const string& file_content,
                       TF_Status* status) {
  tensorflow::Status s = WriteStringToFile(tensorflow::Env::Default(),
      filename, file_content);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
  }
}

std::vector<string> GetChildren(const string& dir, TF_Status* status) {
  std::vector<string> results;
  tensorflow::Status s = tensorflow::Env::Default()->GetChildren(
      dir, &results);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
  }
  return results;
}

std::vector<string> GetMatchingFiles(const string& filename, TF_Status* status) {
  std::vector<string> results;
  tensorflow::Status s = tensorflow::Env::Default()->GetMatchingPaths(
      filename, &results);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
  }
  return results;
}

void CreateDir(const string& dirname, TF_Status* status) {
  tensorflow::Status s = tensorflow::Env::Default()->CreateDir(dirname);
  if (!s.ok() && s.code() != tensorflow::error::ALREADY_EXISTS) {
    Set_TF_Status_from_Status(status, s);
  }
}

void RecursivelyCreateDir(const string& dirname, TF_Status* status) {
  tensorflow::Status s = tensorflow::Env::Default()->RecursivelyCreateDir(
      dirname);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
  }
}

void CopyFile(const string& src, const string& target, bool overwrite,
              TF_Status* status) {
  // If overwrite is false and the target file exists then its an error.
  if (!overwrite && tensorflow::Env::Default()->FileExists(target).ok()) {
    TF_SetStatus(status, TF_ALREADY_EXISTS, "file already exists");
    return;
  }
  tensorflow::Status s = tensorflow::Env::Default()->CopyFile(src, target);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
  }
}

void RenameFile(const string& src, const string& target, bool overwrite,
                TF_Status* status) {
  // If overwrite is false and the target file exists then its an error.
  if (!overwrite && tensorflow::Env::Default()->FileExists(target).ok()) {
    TF_SetStatus(status, TF_ALREADY_EXISTS, "file already exists");
    return;
  }
  tensorflow::Status s = tensorflow::Env::Default()->RenameFile(src, target);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
  }
}

using tensorflow::int64;

void DeleteRecursively(const string& dirname, TF_Status* status) {
  int64 undeleted_files, undeleted_dirs;
  tensorflow::Status s = tensorflow::Env::Default()->DeleteRecursively(
      dirname, &undeleted_files, &undeleted_dirs);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
    return;
  }
  if (undeleted_files > 0 || undeleted_dirs > 0) {
    TF_SetStatus(status, TF_PERMISSION_DENIED, "could not fully delete dir");
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

void Stat(const string& filename, FileStatistics* stats, TF_Status* status) {
  tensorflow::Status s = tensorflow::Env::Default()->Stat(filename,
                                                               stats);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
  }
}

tensorflow::io::BufferedInputStream* CreateBufferedInputStream(
    const string& filename, size_t buffer_size, TF_Status* status) {
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  tensorflow::Status s =
      tensorflow::Env::Default()->NewRandomAccessFile(filename, &file);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  std::unique_ptr<tensorflow::io::RandomAccessInputStream> input_stream(
      new tensorflow::io::RandomAccessInputStream(
          file.release(), true /* owns_file */));
  std::unique_ptr<tensorflow::io::BufferedInputStream> buffered_input_stream(
      new tensorflow::io::BufferedInputStream(
          input_stream.release(), buffer_size, true /* owns_input_stream */));
  return buffered_input_stream.release();
}

tensorflow::WritableFile* CreateWritableFile(
    const string& filename, const string& mode, TF_Status* status) {
  std::unique_ptr<tensorflow::WritableFile> file;
  tensorflow::Status s;
  if (mode.find("a") != std::string::npos) {
    s = tensorflow::Env::Default()->NewAppendableFile(filename, &file);
  } else {
    s = tensorflow::Env::Default()->NewWritableFile(filename, &file);
  }
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  return file.release();
}

void AppendToFile(const string& file_content, tensorflow::WritableFile* file,
                  TF_Status* status) {
  tensorflow::Status s = file->Append(file_content);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
  }
}

int64 TellFile(tensorflow::WritableFile* file, TF_Status* status) {
  int64 position = -1;
  tensorflow::Status s = file->Tell(&position);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
  }
  return position;
}


string ReadFromStream(tensorflow::io::BufferedInputStream* stream,
                      size_t bytes,
                      TF_Status* status) {
  tensorflow::tstring result;
  tensorflow::Status s = stream->ReadNBytes(bytes, &result);
  if (!s.ok() && s.code() != tensorflow::error::OUT_OF_RANGE) {
    Set_TF_Status_from_Status(status, s);
    result.clear();
  }
  return result;
}

%}

// Ensure that the returned object is destroyed when its wrapper is
// garbage collected.
%newobject CreateBufferedInputStream;
%newobject CreateWritableFile;

// Wrap the above functions.
inline void FileExists(const string& filename, TF_Status* status);
inline void DeleteFile(const string& filename, TF_Status* status);
string ReadFileToString(const string& filename, TF_Status* status);
void WriteStringToFile(const string& filename, const string& file_content,
                       TF_Status* status);
std::vector<string> GetChildren(const string& dir, TF_Status* status);
std::vector<string> GetMatchingFiles(const string& filename,
                                     TF_Status* status);
void CreateDir(const string& dirname, TF_Status* status);
void RecursivelyCreateDir(const string& dirname, TF_Status* status);
void CopyFile(const string& oldpath, const string& newpath, bool overwrite,
              TF_Status* status);
void RenameFile(const string& oldname, const string& newname, bool overwrite,
                TF_Status* status);
void DeleteRecursively(const string& dirname, TF_Status* status);
bool IsDirectory(const string& dirname, TF_Status* out_status);
void Stat(const string& filename, tensorflow::FileStatistics* stats,
          TF_Status* status);
tensorflow::io::BufferedInputStream* CreateBufferedInputStream(
    const string& filename, size_t buffer_size, TF_Status* status);
tensorflow::WritableFile* CreateWritableFile(const string& filename,
                                             const string& mode,
                                             TF_Status* status);
void AppendToFile(const string& file_content, tensorflow::WritableFile* file,
                  TF_Status* status);
int64 TellFile(tensorflow::WritableFile* file, TF_Status* status);
string ReadFromStream(tensorflow::io::BufferedInputStream* stream,
                      size_t bytes,
                      TF_Status* status);

%ignore tensorflow::Status::operator=;
%include "tensorflow/core/platform/status.h"

%ignoreall
%unignore tensorflow::io;
%unignore tensorflow::io::BufferedInputStream;
%unignore tensorflow::io::BufferedInputStream::~BufferedInputStream;
%unignore tensorflow::io::BufferedInputStream::ReadLineAsString;
%unignore tensorflow::io::BufferedInputStream::Seek;
%unignore tensorflow::io::BufferedInputStream::Tell;
%unignore tensorflow::WritableFile;
%unignore tensorflow::WritableFile::Close;
%unignore tensorflow::WritableFile::Flush;
%unignore tensorflow::WritableFile::~WritableFile;
%include "tensorflow/core/platform/file_system.h"
%include "tensorflow/core/lib/io/inputstream_interface.h"
%include "tensorflow/core/lib/io/buffered_inputstream.h"
%unignoreall

%include "tensorflow/c/tf_status_helper.h"

%ignore tensorflow::io::internal::JoinPathImpl;
%include "tensorflow/core/lib/io/path.h"

%include "tensorflow/core/platform/file_statistics.h"
