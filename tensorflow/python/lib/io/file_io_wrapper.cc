/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <vector>

#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_statistics.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/python/lib/core/pybind11_absl.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace {
namespace py = pybind11;

PYBIND11_MODULE(_pywrap_file_io, m) {
  m.def("FileExists", [](const std::string& filename) {
    tensorflow::Status status;
    {
      py::gil_scoped_release release;
      status = tensorflow::Env::Default()->FileExists(filename);
    }
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
  });
  m.def("DeleteFile", [](const std::string& filename) {
    tensorflow::MaybeRaiseRegisteredFromStatus(
        tensorflow::Env::Default()->DeleteFile(filename));
  });
  m.def("ReadFileToString", [](const std::string& filename) {
    std::string data;
    const auto status =
        ReadFileToString(tensorflow::Env::Default(), filename, &data);
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
    return py::bytes(data);
  });
  m.def("WriteStringToFile",
        [](const std::string& filename, tensorflow::StringPiece data) {
          return WriteStringToFile(tensorflow::Env::Default(), filename, data);
        });
  m.def("GetChildren", [](const std::string& dirname) {
    std::vector<std::string> results;
    const auto status =
        tensorflow::Env::Default()->GetChildren(dirname, &results);
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
    return results;
  });
  m.def("GetMatchingFiles", [](const std::string& pattern) {
    std::vector<std::string> results;
    const auto status =
        tensorflow::Env::Default()->GetMatchingPaths(pattern, &results);
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
    return results;
  });
  m.def("CreateDir", [](const std::string& dirname) {
    const auto status = tensorflow::Env::Default()->CreateDir(dirname);
    if (tensorflow::errors::IsAlreadyExists(status)) {
      return;
    }
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
  });
  m.def("RecursivelyCreateDir", [](const std::string& dirname) {
    tensorflow::MaybeRaiseRegisteredFromStatus(
        tensorflow::Env::Default()->RecursivelyCreateDir(dirname));
  });
  m.def("CopyFile",
        [](const std::string& src, const std::string& target, bool overwrite) {
          auto* env = tensorflow::Env::Default();
          tensorflow::Status status;
          if (!overwrite && env->FileExists(target).ok()) {
            status = tensorflow::errors::AlreadyExists("file already exists");
          } else {
            status = env->CopyFile(src, target);
          }
          tensorflow::MaybeRaiseRegisteredFromStatus(status);
        });
  m.def("RenameFile",
        [](const std::string& src, const std::string& target, bool overwrite) {
          auto* env = tensorflow::Env::Default();
          tensorflow::Status status;
          if (!overwrite && env->FileExists(target).ok()) {
            status = tensorflow::errors::AlreadyExists("file already exists");
          } else {
            status = env->RenameFile(src, target);
          }
          tensorflow::MaybeRaiseRegisteredFromStatus(status);
        });
  m.def("DeleteRecursively", [](const std::string& dirname) {
    tensorflow::int64 undeleted_files;
    tensorflow::int64 undeleted_dirs;
    auto status = tensorflow::Env::Default()->DeleteRecursively(
        dirname, &undeleted_files, &undeleted_dirs);
    if (status.ok() && (undeleted_files > 0 || undeleted_dirs > 0)) {
      status =
          tensorflow::errors::PermissionDenied("could not fully delete dir");
    }
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
  });
  m.def("IsDirectory", [](const std::string& dirname) {
    const auto status = tensorflow::Env::Default()->IsDirectory(dirname);
    // FAILED_PRECONDITION response means path exists but isn't a dir.
    if (tensorflow::errors::IsFailedPrecondition(status)) {
      return false;
    }

    tensorflow::MaybeRaiseRegisteredFromStatus(status);
    return true;
  });

  py::class_<tensorflow::FileStatistics>(m, "FileStatistics")
      .def_readonly("length", &tensorflow::FileStatistics::length)
      .def_readonly("mtime_nsec", &tensorflow::FileStatistics::mtime_nsec)
      .def_readonly("is_directory", &tensorflow::FileStatistics::is_directory);

  m.def("Stat", [](const std::string& filename) {
    std::unique_ptr<tensorflow::FileStatistics> self(
        new tensorflow::FileStatistics);
    const auto status = tensorflow::Env::Default()->Stat(filename, self.get());
    tensorflow::MaybeRaiseRegisteredFromStatus(status);
    return self.release();
  });

  using tensorflow::WritableFile;
  py::class_<WritableFile>(m, "WritableFile")
      .def(py::init([](const std::string& filename, const std::string& mode) {
        auto* env = tensorflow::Env::Default();
        std::unique_ptr<WritableFile> self;
        const auto status = mode.find("a") == std::string::npos
                                ? env->NewWritableFile(filename, &self)
                                : env->NewAppendableFile(filename, &self);
        tensorflow::MaybeRaiseRegisteredFromStatus(status);
        return self.release();
      }))
      .def("append",
           [](WritableFile* self, tensorflow::StringPiece data) {
             tensorflow::MaybeRaiseRegisteredFromStatus(self->Append(data));
           })
      // TODO(slebedev): Make WritableFile::Tell const and change self
      // to be a reference.
      .def("tell",
           [](WritableFile* self) {
             tensorflow::int64 pos = -1;
             const auto status = self->Tell(&pos);
             tensorflow::MaybeRaiseRegisteredFromStatus(status);
             return pos;
           })
      .def("flush",
           [](WritableFile* self) {
             tensorflow::MaybeRaiseRegisteredFromStatus(self->Flush());
           })
      .def("close", [](WritableFile* self) {
        tensorflow::MaybeRaiseRegisteredFromStatus(self->Close());
      });

  using tensorflow::io::BufferedInputStream;
  py::class_<BufferedInputStream>(m, "BufferedInputStream")
      .def(py::init([](const std::string& filename, size_t buffer_size) {
        std::unique_ptr<tensorflow::RandomAccessFile> file;
        const auto status =
            tensorflow::Env::Default()->NewRandomAccessFile(filename, &file);
        tensorflow::MaybeRaiseRegisteredFromStatus(status);
        std::unique_ptr<tensorflow::io::RandomAccessInputStream> input_stream(
            new tensorflow::io::RandomAccessInputStream(file.release(),
                                                        /*owns_file=*/true));
        return new BufferedInputStream(input_stream.release(), buffer_size,
                                       /*owns_input_stream=*/true);
      }))
      .def("read",
           [](BufferedInputStream* self, tensorflow::int64 bytes_to_read) {
             tensorflow::tstring result;
             const auto status = self->ReadNBytes(bytes_to_read, &result);
             if (!status.ok() && !tensorflow::errors::IsOutOfRange(status)) {
               result.clear();
               tensorflow::MaybeRaiseRegisteredFromStatus(status);
             }
             return py::bytes(result);
           })
      .def("readline",
           [](BufferedInputStream* self) {
             return py::bytes(self->ReadLineAsString());
           })
      .def("seek",
           [](BufferedInputStream* self, tensorflow::int64 pos) {
             tensorflow::MaybeRaiseRegisteredFromStatus(self->Seek(pos));
           })
      .def("tell", [](BufferedInputStream* self) { return self->Tell(); });
}
}  // namespace
