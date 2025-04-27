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
#include <utility>

#include "absl/status/status.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/lib/core/pybind11_absl.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace {

namespace py = ::pybind11;

class PyRecordReader {
 public:
  // NOTE(sethtroisi): At this time PyRecordReader doesn't benefit from taking
  // RecordReaderOptions, if this changes the API can be updated at that time.
  static absl::Status New(const std::string& filename,
                          const std::string& compression_type,
                          PyRecordReader** out) {
    auto tmp = new PyRecordReader(filename, compression_type);
    TF_RETURN_IF_ERROR(tmp->Reopen());
    *out = tmp;
    return absl::OkStatus();
  }

  PyRecordReader() = delete;
  ~PyRecordReader() { Close(); }

  absl::Status ReadNextRecord(tensorflow::tstring* out) {
    if (IsClosed()) {
      return tensorflow::errors::FailedPrecondition("Reader is closed.");
    }
    return reader_->ReadRecord(&offset_, out);
  }

  bool IsClosed() const { return file_ == nullptr && reader_ == nullptr; }

  void Close() {
    reader_ = nullptr;
    file_ = nullptr;
  }

  // Reopen a closed writer by re-opening the file and re-creating the reader,
  // but preserving the prior read offset. If not closed, returns an error.
  //
  // This is useful to allow "refreshing" the underlying file handle, in cases
  // where the file was replaced with a newer version containing additional data
  // that otherwise wouldn't be available via the existing file handle. This
  // allows the file to be polled continuously using the same iterator, even as
  // it grows, which supports use cases such as TensorBoard.
  absl::Status Reopen() {
    if (!IsClosed()) {
      return tensorflow::errors::FailedPrecondition("Reader is not closed.");
    }
    TF_RETURN_IF_ERROR(
        tensorflow::Env::Default()->NewRandomAccessFile(filename_, &file_));
    reader_ =
        std::make_unique<tensorflow::io::RecordReader>(file_.get(), options_);
    return absl::OkStatus();
  }

 private:
  static constexpr tensorflow::uint64 kReaderBufferSize = 16 * 1024 * 1024;

  PyRecordReader(const std::string& filename,
                 const std::string& compression_type)
      : filename_(filename),
        options_(CreateOptions(compression_type)),
        offset_(0),
        file_(nullptr),
        reader_(nullptr) {}

  static tensorflow::io::RecordReaderOptions CreateOptions(
      const std::string& compression_type) {
    auto options =
        tensorflow::io::RecordReaderOptions::CreateRecordReaderOptions(
            compression_type);
    options.buffer_size = kReaderBufferSize;
    return options;
  }

  const std::string filename_;
  const tensorflow::io::RecordReaderOptions options_;
  tensorflow::uint64 offset_;
  std::unique_ptr<tensorflow::RandomAccessFile> file_;
  std::unique_ptr<tensorflow::io::RecordReader> reader_;

  PyRecordReader(const PyRecordReader&) = delete;
  void operator=(const PyRecordReader&) = delete;
};

class PyRecordRandomReader {
 public:
  static absl::Status New(const std::string& filename,
                          PyRecordRandomReader** out) {
    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(
        tensorflow::Env::Default()->NewRandomAccessFile(filename, &file));
    auto options =
        tensorflow::io::RecordReaderOptions::CreateRecordReaderOptions("");
    options.buffer_size = kReaderBufferSize;
    auto reader =
        std::make_unique<tensorflow::io::RecordReader>(file.get(), options);
    *out = new PyRecordRandomReader(std::move(file), std::move(reader));
    return absl::OkStatus();
  }

  PyRecordRandomReader() = delete;
  ~PyRecordRandomReader() { Close(); }

  absl::Status ReadRecord(tensorflow::uint64* offset,
                          tensorflow::tstring* out) {
    if (IsClosed()) {
      return tensorflow::errors::FailedPrecondition(
          "Random TFRecord Reader is closed.");
    }
    return reader_->ReadRecord(offset, out);
  }

  bool IsClosed() const { return file_ == nullptr && reader_ == nullptr; }

  void Close() {
    reader_ = nullptr;
    file_ = nullptr;
  }

 private:
  static constexpr tensorflow::uint64 kReaderBufferSize = 16 * 1024 * 1024;

  PyRecordRandomReader(std::unique_ptr<tensorflow::RandomAccessFile> file,
                       std::unique_ptr<tensorflow::io::RecordReader> reader)
      : file_(std::move(file)), reader_(std::move(reader)) {}

  std::unique_ptr<tensorflow::RandomAccessFile> file_;
  std::unique_ptr<tensorflow::io::RecordReader> reader_;

  PyRecordRandomReader(const PyRecordRandomReader&) = delete;
  void operator=(const PyRecordRandomReader&) = delete;
};

class PyRecordWriter {
 public:
  static absl::Status New(const std::string& filename,
                          const tensorflow::io::RecordWriterOptions& options,
                          PyRecordWriter** out) {
    std::unique_ptr<tensorflow::WritableFile> file;
    TF_RETURN_IF_ERROR(
        tensorflow::Env::Default()->NewWritableFile(filename, &file));
    auto writer =
        std::make_unique<tensorflow::io::RecordWriter>(file.get(), options);
    *out = new PyRecordWriter(std::move(file), std::move(writer));
    return absl::OkStatus();
  }

  PyRecordWriter() = delete;
  ~PyRecordWriter() { (void)Close(); }

  absl::Status WriteRecord(absl::string_view record) {
    if (IsClosed()) {
      return tensorflow::errors::FailedPrecondition("Writer is closed.");
    }
    return writer_->WriteRecord(record);
  }

  absl::Status Flush() {
    if (IsClosed()) {
      return tensorflow::errors::FailedPrecondition("Writer is closed.");
    }

    auto status = writer_->Flush();
    if (status.ok()) {
      // Per the RecordWriter contract, flushing the RecordWriter does not
      // flush the underlying file.  Here we need to do both.
      return file_->Flush();
    }
    return status;
  }

  bool IsClosed() const { return file_ == nullptr && writer_ == nullptr; }

  absl::Status Close() {
    if (writer_ != nullptr) {
      auto status = writer_->Close();
      writer_ = nullptr;
      if (!status.ok()) return status;
    }
    if (file_ != nullptr) {
      auto status = file_->Close();
      file_ = nullptr;
      if (!status.ok()) return status;
    }
    return absl::OkStatus();
  }

 private:
  PyRecordWriter(std::unique_ptr<tensorflow::WritableFile> file,
                 std::unique_ptr<tensorflow::io::RecordWriter> writer)
      : file_(std::move(file)), writer_(std::move(writer)) {}

  std::unique_ptr<tensorflow::WritableFile> file_;
  std::unique_ptr<tensorflow::io::RecordWriter> writer_;

  PyRecordWriter(const PyRecordWriter&) = delete;
  void operator=(const PyRecordWriter&) = delete;
};

PYBIND11_MODULE(_pywrap_record_io, m) {
  py::class_<PyRecordReader>(m, "RecordIterator")
      .def(py::init(
          [](const std::string& filename, const std::string& compression_type) {
            absl::Status status;
            PyRecordReader* self = nullptr;
            {
              py::gil_scoped_release release;
              status = PyRecordReader::New(filename, compression_type, &self);
            }
            tsl::MaybeRaiseRegisteredFromStatus(status);
            return self;
          }))
      .def("__iter__", [](const py::object& self) { return self; })
      .def("__next__",
           [](PyRecordReader* self) {
             if (self->IsClosed()) {
               throw py::stop_iteration();
             }

             tensorflow::tstring record;
             absl::Status status;
             {
               py::gil_scoped_release release;
               status = self->ReadNextRecord(&record);
             }
             if (absl::IsOutOfRange(status)) {
               // Don't close because the file being read could be updated
               // in-between
               // __next__ calls.
               throw py::stop_iteration();
             }
             tsl::MaybeRaiseRegisteredFromStatus(status);
             return py::bytes(record);
           })
      .def("close", [](PyRecordReader* self) { self->Close(); })
      .def("reopen", [](PyRecordReader* self) {
        absl::Status status;
        {
          py::gil_scoped_release release;
          status = self->Reopen();
        }
        tsl::MaybeRaiseRegisteredFromStatus(status);
      });

  py::class_<PyRecordRandomReader>(m, "RandomRecordReader")
      .def(py::init([](const std::string& filename) {
        absl::Status status;
        PyRecordRandomReader* self = nullptr;
        {
          py::gil_scoped_release release;
          status = PyRecordRandomReader::New(filename, &self);
        }
        tsl::MaybeRaiseRegisteredFromStatus(status);
        return self;
      }))
      .def("read",
           [](PyRecordRandomReader* self, tensorflow::uint64 offset) {
             tensorflow::uint64 temp_offset = offset;
             tensorflow::tstring record;
             absl::Status status;
             {
               py::gil_scoped_release release;
               status = self->ReadRecord(&temp_offset, &record);
             }
             if (absl::IsOutOfRange(status)) {
               throw py::index_error(tensorflow::strings::StrCat(
                   "Out of range at reading offset ", offset));
             }
             tsl::MaybeRaiseRegisteredFromStatus(status);
             return py::make_tuple(py::bytes(record), temp_offset);
           })
      .def("close", [](PyRecordRandomReader* self) { self->Close(); });

  using tensorflow::io::ZlibCompressionOptions;
  py::class_<ZlibCompressionOptions>(m, "ZlibCompressionOptions")
      .def_readwrite("flush_mode", &ZlibCompressionOptions::flush_mode)
      .def_readwrite("input_buffer_size",
                     &ZlibCompressionOptions::input_buffer_size)
      .def_readwrite("output_buffer_size",
                     &ZlibCompressionOptions::output_buffer_size)
      .def_readwrite("window_bits", &ZlibCompressionOptions::window_bits)
      .def_readwrite("compression_level",
                     &ZlibCompressionOptions::compression_level)
      .def_readwrite("compression_method",
                     &ZlibCompressionOptions::compression_method)
      .def_readwrite("mem_level", &ZlibCompressionOptions::mem_level)
      .def_readwrite("compression_strategy",
                     &ZlibCompressionOptions::compression_strategy);

  using tensorflow::io::RecordWriterOptions;
  py::class_<RecordWriterOptions>(m, "RecordWriterOptions")
      .def(py::init(&RecordWriterOptions::CreateRecordWriterOptions))
      .def_readonly("compression_type", &RecordWriterOptions::compression_type)
      .def_readonly("zlib_options", &RecordWriterOptions::zlib_options);

  using tensorflow::MaybeRaiseRegisteredFromStatus;

  py::class_<PyRecordWriter>(m, "RecordWriter")
      .def(py::init(
          [](const std::string& filename, const RecordWriterOptions& options) {
            PyRecordWriter* self = nullptr;
            absl::Status status;
            {
              py::gil_scoped_release release;
              status = PyRecordWriter::New(filename, options, &self);
            }
            tsl::MaybeRaiseRegisteredFromStatus(status);
            return self;
          }))
      .def("__enter__", [](const py::object& self) { return self; })
      .def("__exit__",
           [](PyRecordWriter* self, py::args) {
             tsl::MaybeRaiseRegisteredFromStatus(self->Close());
           })
      .def(
          "write",
          [](PyRecordWriter* self, absl::string_view record) {
            absl::Status status;
            {
              py::gil_scoped_release release;
              status = self->WriteRecord(record);
            }
            tsl::MaybeRaiseRegisteredFromStatus(status);
          },
          py::arg("record"))
      .def("flush",
           [](PyRecordWriter* self) {
             tsl::MaybeRaiseRegisteredFromStatus(self->Flush());
           })
      .def("close", [](PyRecordWriter* self) {
        tsl::MaybeRaiseRegisteredFromStatus(self->Close());
      });
}

}  // namespace
