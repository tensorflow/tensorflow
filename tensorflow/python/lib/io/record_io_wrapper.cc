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

#include "absl/memory/memory.h"
#include "include/pybind11/pybind11.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/platform/env.h"
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
  static tensorflow::Status New(const std::string& filename,
                                const std::string& compression_type,
                                PyRecordReader** out) {
    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(
        tensorflow::Env::Default()->NewRandomAccessFile(filename, &file));
    auto options =
        tensorflow::io::RecordReaderOptions::CreateRecordReaderOptions(
            compression_type);
    options.buffer_size = kReaderBufferSize;
    auto reader =
        absl::make_unique<tensorflow::io::RecordReader>(file.get(), options);
    *out = new PyRecordReader(std::move(file), std::move(reader));
    return tensorflow::Status::OK();
  }

  PyRecordReader() = delete;
  ~PyRecordReader() { Close(); }

  tensorflow::Status ReadNextRecord(tensorflow::tstring* out) {
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

 private:
  static constexpr tensorflow::uint64 kReaderBufferSize = 16 * 1024 * 1024;

  PyRecordReader(std::unique_ptr<tensorflow::RandomAccessFile> file,
                 std::unique_ptr<tensorflow::io::RecordReader> reader)
      : file_(std::move(file)), reader_(std::move(reader)) {
    offset_ = 0;
  }

  tensorflow::uint64 offset_;
  std::unique_ptr<tensorflow::RandomAccessFile> file_;
  std::unique_ptr<tensorflow::io::RecordReader> reader_;

  TF_DISALLOW_COPY_AND_ASSIGN(PyRecordReader);
};

class PyRecordWriter {
 public:
  static tensorflow::Status New(
      const std::string& filename,
      const tensorflow::io::RecordWriterOptions& options,
      PyRecordWriter** out) {
    std::unique_ptr<tensorflow::WritableFile> file;
    TF_RETURN_IF_ERROR(
        tensorflow::Env::Default()->NewWritableFile(filename, &file));
    auto writer =
        absl::make_unique<tensorflow::io::RecordWriter>(file.get(), options);
    *out = new PyRecordWriter(std::move(file), std::move(writer));
    return tensorflow::Status::OK();
  }

  PyRecordWriter() = delete;
  ~PyRecordWriter() { Close(); }

  tensorflow::Status WriteRecord(tensorflow::StringPiece record) {
    if (IsClosed()) {
      return tensorflow::errors::FailedPrecondition("Writer is closed.");
    }
    return writer_->WriteRecord(record);
  }

  tensorflow::Status Flush() {
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

  tensorflow::Status Close() {
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
    return tensorflow::Status::OK();
  }

 private:
  PyRecordWriter(std::unique_ptr<tensorflow::WritableFile> file,
                 std::unique_ptr<tensorflow::io::RecordWriter> writer)
      : file_(std::move(file)), writer_(std::move(writer)) {}

  std::unique_ptr<tensorflow::WritableFile> file_;
  std::unique_ptr<tensorflow::io::RecordWriter> writer_;

  TF_DISALLOW_COPY_AND_ASSIGN(PyRecordWriter);
};

PYBIND11_MODULE(_pywrap_record_io, m) {
  py::class_<PyRecordReader>(m, "RecordIterator")
      .def(py::init(
          [](const std::string& filename, const std::string& compression_type) {
            tensorflow::Status status;
            PyRecordReader* self = nullptr;
            {
              py::gil_scoped_release release;
              status = PyRecordReader::New(filename, compression_type, &self);
            }
            MaybeRaiseRegisteredFromStatus(status);
            return self;
          }))
      .def("__iter__", [](const py::object& self) { return self; })
      .def("__next__",
           [](PyRecordReader* self) {
             if (self->IsClosed()) {
               throw py::stop_iteration();
             }

             tensorflow::tstring record;
             tensorflow::Status status;
             {
               py::gil_scoped_release release;
               status = self->ReadNextRecord(&record);
             }
             if (tensorflow::errors::IsOutOfRange(status)) {
               // Don't close because the file being read could be updated
               // in-between
               // __next__ calls.
               throw py::stop_iteration();
             }
             MaybeRaiseRegisteredFromStatus(status);
             return py::bytes(record);
           })
      .def("close", [](PyRecordReader* self) { self->Close(); });

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
            tensorflow::Status status;
            {
              py::gil_scoped_release release;
              status = PyRecordWriter::New(filename, options, &self);
            }
            MaybeRaiseRegisteredFromStatus(status);
            return self;
          }))
      .def("__enter__", [](const py::object& self) { return self; })
      .def("__exit__",
           [](PyRecordWriter* self, py::args) {
             MaybeRaiseRegisteredFromStatus(self->Close());
           })
      .def(
          "write",
          [](PyRecordWriter* self, tensorflow::StringPiece record) {
            tensorflow::Status status;
            {
              py::gil_scoped_release release;
              status = self->WriteRecord(record);
            }
            MaybeRaiseRegisteredFromStatus(status);
          },
          py::arg("record"))
      .def("flush",
           [](PyRecordWriter* self) {
             MaybeRaiseRegisteredFromStatus(self->Flush());
           })
      .def("close", [](PyRecordWriter* self) {
        MaybeRaiseRegisteredFromStatus(self->Close());
      });
}

}  // namespace
