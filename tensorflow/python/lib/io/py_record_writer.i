/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

%nothread tensorflow::io::PyRecordWriter::WriteRecord;

%include "tensorflow/python/platform/base.i"
%include "tensorflow/python/lib/core/strings.i"

// Define int8_t explicitly instead of including "stdint.i", since "stdint.h"
// and "stdint.i" disagree on the definition of int64_t.
typedef signed char int8;
%{ typedef signed char int8; %}

%feature("except") tensorflow::io::PyRecordWriter::New {
  // Let other threads run while we write
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}

%newobject tensorflow::io::PyRecordWriter::New;
%newobject tensorflow::io::RecordWriterOptions::CreateRecordWriterOptions;

%feature("except") tensorflow::io::PyRecordWriter::WriteRecord {
  // Let other threads run while we write
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}

%{
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/python/lib/io/py_record_writer.h"
%}

%ignoreall

%unignore tensorflow;
%unignore tensorflow::io;
%unignore tensorflow::io::PyRecordWriter;
%unignore tensorflow::io::PyRecordWriter::~PyRecordWriter;
%unignore tensorflow::io::PyRecordWriter::WriteRecord;
%unignore tensorflow::io::PyRecordWriter::Flush;
%unignore tensorflow::io::PyRecordWriter::Close;
%unignore tensorflow::io::PyRecordWriter::New;
%unignore tensorflow::io::ZlibCompressionOptions;
%unignore tensorflow::io::ZlibCompressionOptions::flush_mode;
%unignore tensorflow::io::ZlibCompressionOptions::input_buffer_size;
%unignore tensorflow::io::ZlibCompressionOptions::output_buffer_size;
%unignore tensorflow::io::ZlibCompressionOptions::window_bits;
%unignore tensorflow::io::ZlibCompressionOptions::compression_level;
%unignore tensorflow::io::ZlibCompressionOptions::compression_method;
%unignore tensorflow::io::ZlibCompressionOptions::mem_level;
%unignore tensorflow::io::ZlibCompressionOptions::compression_strategy;
%unignore tensorflow::io::RecordWriterOptions;
%unignore tensorflow::io::RecordWriterOptions::CreateRecordWriterOptions;
%unignore tensorflow::io::RecordWriterOptions::zlib_options;

%include "tensorflow/core/lib/io/record_writer.h"
%include "tensorflow/core/lib/io/zlib_compression_options.h"
%include "tensorflow/python/lib/io/py_record_writer.h"

%unignoreall
