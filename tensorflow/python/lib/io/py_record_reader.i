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

%nothread tensorflow::io::PyRecordReader::GetNext;

%include "tensorflow/python/platform/base.i"

%feature("except") tensorflow::io::PyRecordReader::New {
  // Let other threads run while we read
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}

%newobject tensorflow::io::PyRecordReader::New;

%feature("except") tensorflow::io::PyRecordReader::GetNext {
  // Let other threads run while we read
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}

%{
#include "tensorflow/python/lib/io/py_record_reader.h"
%}

%ignoreall

%unignore tensorflow;
%unignore tensorflow::io;
%unignore tensorflow::io::PyRecordReader;
%unignore tensorflow::io::PyRecordReader::~PyRecordReader;
%unignore tensorflow::io::PyRecordReader::GetNext;
%unignore tensorflow::io::PyRecordReader::offset;
%unignore tensorflow::io::PyRecordReader::record;
%unignore tensorflow::io::PyRecordReader::Close;
%unignore tensorflow::io::PyRecordReader::New;

%include "tensorflow/python/lib/io/py_record_reader.h"

%unignoreall
