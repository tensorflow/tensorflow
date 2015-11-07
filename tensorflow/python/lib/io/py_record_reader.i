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
