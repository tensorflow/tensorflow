%nothread tensorflow::io::PyRecordWriter::WriteRecord;

%include "tensorflow/python/platform/base.i"
%include "tensorflow/python/lib/core/strings.i"

%feature("except") tensorflow::io::PyRecordWriter::New {
  // Let other threads run while we write
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}

%newobject tensorflow::io::PyRecordWriter::New;

%feature("except") tensorflow::io::PyRecordWriter::WriteRecord {
  // Let other threads run while we write
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}

%{
#include "tensorflow/python/lib/io/py_record_writer.h"
%}

%ignoreall

%unignore tensorflow;
%unignore tensorflow::io;
%unignore tensorflow::io::PyRecordWriter;
%unignore tensorflow::io::PyRecordWriter::~PyRecordWriter;
%unignore tensorflow::io::PyRecordWriter::WriteRecord;
%unignore tensorflow::io::PyRecordWriter::Close;
%unignore tensorflow::io::PyRecordWriter::New;

%include "tensorflow/python/lib/io/py_record_writer.h"

%unignoreall
