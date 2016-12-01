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

%include "tensorflow/python/lib/core/strings.i"
%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/core/util/events_writer.h"
#include "tensorflow/core/util/event.pb.h"
%}

%nodefaultctor EventsWriter;

%ignoreall
%unignore tensorflow;
%unignore tensorflow::EventsWriter;
%unignore tensorflow::EventsWriter::EventsWriter;
%unignore tensorflow::EventsWriter::~EventsWriter;
%unignore tensorflow::EventsWriter::FileName;
%rename("_WriteSerializedEvent") tensorflow::EventsWriter::WriteSerializedEvent;
%unignore tensorflow::EventsWriter::Flush;
%unignore tensorflow::EventsWriter::Close;
%include "tensorflow/core/util/events_writer.h"
%unignoreall

%newobject tensorflow::EventsWriter::EventsWriter;


%extend tensorflow::EventsWriter {
%insert("python") %{
  def WriteEvent(self, event):
    from tensorflow.core.util.event_pb2 import Event
    if not isinstance(event, Event):
      raise TypeError("Expected an event_pb2.Event proto, "
                      " but got %s" % type(event))
    return self._WriteSerializedEvent(event.SerializeToString())
%}
}
