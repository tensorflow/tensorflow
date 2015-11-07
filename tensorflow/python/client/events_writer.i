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
