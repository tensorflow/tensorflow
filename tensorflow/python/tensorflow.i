/* SWIG wrapper for all of TensorFlow native functionality.
 * The includes are intentionally not alphabetically sorted, as the order of
 * includes follows dependency order */

%include "tensorflow/python/util/port.i"

%include "tensorflow/python/lib/core/status.i"
%include "tensorflow/python/lib/core/status_helper.i"

%include "tensorflow/python/lib/io/py_record_reader.i"
%include "tensorflow/python/lib/io/py_record_writer.i"
%include "tensorflow/python/client/events_writer.i"

%include "tensorflow/python/client/tf_session.i"
