// SWIG test helper for lib::tensorflow::Status

%include "tensorflow/python/platform/base.i"
%import(module="tensorflow.python.pywrap_tensorflow") "tensorflow/python/lib/core/status.i"

%inline %{
#include "tensorflow/core/public/status.h"

tensorflow::Status NotOkay() {
  return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT, "Testing 1 2 3");
}

tensorflow::Status Okay() {
  return tensorflow::Status();
}
%}
