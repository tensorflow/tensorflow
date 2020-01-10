%include "tensorflow/python/platform/base.i"
%import(module="tensorflow.python.pywrap_tensorflow") "tensorflow/python/lib/core/status.i"

%{
#include "tensorflow/core/public/tensorflow_server.h"
%}

%ignoreall

%unignore tensorflow;
%unignore tensorflow::LaunchTensorFlow;

%include "tensorflow/core/public/tensorflow_server.h"

%unignoreall

