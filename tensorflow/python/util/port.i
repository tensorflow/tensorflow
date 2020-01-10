%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/core/util/port.h"
%}

%ignoreall
%unignore tensorflow;
%unignore tensorflow::IsGoogleCudaEnabled;
%include "tensorflow/core/util/port.h"
%unignoreall
