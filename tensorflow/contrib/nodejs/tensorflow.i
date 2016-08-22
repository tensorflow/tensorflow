
%{
#include "tensorflow/core/public/version.h"

extern const char version[] = TF_VERSION_STRING;
%}

%include "../../core/public/version.h"
extern const char version[] = TF_VERSION_STRING;
