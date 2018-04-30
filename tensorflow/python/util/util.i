/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/python/util/util.h"
%}

%ignoreall

%unignore tensorflow;
%unignore tensorflow::swig;
// The %exception block defined in tf_session.i releases the Python GIL for
// the length of each wrapped method. This file is included in tensorflow.i
// after tf_session.i and inherits this definition. We disable this behavior
// for functions in this module because they use python methods that need GIL.
// TODO(iga): Find a way not to leak such definitions across files.

%unignore tensorflow::swig::RegisterSequenceClass;
%noexception tensorflow::swig::RegisterSequenceClass;

%unignore tensorflow::swig::IsSequence;
%noexception tensorflow::swig::IsSequence;

%unignore tensorflow::swig::IsNamedtuple;
%noexception tensorflow::swig::IsNamedtuple;

%unignore tensorflow::swig::SameNamedtuples;
%noexception tensorflow::swig::SameNamedtuples;

%unignore tensorflow::swig::AssertSameStructure;
%noexception tensorflow::swig::AssertSameStructure;

%unignore tensorflow::swig::Flatten;
%noexception tensorflow::swig::Flatten;

%include "tensorflow/python/util/util.h"

%unignoreall
