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

/* SWIG wrapper for all of TensorFlow native functionality.
 * The includes are intentionally not alphabetically sorted, as the order of
 * includes follows dependency order */

%include "tensorflow/python/util/port.i"
%include "tensorflow/python/util/py_checkpoint_reader.i"
%include "tensorflow/python/util/stat_summarizer.i"

%include "tensorflow/python/lib/core/py_func.i"

%include "tensorflow/python/lib/io/py_record_reader.i"
%include "tensorflow/python/lib/io/py_record_writer.i"
%include "tensorflow/python/client/events_writer.i"

%include "tensorflow/python/client/tf_session.i"
%include "tensorflow/python/client/device_lib.i"
%include "tensorflow/python/client/quantize_training.i"

%include "tensorflow/python/lib/io/file_io.i"
%include "tensorflow/python/training/server_lib.i"

%include "tensorflow/python/framework/python_op_gen.i"

%include "tensorflow/python/framework/cpp_shape_inference.i"
%include "tensorflow/python/util/kernel_registry.i"
