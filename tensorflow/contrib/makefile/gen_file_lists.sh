#!/bin/bash -x
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This script generates the source file lists needed by the makefile by querying
# the master Bazel build configuration.

bazel query 'kind("source file", deps(//tensorflow/core:android_tensorflow_lib))' | \
grep "//tensorflow/.*\.proto$" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' \
> tensorflow/contrib/makefile/tf_proto_files.txt

bazel query 'kind("generated file", deps(//tensorflow/core:proto_text))' | \
grep "pb_text\.cc$" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' \
> tensorflow/contrib/makefile/tf_pb_text_files.txt

bazel query 'kind("source file", deps(//tensorflow/tools/proto_text:gen_proto_text_functions))' | \
grep -E "//tensorflow/.*\.cc$" | \
grep -E -v "jpeg" | \
grep -E -v "png" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' \
> tensorflow/contrib/makefile/proto_text_cc_files.txt

bazel query 'kind("generated file", deps(//tensorflow/tools/proto_text:gen_proto_text_functions))' | \
grep -E "//tensorflow/.*\.cc$" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' \
> tensorflow/contrib/makefile/proto_text_pb_cc_files.txt

bazel query 'kind("generated file", deps(//tensorflow/tools/proto_text:gen_proto_text_functions))' | \
grep -E "//tensorflow/.*\.h$" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' \
> tensorflow/contrib/makefile/proto_text_pb_h_files.txt
