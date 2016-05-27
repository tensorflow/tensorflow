#!/bin/bash -x

# This script generates the source file lists needed by the makefile by querying
# the master Bazel build configuration.

bazel query 'kind("source file", deps(//tensorflow/core:android_tensorflow_lib))' | \
grep "//tensorflow/.*\.cc$" | \
grep -v "gen_proto_text" | \
grep -E -v "jpeg" | \
grep -E -v "png" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' \
> make/tf_cc_files.txt

bazel query 'kind("source file", deps(//tensorflow/core:android_tensorflow_lib))' | \
grep "//tensorflow/.*\.proto$" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' \
> make/tf_proto_files.txt

bazel query 'kind("generated file", deps(//tensorflow/core:proto_text))' | \
grep "pb_text\.cc$" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' \
> make/tf_pb_text_files.txt

bazel query 'kind("source file", deps(//tensorflow/tools/proto_text:gen_proto_text_functions))' | \
grep -E "//tensorflow/.*\.cc$" | \
grep -E -v "jpeg" | \
grep -E -v "png" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' \
> make/proto_text_cc_files.txt

bazel query 'kind("generated file", deps(//tensorflow/tools/proto_text:gen_proto_text_functions))' | \
grep -E "//tensorflow/.*\.cc$" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' \
> make/proto_text_pb_cc_files.txt

bazel query 'kind("generated file", deps(//tensorflow/tools/proto_text:gen_proto_text_functions))' | \
grep -E "//tensorflow/.*\.h$" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' \
> make/proto_text_pb_h_files.txt
