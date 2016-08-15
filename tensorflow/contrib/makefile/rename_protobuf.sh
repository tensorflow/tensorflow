#!/bin/bash

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

# This script modifies the downloaded protobuf library and the TensorFlow source
# to put all protobuf-related symbols into the google::protobuf3 namespace
# instead of the default google::protobuf. This is necessary to work around
# linking issues for applications that use protobuf v2 already, and want to
# adopt TensorFlow, since otherwise the two libraries have duplicate function
# symbols that clash. It also renames all the include paths to google/protobuf3
# throughout the protobuf and TensorFlow code.
# This is a massive hack, and if possible it's recommended that you switch your
# whole application to protobuf v3 so there's no mismatch with TensorFlow. There
# are also no guarantees that this will continue to work with future versions of
# protobuf or TensorFlow, or that it's a bulletproof solution for every
# application.
#
# To use this script, run the following sequence:
# tensorflow/contrib/makefile/download_dependencies.sh
# tensorflow/contrib/makefile/rename_protobuf.sh
#
# You can then build the source as normal. For example on iOS:
# tensorflow/contrib/makefile/compile_ios_protobuf.sh
# tensorflow/contrib/makefile/compile_ios_tensorflow.sh
#
# Note that this script modifies the source code in-place, so once it's been run
# it's no longer suitable for further manual modifications, since the difference
# with the top of tree will already be large. 

mv tensorflow/contrib/makefile/downloads/protobuf/src/google/protobuf \
 tensorflow/contrib/makefile/downloads/protobuf//src/google/protobuf3

# Rename protobuf #includes to use protobuf3.
find . \
 -type f \
 \( -name "*.cc" -or -name "*.h" \) \
 -exec sed -i '' \
 's%#include \([<"]\)google/protobuf/%#include \1google/protobuf3/%' {} \;
find . \
 -type f \
 -name "*.proto" \
 -exec sed -i '' \
 's%import \(.*\)\([<"]\)google/protobuf/%import \1\2google/protobuf3/%' {} \;

# Rename the namespace mentions.
find . \
 -type f \
 \( -name "*.cc" -or -name "*.h" \) \
 -exec sed -i '' \
 's%namespace protobuf\([^3]\)%namespace protobuf3\1%' {} \;
find . \
 -type f \
 \( -name "*.cc" -or -name "*.h" \) \
 -exec sed -i '' \
 's%protobuf::%protobuf3::%g' {} \;
sed -i '' 's%::google::protobuf;%google::protobuf3;%' \
 tensorflow/core/platform/default/protobuf.h

# Fix up a couple of special build scripts that look for particular files.
sed -i '' 's%src/google/protobuf/message.cc%src/google/protobuf3/message.cc%' \
 tensorflow/contrib/makefile/downloads/protobuf/configure.ac 
sed -i '' 's%src/google/protobuf/stubs/common.h%src/google/protobuf3/stubs/common.h%' \
 tensorflow/contrib/makefile/downloads/protobuf/autogen.sh

# Update the locations within the protobuf makefile.
sed -i '' 's%google/protobuf/%google/protobuf3/%g' \
 tensorflow/contrib/makefile/downloads/protobuf/src/Makefile.am

# Make sure protoc can find the new google/protobuf3 paths by putting them at
# the root directory.
cp -r tensorflow/contrib/makefile/downloads/protobuf/src/google .

# Update the protobuf commands used in the makefile.
sed -i '' 's%$(PROTOC) $(PROTOCFLAGS) $< --cpp_out $(PROTOGENDIR)%tensorflow/contrib/makefile/rename_protoc.sh $(PROTOC) $(PROTOCFLAGS) $< --cpp_out $(PROTOGENDIR)%' tensorflow/contrib/makefile/Makefile
sed -i '' 's%$(PROTOC) $(PROTOCFLAGS) $< --cpp_out $(HOST_GENDIR)%tensorflow/contrib/makefile/rename_protoc.sh $(PROTOC) $(PROTOCFLAGS) $< --cpp_out $(HOST_GENDIR)%' tensorflow/contrib/makefile/Makefile
sed -i '' 's%$(PROTO_TEXT) \\%tensorflow/contrib/makefile/rename_prototext.sh $(PROTO_TEXT) \\%' tensorflow/contrib/makefile/Makefile
