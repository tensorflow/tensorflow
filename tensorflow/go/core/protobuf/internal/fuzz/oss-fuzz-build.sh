# Copyright 2020 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script is executed by OSS-Fuzz's build to create fuzzer binaries.

for x in internal/fuzz/*; do
  if [ -d $x/corpus ]; then
    name=$(basename $x)
    compile_go_fuzzer google.golang.org/protobuf/$x Fuzz $name protolegacy
    zip -jr $OUT/${name}_seed_corpus.zip $x/corpus
  fi
done
