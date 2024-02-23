#!/bin/bash
# Copyright 2019 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

cd "$(git rev-parse --show-toplevel)"
mkdir -p .cache/benchdata
cd .cache/benchdata

echo "This script needs to be updated to work with https://github.com/google/fleetbench" >&2
echo "See https://github.com/golang/protobuf/issues/1570" >&2
exit 1
