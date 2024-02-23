#!/bin/bash
# Copyright 2018 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

go test google.golang.org/protobuf -run='^TestIntegration$' -v -timeout=60m -count=1 "$@" -regenerate
exit $?
