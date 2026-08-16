#!/bin/bash

# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
# ============================================================================
set -xe

source "$(dirname "${BASH_SOURCE[0]}")/shell_common.sh"

BUILDIFIER_VERSION="${BUILDIFIER_VERSION:-433ea85}" # 6.4.0

if ! command -v go >/dev/null 2>&1; then
  echoerr "Go is required to run the buildifier check."
  echoerr "Install Go from https://go.dev/doc/install"
  exit 1
fi

if ! command -v parallel >/dev/null 2>&1; then
  echoerr "GNU Parallel is required to run the buildifier check."
  exit 1
fi

GOBIN="${GOBIN:-$(go env GOPATH)/bin}"
parallel --ungroup --retries 3 --delay 15 --nonall -- \
  env GOBIN="$GOBIN" go install \
  "github.com/bazelbuild/buildtools/buildifier@${BUILDIFIER_VERSION}"

"$GOBIN/buildifier" \
  --lint=warn --warnings=-out-of-order-load -r xla/
