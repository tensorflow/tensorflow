/* Copyright 2017 The OpenXLA Authors.

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

// Usage: show_literal <path-to-serialized-literal-proto>
//
// Dumps out the Literal::ToString of a tsl::WriteBinaryProto format
// Literal serialized on disk.

#include <stdio.h>

#include <string>

#include "xla/literal.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

int main(int argc, char **argv) {
  tsl::port::InitMain(argv[0], &argc, &argv);

  if (argc < 2) {
    LOG(QFATAL) << "Usage: " << argv[0]
                << " <path-to-serialized-literal-proto>";
  }

  xla::LiteralProto literal_proto;
  TF_CHECK_OK(
      tsl::ReadBinaryProto(tsl::Env::Default(), argv[1], &literal_proto));
  xla::Literal literal = xla::Literal::CreateFromProto(literal_proto).value();
  LOG(INFO) << "literal: " << literal_proto.ShortDebugString();
  fprintf(stderr, "%s\n", literal.ToString().c_str());
}
