// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include <stddef.h>
#include <stdint.h>
#include <string>

#include "monster_test_generated.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  flatbuffers::Verifier verifier(data, size);
  MyGame::Example::VerifyMonsterBuffer(verifier);
  return 0;
}
