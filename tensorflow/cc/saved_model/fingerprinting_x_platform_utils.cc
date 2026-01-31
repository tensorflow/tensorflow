/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/saved_model/fingerprinting_x_platform_utils.h"

#include <string>

#include "absl/numeric/int128.h"
#include "absl/strings/str_format.h"
#include "tsl/platform/random.h"

// UINT64MAX is 18'446'744'073'709'551'615 (20 digits)
// UINT128MAX is 340'282'366'920'938'463'463'374'607'431'768'211'455 (39 dgts)
// After sqrt(INT64MAX) = 4'294'967'296 (4B models), it's 50% likely to be
// duplicates in the ID space. In comparison, sqrt(UINT128MAX) = UINT64MAX,
// meaning that we can continue generating unique IDs for a lot longer time
// if the UUID is generated from two random UINT64s. This can be replaced by
// random::New128() if that becomes available.
std::string tensorflow::saved_model::fingerprinting::CreateRandomUUID() {
  absl::uint128 uuid_1 = tsl::random::New64();
  absl::uint128 uuid_2 = tsl::random::New64();
  absl::uint128 uuid_complete = (uuid_1 << 64) | uuid_2;
  return absl::StrFormat("%020d", uuid_complete);
}
