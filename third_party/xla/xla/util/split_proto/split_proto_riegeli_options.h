/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_UTIL_SPLIT_PROTO_SPLIT_PROTO_RIEGELI_OPTIONS_H_
#define XLA_UTIL_SPLIT_PROTO_SPLIT_PROTO_RIEGELI_OPTIONS_H_

#include "riegeli/records/record_writer.h"

namespace xla {

inline riegeli::RecordWriterBase::Options GetSplitProtoRiegeliOptions() {
  riegeli::RecordWriterBase::Options options;
  // We mainly want to optimize for reading speed, over compression ratio.
  // In our benchmarks Snappy level 2 showed the fastest decompression speeds,
  // which also aligns with the recommendations in go/fast/68.
  options.set_snappy(2);
  return options;
}

}  // namespace xla

#endif  // XLA_UTIL_SPLIT_PROTO_SPLIT_PROTO_RIEGELI_OPTIONS_H_
