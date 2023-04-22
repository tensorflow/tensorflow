/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_DENORMAL_H_
#define TENSORFLOW_CORE_PLATFORM_DENORMAL_H_

#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace port {

// State for handling of denormals.
class DenormalState {
 public:
  DenormalState(bool flush_to_zero, bool denormals_are_zero)
      : flush_to_zero_(flush_to_zero),
        denormals_are_zero_(denormals_are_zero) {}

  // Output denormals of floating-point operations are flushed to zero.
  inline bool flush_to_zero() const { return flush_to_zero_; }

  // Input denormals to floating-point operations are treated as zero.
  inline bool denormals_are_zero() const { return denormals_are_zero_; }

  bool operator==(const DenormalState& other) const;
  bool operator!=(const DenormalState& other) const;

 private:
  bool flush_to_zero_;
  bool denormals_are_zero_;
};

// Gets the platform's current state for handling denormals.
DenormalState GetDenormalState();

// Sets handling of denormals if the platform allows it. Returns `true` if the
// platform supports setting denormals to the specified state. Otherwise the
// denormal state remains unmodified and false is returned.
bool SetDenormalState(const DenormalState& state);

// Remembers the flush denormal state on construction and restores that same
// state on destruction.
class ScopedRestoreFlushDenormalState {
 public:
  ScopedRestoreFlushDenormalState();
  ~ScopedRestoreFlushDenormalState();

 private:
  DenormalState denormal_state_;
  TF_DISALLOW_COPY_AND_ASSIGN(ScopedRestoreFlushDenormalState);
};

// While this class is active, denormal floating point numbers are flushed
// to zero.  The destructor restores the original flags.
class ScopedFlushDenormal {
 public:
  ScopedFlushDenormal();

 private:
  ScopedRestoreFlushDenormalState restore_;
  TF_DISALLOW_COPY_AND_ASSIGN(ScopedFlushDenormal);
};

// While this class is active, denormal floating point numbers are not flushed
// to zero.  The destructor restores the original flags.
class ScopedDontFlushDenormal {
 public:
  ScopedDontFlushDenormal();

 private:
  ScopedRestoreFlushDenormalState restore_;
  TF_DISALLOW_COPY_AND_ASSIGN(ScopedDontFlushDenormal);
};

}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_DENORMAL_H_
