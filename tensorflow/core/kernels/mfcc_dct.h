/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Basic minimal DCT class for MFCC speech processing.

#ifndef TENSORFLOW_CORE_KERNELS_MFCC_DCT_H_
#define TENSORFLOW_CORE_KERNELS_MFCC_DCT_H_

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class MfccDct {
 public:
  MfccDct();
  bool Initialize(int input_length, int coefficient_count);
  void Compute(const std::vector<double>& input,
               std::vector<double>* output) const;

 private:
  bool initialized_;
  int coefficient_count_;
  int input_length_;
  std::vector<std::vector<double> > cosines_;
  TF_DISALLOW_COPY_AND_ASSIGN(MfccDct);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MFCC_DCT_H_
