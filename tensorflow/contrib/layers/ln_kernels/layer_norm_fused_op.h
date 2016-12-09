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

namespace tensorflow {

struct LayerNormFusedArgs {
  // Input layer dimensions
  int depth;
  int n_slices;
  int slice_size;
  int n_inputs;
  float epsilon;

  LayerNormFusedArgs()
      : depth(0),
        n_slices(0),
        slice_size(0),
        n_inputs(0),
        epsilon(0){}
};

}  // namespace tensorflow
