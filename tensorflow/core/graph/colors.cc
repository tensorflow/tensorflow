/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/graph/colors.h"

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Color palette
// http://www.mulinblog.com/a-color-palette-optimized-for-data-visualization/
static const char* kColors[] = {
    "#F15854",  // red
    "#5DA5DA",  // blue
    "#FAA43A",  // orange
    "#60BD68",  // green
    "#F17CB0",  // pink
    "#B2912F",  // brown
    "#B276B2",  // purple
    "#DECF3F",  // yellow
    "#4D4D4D",  // gray
};

const char* ColorFor(int dindex) {
  return kColors[dindex % TF_ARRAYSIZE(kColors)];
}

}  // namespace tensorflow
