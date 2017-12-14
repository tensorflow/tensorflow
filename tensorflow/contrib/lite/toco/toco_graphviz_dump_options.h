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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOCO_TOCO_GRAPHVIZ_DUMP_OPTIONS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOCO_TOCO_GRAPHVIZ_DUMP_OPTIONS_H_

#include <string>

namespace toco {

// Global data for determining whether to output graph viz format from toco.
struct GraphVizDumpOptions {
  std::string graphviz_first_array;
  std::string graphviz_last_array;
  std::string dump_graphviz;
  bool dump_graphviz_video = false;

  static GraphVizDumpOptions* singleton();
};

}  // namespace toco

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOCO_TOCO_GRAPHVIZ_DUMP_OPTIONS_H_
