/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "flatbuffers/flexbuffers.h"

const char* license =
    "/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.\n"
    "Licensed under the Apache License, Version 2.0 (the \"License\");\n"
    "you may not use this file except in compliance with the License.\n"
    "You may obtain a copy of the License at\n\n"
    "    http://www.apache.org/licenses/LICENSE-2.0\n\n"
    "Unless required by applicable law or agreed to in writing, software\n"
    "distributed under the License is distributed on an \"AS IS\" BASIS,\n"
    "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
    "See the License for the specific language governing permissions and\n"
    "limitations under the License.\n"
    "======================================================================="
    "=======*/\n";

void generate(const char* name, bool use_regular_nms) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("max_detections", 3);
    fbb.Int("max_classes_per_detection", 1);
    fbb.Int("detections_per_class", 1);
    fbb.Bool("use_regular_nms", use_regular_nms);
    fbb.Float("nms_score_threshold", 0.0);
    fbb.Float("nms_iou_threshold", 0.5);
    fbb.Int("num_classes", 2);
    fbb.Float("y_scale", 10.0);
    fbb.Float("x_scale", 10.0);
    fbb.Float("h_scale", 5.0);
    fbb.Float("w_scale", 5.0);
  });
  fbb.Finish();
  const uint8_t* init_data = reinterpret_cast<const uint8_t*>(
      fbb.GetBuffer().data());
  int fbb_size = fbb.GetBuffer().size();

  printf("int gen_data_size_%s = %d;\n", name, fbb_size);
  printf("static constexpr uint8_t gen_data_%s[%d] = { ", name, fbb_size);
  for (size_t i = 0; i < fbb_size; i++) {
      printf("%u, ", init_data[i]);
  }
  printf("};\n\n");
}

int main() {
  printf("%s\n", license);
  printf("// This file is generated. See:\n");
  printf("// tensorflow/lite/micro/kernels/detection_postprocess_test/readme");
  printf("\n\n");
  printf("namespace tflite {\n");
  printf("namespace testing {\n");
  printf("namespace {\n");
  generate("none_regular_nms", false);
  generate("regular_nms", true);
  printf("}\n");
  printf("}\n");
  printf("}\n");
}
