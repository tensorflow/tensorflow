/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This data was created from a sample image from with a person in it.
// Convert original image to simpler format:
// convert -resize 96x96\! person.PNG person.bmp3
// Skip the 54 byte bmp3 header and add the reset of the bytes to a C array:
// xxd -s 54 -i /tmp/person.bmp3 > /tmp/person.cc

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_PERSON_IMAGE_DATA_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_PERSON_IMAGE_DATA_H_

#include <cstdint>

extern const int g_person_data_size;
extern const uint8_t g_person_data[];

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_PERSON_IMAGE_DATA_H_
