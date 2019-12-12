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

#include "tensorflow/lite/experimental/micro/examples/magic_wand/output_handler.h"

#include "Particle.h"

void HandleOutput(tflite::ErrorReporter *error_reporter, int kind)
{
  // The first time this method runs, set up our LED
  static bool is_initialized = false;
  if (!is_initialized)
  {
    pinMode(D7, OUTPUT);
    is_initialized = true;
  }
  // Toggle the LED every time an inference is performed
  static int count = 0;
  ++count;
  if (count & 1)
  {
    digitalWrite(D7, HIGH);
  }
  else
  {
    digitalWrite(D7, LOW);
  }
  // Print some ASCII art for each gesture
  if (kind == 0)
  {
    RGB.control(true);
    RGB.color(125, 0, 125); // Magenta

    Mesh.publish("gesture", "wing");

    error_reporter->Report(
        "WING:\n\r*         *         *\n\r *       * *       "
        "*\n\r  *     *   *     *\n\r   *   *     *   *\n\r    * *       "
        "* *\n\r     *         *\n\r");
  }
  else if (kind == 1)
  {
    RGB.control(true);
    RGB.color(0, 0, 255); // Blue

    Mesh.publish("gesture", "ring");

    error_reporter->Report(
        "RING:\n\r          *\n\r       *     *\n\r     *         *\n\r "
        "   *           *\n\r     *         *\n\r       *     *\n\r      "
        "    *\n\r");
  }
  else if (kind == 2)
  {
    RGB.control(true);
    RGB.color(255, 0, 0); // Red

    Mesh.publish("gesture", "slope");

    error_reporter->Report(
        "SLOPE:\n\r        *\n\r       *\n\r      *\n\r     *\n\r    "
        "*\n\r   *\n\r  *\n\r * * * * * * * *\n\r");
  }
  else
  {
    RGB.control(false);
  }
}
