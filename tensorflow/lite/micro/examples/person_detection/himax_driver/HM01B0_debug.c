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

#include "HM01B0_debug.h"
#include "am_util.h" // NOLINT

void hm01b0_framebuffer_dump(uint8_t* frame, uint32_t length) {
  am_util_stdio_printf("+++ frame +++");

  for (uint32_t i = 0; i < length; i++) {
    if ((i & 0xF) == 0x00) {
      am_util_stdio_printf("\n0x%08LX ", i);
      // this delay is to let itm have time to flush out data.
      am_util_delay_ms(1);
    }

    am_util_stdio_printf("%02X ", frame[i]);
  }

  am_util_stdio_printf("\n--- frame ---\n");
  am_util_delay_ms(1);
}

