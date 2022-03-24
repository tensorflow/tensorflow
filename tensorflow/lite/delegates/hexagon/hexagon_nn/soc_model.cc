/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/soc_model.h"

#include <cstdlib>

namespace tflite {
namespace delegates {
// Implementation below is similar to the one inside the Hexagon SDK for
// fetching the SoC information.
// TODO(b/144536839): Look in sharing the code with Hexagon SDK if possible.

int get_soc_id(int* soc_id) {
  int fd;
  if (!access("/sys/devices/soc0/soc_id", F_OK)) {
    fd = open("/sys/devices/soc0/soc_id", O_RDONLY);
  } else {
    fd = open("/sys/devices/system/soc/soc0/id", O_RDONLY);
  }
  if (fd == -1) {
    return -1;
  }

  char raw_buf[SOC_ID_BUFFER_LENGTH];
  const int bytes_read = read(fd, raw_buf, SOC_ID_BUFFER_LENGTH - 1);
  // read returns -1 on failure, so check and return if failed.
  if (bytes_read == -1) {
    return -1;  // failure
  }
  raw_buf[SOC_ID_BUFFER_LENGTH - 1] = 0;
  *soc_id = atoi(raw_buf);
  close(fd);

  return 0;
}

SocSkelTable getsoc_model() {
  int soc_id;
  get_soc_id(&soc_id);

  int i = 0;
  for (i = 0; socSkelInfo[i].soc_id != 0; i++) {
    if (socSkelInfo[i].soc_id == soc_id) {
      return socSkelInfo[i];
    }
  }

  return socSkelInfo[i];
}
}  // namespace delegates
}  // namespace tflite
