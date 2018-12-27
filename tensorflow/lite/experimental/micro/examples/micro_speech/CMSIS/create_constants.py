# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Outputs tables used for fast calculations at runtime."""

# import soundfile as sf
import numpy as np


def to_cc(x, varname, directory='', scale_factor=1):
  """Writes table values to a C++ source file."""
  x = (x / np.max(np.abs(x))) * 32768 * scale_factor
  x[x > 32767] = 32767
  x[x < -32768] = -32768
  x = x.astype(int)
  x = [str(v) if i % 10 != 0 else '\n    ' + str(v) for i, v in enumerate(x)]

  cmsis_path = 'tensorflow/lite/experimental/micro/examples/micro_speech/CMSIS'
  xstr = '#include "{}/{}.h"\n\n'.format(cmsis_path, varname)
  xstr += 'const int g_{}_size = {};\n'.format(varname, len(x))
  xstr += 'const int16_t g_{}[{}] = {{{}}};\n'.format(varname, len(x),
                                                      ', '.join(x))

  with open(directory + varname + '.cc', 'w') as f:
    f.write(xstr)


def to_h(_, varname, directory=''):
  """Writes a header file for the table values."""
  tf_prepend = 'TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_'
  xstr = '#ifndef {}{}_H_\n'.format(tf_prepend, varname.upper())
  xstr += '#define {}{}_H_\n\n'.format(tf_prepend, varname.upper())
  xstr += '#include <cstdint>\n\n'
  xstr += 'extern const int g_{}_size;\n'.format(varname)
  xstr += 'extern const int16_t g_{}[];\n\n'.format(varname)
  xstr += '#endif'

  with open(directory + varname + '.h', 'w') as f:
    f.write(xstr)


# x = sf.read('yes_f2e59fea_nohash_1.wav')[0]
# to_cc(x, 'yes_waveform')
# to_h(x, 'yes_waveform')
#
# x = sf.read('no_f9643d42_nohash_4.wav')[0]
# to_cc(x, 'no_waveform')
# to_h(x, 'no_waveform')

# 30ms of data @ 16 kHz = 480 samples
hann = np.hanning(int(16000 * 0.03))  # Window 30ms of data
to_cc(hann, 'hanning', directory='./')
to_h(hann, 'hanning', directory='./')

t = np.arange(16000. * 0.03) / 16000.
sin1k = np.sin(
    2 * np.pi * 1000 *
    t)  # Factor of 10 because micro preprocessing overflows otherwise
to_cc(sin1k, 'sin_1k', directory='./', scale_factor=0.1)
to_h(sin1k, 'sin_1k', directory='./')
