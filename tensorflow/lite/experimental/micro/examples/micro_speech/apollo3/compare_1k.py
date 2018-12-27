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

"""Debugging script for checking calculation values."""

import struct
import matplotlib.pyplot as plt
import numpy as np
# import soundfile as sf


def new_data_to_array(fn, datatype='int16'):
  """Converts file information to an in-memory array."""
  vals = []
  with open(fn) as f:
    for n, line in enumerate(f):
      if n is not 0:
        vals.extend([int(v, 16) for v in line.split()])
  b = ''.join(map(chr, vals))

  if datatype == 'int8':
    typestr = 'b'
    arraylen = int(len(b))
  elif datatype == 'int16':
    typestr = 'h'
    arraylen = int(len(b) // 2)
  elif datatype == 'int32':
    typestr = 'i'
    arraylen = int(len(b) // 4)
  if datatype == 'uint8':
    typestr = 'B'
    arraylen = int(len(b))
  elif datatype == 'uint16':
    typestr = 'H'
    arraylen = int(len(b) // 2)
  elif datatype == 'uint32':
    typestr = 'I'
    arraylen = int(len(b) // 4)

  y = np.array(struct.unpack('<' + typestr * arraylen, b))

  return y


# x is the fixed-point input in Qm.n format
def to_float(x, n):
  return x.astype(float) * 2**(-n)


micro_windowed_input = new_data_to_array(
    'micro_windowed_input.txt', datatype='int32')
cmsis_windowed_input = new_data_to_array(
    'cmsis_windowed_input.txt', datatype='int16')

micro_dft = new_data_to_array('micro_dft.txt', datatype='int32')
cmsis_dft = new_data_to_array('cmsis_dft.txt', datatype='int16')
py_dft = np.fft.rfft(to_float(cmsis_windowed_input, 15), n=512)
py_result = np.empty((2 * py_dft.size), dtype=np.float)
py_result[0::2] = np.real(py_dft)
py_result[1::2] = np.imag(py_dft)

micro_power = new_data_to_array('micro_power.txt', datatype='int32')
cmsis_power = new_data_to_array('cmsis_power.txt', datatype='int16')
py_power = np.square(np.abs(py_dft))

micro_power_avg = new_data_to_array('micro_power_avg.txt', datatype='uint8')
cmsis_power_avg = new_data_to_array('cmsis_power_avg.txt', datatype='uint8')

plt.figure(1)
plt.subplot(311)
plt.plot(micro_windowed_input, label='Micro fixed')
plt.legend()
plt.subplot(312)
plt.plot(cmsis_windowed_input, label='CMSIS fixed')
plt.legend()
plt.subplot(313)
plt.plot(to_float(micro_windowed_input, 30), label='Micro to float')
plt.plot(to_float(cmsis_windowed_input, 15), label='CMSIS to float')
plt.legend()

plt.figure(2)
plt.subplot(311)
plt.plot(micro_dft, label='Micro fixed')
plt.legend()
plt.subplot(312)
plt.plot(cmsis_dft, label='CMSIS fixed')
plt.legend()
plt.subplot(313)
plt.plot(to_float(micro_dft, 22), label='Micro to float')
# CMSIS result has 6 fractionanl bits (not 7) due to documentation error (see
# README.md)
plt.plot(to_float(cmsis_dft, 6), label='CMSIS to float')
plt.plot(py_result, label='Python result')
plt.legend()

plt.figure(3)
plt.subplot(311)
plt.plot(micro_power, label='Micro fixed')
plt.legend()
plt.subplot(312)
plt.plot(cmsis_power[0:256], label='CMSIS fixed')
plt.legend()
plt.subplot(313)
plt.plot(to_float(micro_power, 22), label='Micro to float')
plt.plot(to_float(cmsis_power[0:256], 6), label='CMSIS to float')
plt.plot(py_power, label='Python result')
plt.legend()

plt.figure(4)
plt.plot(micro_power_avg, label='Micro fixed')
plt.plot(cmsis_power_avg, label='CMSIS fixed')
plt.legend()
plt.show()

# t = np.arange(16000.*0.03)/16000.
# # Factor of 10 because micro preprocessing overflows otherwise
# sin1k = 0.1*np.sin(2*np.pi*1000*t)
#
# plt.figure(1)
# plt.subplot(511)
# plt.plot(sin1k)
# plt.title('Input sine')
#
# plt.subplot(512)
# plt.plot(to_float(micro_windowed_input, 30), label='Micro-Lite')
# plt.plot(to_float(cmsis_windowed_input, 15), label='CMSIS')
# plt.title('Windowed sine')
# plt.legend(loc='center right')
#
# plt.subplot(513)
# plt.plot(to_float(micro_dft, 22), label='Micro-Lite')
# plt.plot(to_float(cmsis_dft, 6), label='CMSIS')
# plt.title('FFT')
# plt.legend(loc='center')
#
# plt.subplot(514)
# plt.plot(to_float(micro_power, 22), label='Micro-Lite')
# plt.plot(to_float(cmsis_power[0:256], 6), label='CMSIS')
# plt.title('|FFT|^2')
# plt.legend(loc='center right')
#
# plt.subplot(515)
# plt.plot(micro_power_avg, label='Micro-Lite')
# plt.plot(cmsis_power_avg, label='CMSIS')
# plt.title('Averaged |FFT|^2')
# plt.legend(loc='center right')
#
# plt.tight_layout(pad=0, w_pad=0.2, h_pad=0.2)
#
# plt.show()
#
