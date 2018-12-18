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

import numpy as np
import re
import struct
import matplotlib.pyplot as plt
import soundfile as sf

def new_data_to_array(fn):
    vals = []
    with open(fn) as f:
        for n, line in enumerate(f):
            if n is not 0:
                vals.extend([int(v, 16) for v in line.split()])
    b = ''.join(map(chr, vals))
    y = struct.unpack('<'+'h'*int(len(b)/2), b)
            
    return y

        
data = 'captured_data.txt'            
vals = np.array(new_data_to_array(data)).astype(float)

#plt.plot(vals, 'o-')
#plt.show(block=False)

wav = vals/np.max(np.abs(vals))
sf.write('captured_data.wav', wav, 16000)
