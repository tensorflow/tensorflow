# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Signal processing operations.

See the [tf.signal](https://tensorflow.org/api_guides/python/contrib.signal)
guide.

@@frame
@@hamming_window
@@hann_window
@@inverse_stft
@@inverse_stft_window_fn
@@mfccs_from_log_mel_spectrograms
@@linear_to_mel_weight_matrix
@@overlap_and_add
@@stft

[hamming]: https://en.wikipedia.org/wiki/Window_function#Hamming_window
[hann]: https://en.wikipedia.org/wiki/Window_function#Hann_window
[mel]: https://en.wikipedia.org/wiki/Mel_scale
[mfcc]: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
[stft]: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.python.ops.signal.dct_ops import dct
from tensorflow.python.ops.signal.fft_ops import fft
from tensorflow.python.ops.signal.fft_ops import fft2d
from tensorflow.python.ops.signal.fft_ops import fft3d
from tensorflow.python.ops.signal.fft_ops import fftshift
from tensorflow.python.ops.signal.fft_ops import rfft
from tensorflow.python.ops.signal.fft_ops import rfft2d
from tensorflow.python.ops.signal.fft_ops import rfft3d
from tensorflow.python.ops.signal.dct_ops import idct
from tensorflow.python.ops.signal.fft_ops import ifft
from tensorflow.python.ops.signal.fft_ops import ifft2d
from tensorflow.python.ops.signal.fft_ops import ifft3d
from tensorflow.python.ops.signal.fft_ops import ifftshift
from tensorflow.python.ops.signal.fft_ops import irfft
from tensorflow.python.ops.signal.fft_ops import irfft2d
from tensorflow.python.ops.signal.fft_ops import irfft3d
from tensorflow.python.ops.signal.mel_ops import linear_to_mel_weight_matrix
from tensorflow.python.ops.signal.mfcc_ops import mfccs_from_log_mel_spectrograms
from tensorflow.python.ops.signal.reconstruction_ops import overlap_and_add
from tensorflow.python.ops.signal.shape_ops import frame
from tensorflow.python.ops.signal.spectral_ops import inverse_stft
from tensorflow.python.ops.signal.spectral_ops import inverse_stft_window_fn
from tensorflow.python.ops.signal.spectral_ops import stft
from tensorflow.python.ops.signal.window_ops import hamming_window
from tensorflow.python.ops.signal.window_ops import hann_window
# pylint: enable=unused-import
