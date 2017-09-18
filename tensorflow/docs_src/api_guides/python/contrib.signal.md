# Signal Processing (contrib)
[TOC]

@{tf.contrib.signal} is a module for signal processing primitives. All 
operations have GPU support and are differentiable.

# Common Tasks

## Framing variable length sequences:

When dealing with variable length signals (e.g. audio) it is common to
"frame" them into multiple fixed length, potentially overlapping windows.
@{tf.contrib.signal.frame} does exactly this. For example:

```python
# A batch of float32 time-domain signals in the range [-1, 1] with shape
# [batch_size, signal_length]. Both batch_size and signal_length may be unknown.
signals = tf.placeholder(tf.float32, [None, None])

# Compute a [batch_size, ?, 128] tensor of fixed length, overlapping windows
# where each window overlaps the previous by 50%.
frames = tf.contrib.signal.frame(signals, frame_length=128, frame_step=64)
```

The `axis` parameter to @{tf.contrib.signal.frame} allows you to frame tensors
with inner structure (e.g. a spectrogram):

```python
# `magnitude_spectrograms` is a [batch_size, ?, 127] tensor of spectrograms. We
# would like to produce overlapping fixed-size spectrogram patches e.g. for use
# in a situation where a fixed size input is needed.
magnitude_spectrograms = tf.abs(tf.contrib.signal.stft(
    signals, frame_length=256, frame_step=128, fft_length=256))

# `spectrogram_patches` is a [batch_size, ?, 64, 127] tensor containing a 
# variable number of [64, 127] spectrogram patches per batch item.
spectrogram_patches = tf.contrib.signal.frame(
    magnitude_spectrograms, frame_length=64, frame_step=32, axis=1)
```

## Reconstructing framed sequences and applying a tapering window:

@{tf.contrib.signal.overlap_and_add} can be used to reconstruct a signal from a
framed representation produced in the above example.

```python
# Reconstructs `signals` from `frames` produced in the above example. However,
# the magnitude of `reconstructed_signals` will be greater than `signals`.
reconstructed_signals = tf.contrib.signal.overlap_and_add(frames, frame_step=64)
```

Note that because `frame_step` is 50% of `frame_length` in the above example,
the resulting reconstruction will have a greater magnitude than the original
`signals`.

To compensate for this, we can use a tapering window function. If the
window function satisfies the Constant Overlap-Add (COLA) property for the given
frame step, then it will recover the original `signals`.

@{tf.contrib.signal.hamming_window} and @{tf.contrib.signal.hann_window} both
satisfy the COLA property for a 50% overlap.

```python
frame_length = 128
frame_step = 64
windowed_frames = frames * tf.contrib.signal.hann_window(frame_length)
reconstructed_signals = tf.contrib.signal.overlap_and_add(
    windowed_frames, frame_step)
```

## Computing spectrograms:

A spectrogram is a time-frequency decomposition of a signal that indicates its
frequency content over time. There are many variants on how to compute a
spectrogram, but the most common approach is by taking the magnitude of the
[Short-time Fourier Transform][stft] (STFT), which can be computed with
@{tf.contrib.signal.stft}.

```python
# A batch of float32 time-domain signals in the range [-1, 1] with shape
# [batch_size, signal_length]. Both batch_size and signal_length may be unknown.
signals = tf.placeholder(tf.float32, [None, None])

# `stfts` is a complex64 Tensor representing the Short-time Fourier Transform of
# each signal in `signals`. Its shape is [batch_size, ?, fft_unique_bins] 
# where fft_unique_bins = fft_length // 2 + 1 = 513.
stfts = tf.contrib.signal.stft(signals, frame_length=1024, frame_step=512,
                               fft_length=1024)

# A power spectrogram is the squared magnitude of the complex-valued STFT.
# A float32 Tensor of shape [batch_size, ?, 513].
power_spectrograms = tf.real(stfts * tf.conj(stfts))

# An energy spectrogram is the magnitude of the complex-valued STFT.
# A float32 Tensor of shape [batch_size, ?, 513].
magnitude_spectrograms = tf.abs(stfts)
```

## Logarithmic compression:

It is common practice to apply a compressive nonlinearity such as a logarithm or
power-law compression to spectrograms.

When compressing with a logarithm, it's a good idea to use a stabilizing offset 
to avoid high dynamic ranges caused by the singularity at zero.

```python
log_offset = 1e-6
log_magnitude_spectrograms = tf.log(magnitude_spectrograms + log_offset)
log_power_spectrograms = tf.log(power_spectrograms + log_offset)
```

[stft]: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
