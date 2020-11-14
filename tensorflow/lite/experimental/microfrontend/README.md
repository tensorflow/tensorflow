# Audio "frontend" TensorFlow operations for feature generation
The most common module used by most audio processing modules is the feature
generation (also called frontend). It receives raw audio input, and produces
filter banks (a vector of values).

More specifically the audio signal goes through a pre-emphasis filter
(optionally); then gets sliced into (overlapping) frames and a window function
is applied to each frame; afterwards, we do a Fourier transform on each frame
(or more specifically a Short-Time Fourier Transform) and calculate the power
spectrum; and subsequently compute the filter banks.

## Operations
Here we provide implementations for both a TensorFlow and TensorFlow Lite
operations that encapsulate the functionality of the audio frontend.

Both frontend Ops receives audio data and produces as many unstacked frames
(filterbanks) as audio is passed in, according to the configuration.

The processing uses a lightweight library to perform:

1. A slicing window function
2. Short-time FFTs
3. Filterbank calculations
4. Noise reduction
5. Auto Gain Control
6. Logarithmic scaling

Please refer to the Op's documentation for details on the different
configuration parameters.

However, it is important to clarify the contract of the Ops:

> *A frontend OP will produce as many unstacked frames as possible with the
> given audio input.*

This means:

1. The output is a rank-2 Tensor, where each row corresponds to the
  sequence/time dimension, and each column is the feature dimension).
2. It is expected that the Op will receive the right input (in terms of
  positioning in the audio stream, and the amount), as needed to produce the
  expected output.
3. Thus, any logic to slice, cache, or otherwise rearrange the input and/or
  output of the operation must be handled externally in the graph.

For example, a 200ms audio input will produce an output tensor of shape
`[18, num_channels]`, when configured with a `window_size=25ms`, and
`window_step=10ms`. The reason being that when reaching the point in the
audio at 180ms there’s not enough audio to construct a complete window.

Due to both functional and efficiency reasons, we provide the following
functionality related to input processing:

**Padding.** A boolean flag `zero_padding` that indicates whether to pad the
audio with zeros such that we generate output frames based on the `window_step`.
This means that in the example above, we would generate a tensor of shape
`[20, num_channels]` by adding enough zeros such that we step over all the
available audio and still be able to create complete windows of audio (some of
the window will just have zeros; in the example above, frame 19 and 20 will have
the equivalent of 5 and 15ms full of zeros respectively).

<!-- TODO
Stacking. An integer that indicates how many contiguous frames to stack in the output tensor’s first dimension, such that the tensor is shaped [-1, stack_size * num_channels]. For example, if the stack_size is 3, the example above would produce an output tensor shaped [18, 120] is padding is false, and [20, 120] is padding is set to true.
-->

**Striding.** An integer `frame_stride` that indicates the striding step used to
generate the output tensor, thus determining the second dimension. In the
example above, with a `frame_stride=3`, the output tensor would have a shape of
`[6, 120]` when `zero_padding` is set to false, and `[7, 120]` when
`zero_padding` is set to true.

<!-- TODO
Note we would not expect the striding step to be larger than the stack_size
(should we enforce that?).
-->
