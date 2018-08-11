# FFmpeg (contrib)
[TOC]

## Encoding and decoding audio using FFmpeg

TensorFlow provides Ops to decode and encode audio files using the
[FFmpeg](https://www.ffmpeg.org/) library. FFmpeg must be
locally [installed](https://ffmpeg.org/download.html) for these Ops to succeed.

Example:

```python
from tensorflow.contrib import ffmpeg

audio_binary = tf.read_file('song.mp3')
waveform = ffmpeg.decode_audio(
    audio_binary, file_format='mp3', samples_per_second=44100, channel_count=2)
uncompressed_binary = ffmpeg.encode_audio(
    waveform, file_format='wav', samples_per_second=44100)
```

*   `tf.contrib.ffmpeg.decode_audio`
*   `tf.contrib.ffmpeg.encode_audio`
