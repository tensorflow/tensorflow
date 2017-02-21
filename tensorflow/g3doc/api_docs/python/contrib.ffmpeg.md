<!-- This file is machine generated: DO NOT EDIT! -->

# FFmpeg (contrib)
[TOC]

Working with audio using FFmpeg. See the @{$python/contrib.ffmpeg} guide.

- - -

### `tf.contrib.ffmpeg.decode_audio(contents, file_format=None, samples_per_second=None, channel_count=None)` {#decode_audio}

Create an op that decodes the contents of an audio file.

Note that ffmpeg is free to select the "best" audio track from an mp4.
https://trac.ffmpeg.org/wiki/Map

##### Args:


*  <b>`contents`</b>: The binary contents of the audio file to decode. This is a
      scalar.
*  <b>`file_format`</b>: A string specifying which format the contents will conform
      to. This can be mp3, mp4, ogg, or wav.
*  <b>`samples_per_second`</b>: The number of samples per second that is assumed.
      In some cases, resampling will occur to generate the correct sample
      rate.
*  <b>`channel_count`</b>: The number of channels that should be created from the
      audio contents. If the contents have more than this number, then
      some channels will be merged or dropped. If contents has fewer than
      this, then additional channels will be created from the existing ones.

##### Returns:

  A rank 2 tensor that has time along dimension 0 and channels along
  dimension 1. Dimension 0 will be `samples_per_second * length` wide, and
  dimension 1 will be `channel_count` wide. If ffmpeg fails to decode the
  audio then an empty tensor will be returned.


- - -

### `tf.contrib.ffmpeg.encode_audio(audio, file_format=None, samples_per_second=None)` {#encode_audio}

Creates an op that encodes an audio file using sampled audio from a tensor.

##### Args:


*  <b>`audio`</b>: A rank 2 tensor that has time along dimension 0 and channels along
      dimension 1. Dimension 0 is `samples_per_second * length` long in
      seconds.
*  <b>`file_format`</b>: The type of file to encode. "wav" is the only supported format.
*  <b>`samples_per_second`</b>: The number of samples in the audio tensor per second of
      audio.

##### Returns:

  A scalar tensor that contains the encoded audio in the specified file
  format.


