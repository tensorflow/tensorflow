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

