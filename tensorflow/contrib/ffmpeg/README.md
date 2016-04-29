# FFmpeg TensorFlow integration

Decoding audio files can be done using a new op that uses
[FFmpeg](http://www.ffmpeg.org) to convert various audio file formats into
tensors.

tf.audio.decode_audio accepts MP3, WAV, and OGG file formats.

FFmpeg must be installed before these ops can be used. The ops will look for the
ffmpeg binary somewhere in `$PATH`. When the binary is unavailable, the error
`FFmpeg must be installed to run this op. FFmpeg can be found at
http://www.ffmpeg.org.` will be returned.
