# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""Video feature."""

import os
import subprocess
import tempfile
from typing import Any, List, Optional, Sequence

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.data.experimental.core import utils
from tensorflow.data.experimental.core.features import image_feature
from tensorflow.data.experimental.core.features import sequence_feature
from tensorflow.data.experimental.core.utils import type_utils

Json = type_utils.Json
PilImage = Any  # Require lazy deps.

# Framerate for the `tfds.as_dataframe` visualization
# Could add a framerate kwargs in __init__ to allow datasets to customize
# the output.
_VISU_FRAMERATE = 10


class Video(sequence_feature.Sequence):
  """`FeatureConnector` for videos, encoding frames individually on disk.

  Video: The image connector accepts as input a 4 dimensional `tf.uint8` array
  representing a video, a sequence of paths to encoded frames, or a path or a
  file object that can be decoded with ffmpeg. Note that not all formats in
  ffmpeg support reading from pipes, so providing a file object might fail.
  Furthermore, if a path is given that is not on the local file system, we first
  copy it to a temporary local file before passing it to ffmpeg.

  Output:
    video: tf.Tensor of type `tf.uint8` and shape
      [num_frames, height, width, channels], where channels must be 1 or 3

  Example:

    * In the DatasetInfo object:

    ```
    features=features.FeatureDict({
        'video': features.Video(shape=(None, 64, 64, 3)),
    })
    ```

    * During generation, you can use any of:

    ```
    yield {
        'video': np.ones(shape=(128, 64, 64, 3), dtype=np.uint8),
    }
    ```

    or list of frames:

    ```
    yield {
        'video': ['path/to/frame001.png', 'path/to/frame002.png'],
    }
    ```

    or path to video (including `os.PathLike` objects):

    ```
    yield {
        'video': '/path/to/video.avi',
    }
    ```

    or file object (or `bytes`):

    ```
    yield {
        'video': tf.io.gfile.GFile('/complex/path/video.avi'),
    }
    ```

  """

  def __init__(
      self,
      shape: Sequence[int],
      encoding_format: str = 'png',
      ffmpeg_extra_args: Sequence[str] = (),
  ):
    """Initializes the connector.

    Args:
      shape: tuple of ints, the shape of the video (num_frames, height, width,
        channels), where channels is 1 or 3.
      encoding_format: The video is stored as a sequence of encoded images.
        You can use any encoding format supported by image_feature.Feature.
      ffmpeg_extra_args: A sequence of additional args to be passed to the
        ffmpeg binary. Specifically, ffmpeg will be called as:
          ``
          ffmpeg -i <input_file> <ffmpeg_extra_args> %010d.<encoding_format>
          ``
    Raises:
      ValueError: If the shape is invalid
    """
    shape = tuple(shape)
    if len(shape) != 4:
      raise ValueError('Video shape should be of rank 4')
    self._encoding_format = encoding_format
    self._extra_ffmpeg_args = list(ffmpeg_extra_args or [])
    super(Video, self).__init__(
        image_feature.Image(shape=shape[1:], encoding_format=encoding_format),
        length=shape[0],
    )

  def _ffmpeg_decode(self, path_or_fobj):
    if isinstance(path_or_fobj, type_utils.PathLikeCls):
      ffmpeg_args = ['-i', os.fspath(path_or_fobj)]
      ffmpeg_stdin = None
    else:
      ffmpeg_args = ['-i', 'pipe:0']
      ffmpeg_stdin = path_or_fobj.read()
    ffmpeg_args += self._extra_ffmpeg_args

    with tempfile.TemporaryDirectory() as ffmpeg_dir:
      out_pattern = os.path.join(ffmpeg_dir, f'%010d.{self._encoding_format}')
      ffmpeg_args.append(out_pattern)
      _ffmpeg_run(ffmpeg_args, ffmpeg_stdin)
      frames = [  # Load all encoded images
          p.read_bytes() for p in sorted(utils.as_path(ffmpeg_dir).iterdir())
      ]
    return frames

  def encode_example(self, video_or_path_or_fobj):
    """Converts the given image into a dict convertible to tf example."""
    if isinstance(video_or_path_or_fobj, type_utils.PathLikeCls):
      video_or_path_or_fobj = os.fspath(video_or_path_or_fobj)
      if not os.path.isfile(video_or_path_or_fobj):
        _, video_temp_path = tempfile.mkstemp()
        try:
          tf.io.gfile.copy(
              video_or_path_or_fobj, video_temp_path, overwrite=True)
          encoded_video = self._ffmpeg_decode(video_temp_path)
        finally:
          os.unlink(video_temp_path)
      else:
        encoded_video = self._ffmpeg_decode(video_or_path_or_fobj)
    elif isinstance(video_or_path_or_fobj, bytes):
      with tempfile.TemporaryDirectory() as tmpdirname:
        video_temp_path = os.path.join(tmpdirname, 'video')
        with tf.io.gfile.GFile(video_temp_path, 'wb') as f:
          f.write(video_or_path_or_fobj)
        encoded_video = self._ffmpeg_decode(video_temp_path)
    elif hasattr(video_or_path_or_fobj, 'read'):
      encoded_video = self._ffmpeg_decode(video_or_path_or_fobj)
    else:  # List of images, np.array,...
      encoded_video = video_or_path_or_fobj
    return super(Video, self).encode_example(encoded_video)

  @classmethod
  def from_json_content(cls, value: Json) -> 'Video':
    shape = tuple(value['shape'])
    encoding_format = value['encoding_format']
    ffmpeg_extra_args = value['ffmpeg_extra_args']
    return cls(shape, encoding_format, ffmpeg_extra_args)

  def to_json_content(self) -> Json:
    return {
        'shape': list(self.shape),
        'encoding_format': self._encoding_format,
        'ffmpeg_extra_args': self._extra_ffmpeg_args
    }

  def repr_html(self, ex: np.ndarray) -> str:
    """Video are displayed as GIFs."""
    # Use GIF to generate a HTML5 compatible video if FFMPEG is not
    # installed on the system.
    images = [image_feature.create_thumbnail(frame) for frame in ex]

    # Display the video HTML (either GIF of mp4 if ffmpeg is installed)
    try:
      _ffmpeg_run(['-version'])  # Check for ffmpeg installation.
    except FileNotFoundError:
      # print as `stderr` is displayed poorly on Colab
      print('FFMPEG not detected. Falling back on GIF.')
      return _get_repr_html_gif(images)
    else:
      return _get_repr_html_ffmpeg(images)


def _ffmpeg_run(
    args: List[str],
    stdin: Optional[bytes] = None,
) -> None:
  """Executes the ffmpeg function."""
  ffmpeg_path = 'ffmpeg'
  try:
    cmd_args = [ffmpeg_path] + args
    subprocess.run(
        cmd_args,
        check=True,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
  except subprocess.CalledProcessError as e:
    raise ValueError(
        f'Command {e.cmd} returned error code {e.returncode}:\n'
        f'stdout={e.stdout.decode("utf-8")}\n'
        f'stderr={e.stderr.decode("utf-8")}\n'
    )
  except FileNotFoundError as e:
    raise FileNotFoundError(
        'It seems that ffmpeg is not installed on the system. Please follow '
        'the instrutions at https://ffmpeg.org/. '
        f'Original exception: {e}'
    )


def _get_repr_html_ffmpeg(images: List[PilImage]) -> str:
  """Runs ffmpeg to get the mp4 encoded <video> str."""
  # Find number of digits in len to give names.
  num_digits = len(str(len(images))) + 1
  with tempfile.TemporaryDirectory() as video_dir:
    for i, img in enumerate(images):
      f = os.path.join(video_dir, f'img{i:0{num_digits}d}.png')
      img.save(f, format='png')

    ffmpeg_args = [
        '-framerate', str(_VISU_FRAMERATE),
        '-i', os.path.join(video_dir, f'img%0{num_digits}d.png'),
        # Using native h264 to encode video stream to H.264 codec
        # Default encoding does not seems to be supported by chrome.
        '-vcodec', 'h264',
        # When outputting H.264, `-pix_fmt yuv420p` maximize compatibility
        # with bad video players.
        # Ref: https://trac.ffmpeg.org/wiki/Slideshow
        '-pix_fmt', 'yuv420p',
        # Native encoder cannot encode images of small scale
        # or the the hardware encoder may be busy which raises
        # Error: cannot create compression session
        # so allow software encoding
        # '-allow_sw', '1',
    ]
    video_path = utils.as_path(video_dir) / 'output.mp4'
    ffmpeg_args.append(os.fspath(video_path))
    _ffmpeg_run(ffmpeg_args)
    video_str = utils.get_base64(video_path.read_bytes())
  return (
      f'<video height="{image_feature.THUMBNAIL_SIZE}" width="175" '
      'controls loop autoplay muted playsinline>'
      f'<source src="data:video/mp4;base64,{video_str}"  type="video/mp4" >'
      '</video>'
  )


def _get_repr_html_gif(images: List[PilImage]) -> str:
  """Get the <img/> str."""

  def write_buff(buff):
    images[0].save(
        buff,
        format='GIF',
        save_all=True,
        append_images=images[1:],
        duration=1000 / _VISU_FRAMERATE,
        loop=0,
    )

  # Convert to base64
  gif_str = utils.get_base64(write_buff)
  return f'<img src="data:image/png;base64,{gif_str}" alt="Gif" />'
