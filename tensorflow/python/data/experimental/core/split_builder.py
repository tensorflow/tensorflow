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

"""Dataset generator code."""

import collections.abc
import contextlib
import itertools
import sys
import typing
from typing import Any, Callable, Dict, Iterator, Iterable, List, Optional, Tuple, Union

from absl import logging
import dataclasses

from tensorflow.data.experimental.core import features as features_lib
from tensorflow.data.experimental.core import file_adapters
from tensorflow.data.experimental.core import lazy_imports_lib
from tensorflow.data.experimental.core import splits as splits_lib
from tensorflow.data.experimental.core import tfrecords_writer
from tensorflow.data.experimental.core import utils
from tensorflow.data.experimental.core.utils import type_utils

if typing.TYPE_CHECKING:
  import apache_beam as beam  # pytype: disable=import-error

# Example key used for shuffling
Key = Union[str, int]
# The nested example dict passed to `features.encode_example`
Example = Dict[str, Any]
KeyExample = Tuple[Key, Example]

# Possible values returned by `GeneratorBasedBuilder._split_generators`
SplitGenerator = Union[
    Iterable[KeyExample],
    # Ideally we should add input/output type annotations
    # `beam.PTransform[[], KeyExample]`, similar to `Callable[[], KeyExample]`
    'beam.PTransform',
    'beam.PCollection[KeyExample]',
]


@utils.docs.deprecated
@dataclasses.dataclass
class SplitGeneratorLegacy:
  """Defines the split information for the generator.

  DEPRECATED: `_split_generators` should return `dict<split_name, generators>`
  instead. See the
  [documentation](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/GeneratorBasedBuilder).

  Attributes:
    name: `str`, name of the Split for which the generator will create the
      examples.
    gen_kwargs: `dict`, kwargs to forward to the _generate_examples() method of
      the builder.
  """
  name: str
  gen_kwargs: Dict[str, Any]


class _SplitInfoFuture:
  """Future containing the `tfds.core.SplitInfo` result."""

  def __init__(self, callback: Callable[[], splits_lib.SplitInfo]):
    self._callback = callback

  def result(self) -> splits_lib.SplitInfo:
    return self._callback()


class SplitBuilder:
  """Util class to build splits.

  Usage is as follow:

  ```py
  split_builder = SplitBuilder(...)

  with split_builder.maybe_beam_pipeline():
    split_info_future = split_builder.submit_split_generation(...)

  split_info = split_info_future.result()
  ```

  * submit_split_generation:
    * For generator based split: Generate the split
    * For Apache Beam based split: Create the `beam.PCollection` and returns
      a future.
  * `split_info_future.result()`: Called after all `beam.PCollection`s have
    finished. Finalize the `split_info` by collecting all pipeline results.

  `submit_split_generation` / `.result` should be called once per
  split.
  """

  def __init__(
      self,
      *,
      split_dict: splits_lib.SplitDict,  # Used for precomputed nb of examples
      features: features_lib.FeatureConnector,
      beam_options: Optional['beam.options.pipeline_options.PipelineOptions'],
      beam_runner: Optional['beam.runners.PipelineRunner'],
      max_examples_per_split: Optional[int],
      file_format: file_adapters.FileFormat = file_adapters.DEFAULT_FILE_FORMAT,
  ):
    self._split_dict = split_dict
    self._features = features
    self._max_examples_per_split = max_examples_per_split

    self._in_contextmanager: bool = False
    self._beam_options = beam_options
    self._beam_runner = beam_runner
    self._beam_pipeline: Optional['beam.Pipeline'] = None
    self._file_format = file_format

  @contextlib.contextmanager
  def maybe_beam_pipeline(self) -> Iterator[None]:
    """Context manager wrapping the beam pipeline.

    If Apache Beam is used, then the pipeline created withing the contextmanager
    will be launched when exiting the context manager:

    ```py
    with split_builder.maybe_beam_pipeline():
      pcollection = (
          split_builder.beam_pipeline
          | beam.Create()
          | beam.Map()
      )
    ```

    Is equivalent to:

    ```py
    with beam.Pipeline() as beam_pipeline:
      pcollection = (
          beam_pipeline
          | beam.Create()
          | beam.Map()
      )
    ```

    If `split_builder.beam_pipeline` is never called, then `beam.Pipeline` is
    never created and this function is a no-op.

    Yields:
      None
    """
    self._in_contextmanager = True
    try:
      # Entering the contextmanager is a no-op. Only if Apache Beam is used
      # is the `beam.Pipeline` contextmanager activated.
      yield
    except Exception:  # pylint: disable=broad-except
      # Close and forward the exception
      if (
          not self._beam_pipeline
          or not self._beam_pipeline.__exit__(*sys.exc_info())
      ):
        raise  # Forward the exception
    else:
      # If the Beam pipeline was used, then exit it.
      if self._beam_pipeline is not None:
        self._beam_pipeline.__exit__(None, None, None)
    self._in_contextmanager = False

  @utils.memoized_property
  def beam_pipeline(self) -> 'beam.Pipeline':
    """Instanciates and returns Apache Beam pipeline.

    Calling this function starts the Apache Beam mode.

    Returns:
      pipeline: The beam pipeline
    """
    if not self._in_contextmanager:
      raise AssertionError(
          'beam_pipeline has to be created from within `SplitBuilder` '
          'contextmanager.'
      )

    beam = lazy_imports_lib.lazy_imports.apache_beam

    # On Colab, stderr isn't displayed by default, so using `print`.
    print_fn = print if utils.is_notebook() else logging.warning
    if not self._beam_runner and not self._beam_options:
      msg = utils.dedent(
          """
          **************************** WARNING *********************************
          Warning: The dataset you're trying to generate is using Apache Beam,
          yet no `beam_runner` nor `beam_options` was explicitly provided.

          Some Beam datasets take weeks to generate, so are usually not suited
          for single machine generation. Please have a look at the instructions
          to setup distributed generation:

          https://www.tensorflow.org/datasets/beam_datasets#generating_a_beam_dataset
          **********************************************************************
          """
      )
      print_fn(msg)

    beam_options = (
        self._beam_options or beam.options.pipeline_options.PipelineOptions()
    )
    # Beam type checking assumes transforms multiple outputs are of same type,
    # which is not our case. Plus it doesn't handle correctly all types, so we
    # are better without it.
    beam_options.view_as(
        beam.options.pipeline_options.TypeOptions
    ).pipeline_type_check = False
    # Create the global pipeline object common for all splits
    pipeline = beam.Pipeline(runner=self._beam_runner, options=beam_options)
    self._beam_pipeline = pipeline.__enter__()
    return self._beam_pipeline

  def normalize_legacy_split_generators(
      self,
      split_generators: Union[
          Dict[str, SplitGenerator], List[SplitGeneratorLegacy]
      ],
      generator_fn: Callable[..., Any],
      is_beam: bool,
  ) -> Dict[str, SplitGenerator]:
    """Normalize legacy split API into new dict[split_name, generator].

    This function convert the legacy `List[tfds.core.SplitGenerator]` into
    the new `{'split_name': generator}` structure.

    Could be removed if all datasets were updated.

    Args:
      split_generators: Either legacy or new split_generators
      generator_fn: The `GeneratorBasedBuilder._generate_examples` function.
      is_beam: `True` if using legacy `tfds.core.BeamBasedBuilder`

    Returns:
      split_generators: New split generator structure.
    """
    if isinstance(split_generators, dict):  # New structure
      return split_generators
    if isinstance(split_generators, list):  # Legacy structure
      if is_beam:  # Legacy `tfds.core.BeamBasedBuilder`
        beam = lazy_imports_lib.lazy_imports.apache_beam
        generator_fn = beam.ptransform_fn(generator_fn)
        return {
            s.name: generator_fn(**s.gen_kwargs)  # Create the `beam.PTransform`
            for s in split_generators
        }
      else:
        return {
            split_generator.name: generator_fn(**split_generator.gen_kwargs)
            for split_generator in split_generators
        }
    else:
      raise TypeError(
          f'Invalid `_split_generators` returned value: {split_generators}'
      )

  def submit_split_generation(
      self,
      split_name: str,
      generator: SplitGenerator,
      path: type_utils.PathLike,
  ) -> _SplitInfoFuture:
    """Start the split generation.

    Args:
      split_name: Name of the split to generate
      generator: Generator, beam.PTransform,... yielding the examples
      path: path where the split should be saved

    Returns:
      split_info_future: Future containing the `split_info`, once generation
        is complete. The `tfds.core.SplitInfo` can be accessed through
        `split_info_future.result()`
    """
    build_kwargs = dict(split_name=split_name, generator=generator, path=path)
    # Depending on the type of generator, we use the corresponding
    # `_build_from_xyz` method.
    if isinstance(generator, collections.abc.Iterable):
      return self._build_from_generator(**build_kwargs)
    else:  # Otherwise, beam required
      unknown_generator_type = TypeError(
          f'Invalid split generator value for split `{split_name}`. '
          'Expected generator or apache_beam object. Got: '
          f'{type(generator)}'
      )
      try:
        import apache_beam as beam  # pylint: disable=g-import-not-at-top
      except ImportError:
        # Beam can't be imported, what was the object returned by the user ?
        raise unknown_generator_type
      if isinstance(generator, beam.PTransform):
        # Generate the beam.PCollection
        pcollection = self.beam_pipeline | split_name >> generator
        build_kwargs['generator'] = pcollection
        return self._build_from_pcollection(**build_kwargs)
      elif isinstance(generator, beam.PCollection):
        return self._build_from_pcollection(**build_kwargs)
      else:
        raise unknown_generator_type

  def _build_from_generator(
      self,
      split_name: str,
      generator: Iterable[KeyExample],
      path: type_utils.PathLike,
  ) -> _SplitInfoFuture:
    """Split generator for example generators.

    Args:
      split_name: str,
      generator: Iterable[KeyExample],
      path: type_utils.PathLike,

    Returns:
      future: The future containing the `tfds.core.SplitInfo`.
    """
    if self._max_examples_per_split is not None:
      logging.warning(
          'Splits capped at %s examples max.', self._max_examples_per_split
      )
      generator = itertools.islice(generator, self._max_examples_per_split)
      total_num_examples = self._max_examples_per_split
    else:
      # If dataset info has been pre-downloaded from the internet,
      # we can use the pre-computed number of example for the progression bar.
      split_info = self._split_dict.get(split_name)
      if split_info and split_info.num_examples:
        total_num_examples = split_info.num_examples
      else:
        total_num_examples = None

    writer = tfrecords_writer.Writer(
        example_specs=self._features.get_serialized_info(),
        path=path,
        hash_salt=split_name,
        file_format=self._file_format,
    )
    for key, example in utils.tqdm(
        generator, unit=' examples', total=total_num_examples, leave=False
    ):
      try:
        example = self._features.encode_example(example)
      except Exception as e:  # pylint: disable=broad-except
        utils.reraise(e, prefix=f'Failed to encode example:\n{example}\n')
      writer.write(key, example)
    shard_lengths, total_size = writer.finalize()

    split_info = splits_lib.SplitInfo(
        name=split_name,
        shard_lengths=shard_lengths,
        num_bytes=total_size,
    )
    return _SplitInfoFuture(lambda: split_info)

  def _build_from_pcollection(
      self,
      split_name: str,
      generator: 'beam.PCollection[KeyExample]',
      path: type_utils.PathLike,
  ) -> _SplitInfoFuture:
    """Split generator for `beam.PCollection`."""
    # TODO(tfds): Should try to add support to `max_examples_per_split`
    beam = lazy_imports_lib.lazy_imports.apache_beam

    beam_writer = tfrecords_writer.BeamWriter(
        example_specs=self._features.get_serialized_info(),
        path=path,
        hash_salt=split_name,
        file_format=self._file_format,
    )

    def _encode_example(key_ex, encode_fn=self._features.encode_example):
      # We do not access self._features in this function to avoid pickling the
      # entire class.
      return key_ex[0], encode_fn(key_ex[1])

    # Note: We need to wrap the pipeline in a PTransform to avoid
    # errors due to duplicated ``>> beam_labels`
    @beam.ptransform_fn
    def _encode_pcollection(pipeline):
      """PTransformation which build a single split."""
      pcoll_examples = pipeline | 'Encode' >> beam.Map(_encode_example)
      return beam_writer.write_from_pcollection(pcoll_examples)

    # Add the PCollection to the pipeline
    _ = generator | f'{split_name}_write' >> _encode_pcollection()  # pylint: disable=no-value-for-parameter

    def _resolve_future():
      if self._in_contextmanager:
        raise AssertionError(
            '`future.result()` should be called after the '
            '`maybe_beam_pipeline` contextmanager.'
        )
      logging.info('Retrieving split info for %s...', split_name)
      shard_lengths, total_size = beam_writer.finalize()
      return splits_lib.SplitInfo(
          name=split_name,
          shard_lengths=shard_lengths,
          num_bytes=total_size,
      )

    return _SplitInfoFuture(_resolve_future)
