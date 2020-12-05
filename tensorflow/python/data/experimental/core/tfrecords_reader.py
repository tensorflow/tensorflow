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

"""Defined Reader and ReadInstruction to read tfrecord files."""

import functools
import math
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Sequence, Union

import attr

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.data.experimental.core import example_parser
from tensorflow.data.experimental.core import file_adapters
from tensorflow.data.experimental.core import naming
from tensorflow.data.experimental.core import utils
from tensorflow.data.experimental.core.utils import read_config as read_config_lib
from tensorflow.data.experimental.core.utils import shard_utils

SplitInfo = Any

# Should be: Callable[[Tensor], Nested[Tensor]]
ParseFn = Callable[[Any], Any]


_BUFFER_SIZE = 8<<20  # 8 MiB per file.

_SUB_SPEC_RE = re.compile(r'''
^
 (?P<split>[\w-]+)
 (\[
  ((?P<from>-?\d+)
   (?P<from_pct>%)?)?
  :
  ((?P<to>-?\d+)
   (?P<to_pct>%)?)?
 \])?
$
''', re.X)

_ADDITION_SEP_RE = re.compile(r'\s*\+\s*')


def _get_dataset_from_filename(filename_skip_take, do_skip, do_take,
                               file_format):
  """Returns a tf.data.Dataset instance from given (filename, skip, take)."""
  filename, skip, take = (filename_skip_take['filename'],
                          filename_skip_take['skip'],
                          filename_skip_take['take'],)

  # Explictly use DatasetV1 for backward compatibility:
  # * isinstance(ds, tf.data.Dataset)
  # * ds.make_one_shot_iterator()
  ds = file_adapters.ADAPTER_FOR_FORMAT[file_format].make_tf_data(
      filename, _BUFFER_SIZE)
  if do_skip:
    ds = ds.skip(skip)
  if do_take:
    ds = ds.take(take)
  return ds


def make_file_instructions(
    name: str,
    split_infos: Iterable[SplitInfo],
    instruction: Union['ReadInstruction', str],
    file_format: file_adapters.FileFormat = file_adapters.DEFAULT_FILE_FORMAT,
) -> List[shard_utils.FileInstruction]:
  """Returns instructions of the split dict.

  Args:
    name: Name of the dataset.
    split_infos: Dataset splits information
    instruction: `ReadInstruction` or `str`
    file_format: Format of the record files in which the dataset
      will be read/written from.

  Returns:
    file_intructions: FileInstructions instance
  """
  name2shard_lengths = {
      info.name: info.shard_lengths for info in split_infos
  }
  name2len = {
      name: sum(lengths) for name, lengths in name2shard_lengths.items()
  }
  if not isinstance(instruction, ReadInstruction):
    instruction = ReadInstruction.from_spec(instruction)
  # Create the absolute instruction (per split)
  absolute_instructions = instruction.to_absolute(name2len)

  return _make_file_instructions_from_absolutes(
      name=name,
      name2shard_lengths=name2shard_lengths,
      absolute_instructions=absolute_instructions,
      file_format=file_format,
  )


def _make_file_instructions_from_absolutes(
    name: str,
    name2shard_lengths: Dict[str, List[int]],
    absolute_instructions: 'ReadInstruction',
    file_format: file_adapters.FileFormat = file_adapters.DEFAULT_FILE_FORMAT,
) -> List[shard_utils.FileInstruction]:
  """Returns the files instructions from the absolute instructions list."""
  # For each split, return the files instruction (skip/take)
  file_instructions = []
  for abs_instr in absolute_instructions:
    shard_lengths = name2shard_lengths[abs_instr.splitname]
    if not shard_lengths:
      raise ValueError(
          'Shard empty. This might means that dataset hasn\'t been generated '
          'yet and info not restored from GCS, or that legacy dataset is used.')
    filenames = naming.filenames_for_dataset_split(
        dataset_name=name,
        split=abs_instr.splitname,
        num_shards=len(shard_lengths),
        filetype_suffix=file_adapters.ADAPTER_FOR_FORMAT[file_format]
        .FILE_SUFFIX)
    from_ = 0 if abs_instr.from_ is None else abs_instr.from_
    to = sum(shard_lengths) if abs_instr.to is None else abs_instr.to
    single_file_instructions = shard_utils.get_file_instructions(
        from_, to, filenames, shard_lengths)
    file_instructions.extend(single_file_instructions)
  return file_instructions


def _read_files(
    file_instructions: Sequence[shard_utils.FileInstruction],
    parse_fn: ParseFn,
    read_config: read_config_lib.ReadConfig,
    shuffle_files: bool,
    file_format: file_adapters.FileFormat,
) -> tf.data.Dataset:
  """Returns tf.data.Dataset for given file instructions.

  Args:
    file_instructions: the information on the files to read including
      `ds.skip().take()`
    parse_fn: function used to parse each record.
    read_config: Additional options to configure the
      input pipeline (e.g. seed, num parallel reads,...).
    shuffle_files: Defaults to False. True to shuffle input files.
    file_format: Format of the record files in which the dataset
      will be read/written from.

  Returns:
    The dataset object.
  """
  # Eventually apply a transformation to the instruction function.
  # This allow the user to have direct control over the interleave order.
  if read_config.experimental_interleave_sort_fn is not None:
    file_instructions = read_config.experimental_interleave_sort_fn(
        file_instructions)

  do_skip = any(f.skip > 0 for f in file_instructions)
  do_take = any(f.take > -1 for f in file_instructions)

  # Transpose the list[dict] into dict[list]
  tensor_inputs = {
      # skip/take need to be converted to int64 explicitly
      k: list(vals) if k == 'filename' else np.array(vals, dtype=np.int64)
      for k, vals in utils.zip_dict(*[f.asdict() for f in file_instructions])
  }

  cycle_length = read_config.interleave_cycle_length
  block_length = read_config.interleave_block_length

  instruction_ds = tf.data.Dataset.from_tensor_slices(tensor_inputs)

  # On distributed environments, we can shard per-file if a
  # `tf.distribute.InputContext` object is provided (e.g. from
  # `experimental_distribute_datasets_from_function`)
  if (read_config.input_context and
      read_config.input_context.num_input_pipelines > 1):
    if len(file_instructions) < read_config.input_context.num_input_pipelines:
      raise ValueError(
          'Cannot shard the pipeline with given `input_context`.'
          '`num_shards={}` but `num_input_pipelines={}`. '
          'This means that some workers won\'t read any data. '
          'To shard the data, you may want to use the subsplit API '
          'instead: https://www.tensorflow.org/datasets/splits'.format(
              len(file_instructions),
              read_config.input_context.num_input_pipelines)
      )
    instruction_ds = instruction_ds.shard(
        num_shards=read_config.input_context.num_input_pipelines,
        index=read_config.input_context.input_pipeline_id,
    )

  # If shuffle is True, we shuffle the instructions/shards
  if shuffle_files:
    instruction_ds = instruction_ds.shuffle(
        len(file_instructions),
        seed=read_config.shuffle_seed,
        reshuffle_each_iteration=read_config.shuffle_reshuffle_each_iteration,
    )

  ds = instruction_ds.interleave(
      functools.partial(
          _get_dataset_from_filename,
          do_skip=do_skip,
          do_take=do_take,
          file_format=file_format),
      cycle_length=cycle_length,
      block_length=block_length,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )

  # If the number of examples read in the tf-record is known, we forward
  # the information to the tf.data.Dataset object.
  # Check the `tf.data.experimental` for backward compatibility with TF <= 2.1
  if (not read_config.input_context and  # TODO(epot): Restore cardinality
      hasattr(tf.data.experimental, 'assert_cardinality')):
    # TODO(b/154963426): Replace by per-shard cardinality (warning if
    # `experimental_interleave_sort_fn` is set).
    cardinality = sum(f.num_examples for f in file_instructions)
    ds = ds.apply(tf.data.experimental.assert_cardinality(cardinality))

  ds = ds.with_options(read_config.options)  # Additional users options

  # TODO(pierrot): `parse_example` uses
  # `tf.io.parse_single_example`. It might be faster to use `parse_example`,
  # after batching.
  # https://www.tensorflow.org/api_docs/python/tf/io/parse_example
  return ds.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


class Reader(object):
  """Build a tf.data.Dataset object out of Instruction instance(s).

  This class should not typically be exposed to the TFDS user.
  """

  def __init__(self,
               path,
               example_specs,
               file_format=file_adapters.DEFAULT_FILE_FORMAT):
    """Initializes Reader.

    Args:
      path (str): path where tfrecords are stored.
      example_specs: spec to build ExampleParser.
      file_format: file_adapters.FileFormat, format of the record files in
        which the dataset will be read/written from.
    """
    self._path = path
    self._parser = example_parser.ExampleParser(example_specs)
    self._file_format = file_format

  def read(
      self,
      name,
      instructions,
      split_infos,
      read_config,
      shuffle_files,
  ):
    """Returns tf.data.Dataset instance(s).

    Args:
      name (str): name of the dataset.
      instructions (ReadInstruction, List[], Dict[]): instruction(s) to read.
        Instructions can be string and will then be passed to the Instruction
        constructor as it.
      split_infos (list of SplitInfo proto): the available splits for dataset.
      read_config: `tfds.ReadConfig`, the input pipeline options
      shuffle_files (bool): If True, input files are shuffled before being read.

    Returns:
       a single tf.data.Dataset instance if instruction is a single
       ReadInstruction instance. Otherwise a dict/list of tf.data.Dataset
       corresponding to given instructions param shape.
    """
    def _read_instruction_to_ds(instruction):
      file_instructions = make_file_instructions(
          name, split_infos, instruction, file_format=self._file_format)
      return self.read_files(
          file_instructions,
          read_config=read_config,
          shuffle_files=shuffle_files,
      )

    return tf.nest.map_structure(_read_instruction_to_ds, instructions)

  def read_files(
      self,
      file_instructions: Sequence[shard_utils.FileInstruction],
      read_config: read_config_lib.ReadConfig,
      shuffle_files: bool,
  ) -> tf.data.Dataset:
    """Returns single tf.data.Dataset instance for the set of file instructions.

    Args:
      file_instructions: The files information.
        The filenames contains the relative path, not absolute.
        skip/take indicates which example read in the shard: `ds.skip().take()`
      read_config: The input pipeline options
      shuffle_files: If True, input files are shuffled before being read.

    Returns:
       a tf.data.Dataset instance.
    """
    if not file_instructions:
      msg = 'Instruction {} corresponds to no data!'.format(
          file_instructions)
      raise AssertionError(msg)

    # Prepend path to filename
    file_instructions = [
        f.replace(filename=os.path.join(self._path, f.filename))
        for f in file_instructions
    ]
    ds = _read_files(
        file_instructions=file_instructions,
        read_config=read_config,
        parse_fn=self._parser.parse_example,
        shuffle_files=shuffle_files,
        file_format=self._file_format,
    )
    return ds


@attr.s(frozen=True)
class _AbsoluteInstruction(object):
  """A machine friendly slice: defined absolute positive boundaries."""
  splitname = attr.ib()  # : str
  from_ = attr.ib()  # uint (starting index).
  to = attr.ib()  # uint (ending index).


@attr.s(frozen=True)
class _RelativeInstruction(object):
  """Represents a single parsed slicing instruction, can use % and negatives."""
  splitname = attr.ib()  # : str
  # starting index, or None if no lower boundary.
  from_ = attr.ib()  # : Optional[int]
  # ending index, or None if no upper boundary.
  to = attr.ib()  # : Optional[int]
  unit = attr.ib(validator=attr.validators.in_(['%', 'abs']))  # : str
  rounding = attr.ib(validator=attr.validators.in_([  # : str
      'closest', 'pct1_dropremainder']))

  @from_.validator
  @to.validator
  def check_boundary_pct(self, unused_attribute, value):
    if self.unit == '%' and value is not None and abs(value) > 100:
      raise AssertionError('Percent slice boundaries must be > -100 and < 100.')


def _str_to_relative_instruction(spec):
  """Returns ReadInstruction for given string."""
  res = _SUB_SPEC_RE.match(spec)
  if not res:
    raise AssertionError('Unrecognized instruction format: %s' % spec)
  unit = '%' if res.group('from_pct') or res.group('to_pct') else 'abs'
  return ReadInstruction(
      split_name=res.group('split'),
      rounding='closest',
      from_=int(res.group('from')) if res.group('from') else None,
      to=int(res.group('to')) if res.group('to') else None,
      unit=unit,
      )


def _pct_to_abs_pct1(boundary, num_examples):
  # Using math.trunc here, since -99.5% should give -99%, not -100%.
  if num_examples < 100:
    msg = ('Using "pct1_dropremainder" rounding on a split with less than 100 '
           'elements is forbidden: it always results in an empty dataset.')
    raise AssertionError(msg)
  return boundary * math.trunc(num_examples / 100.)


def _pct_to_abs_closest(boundary, num_examples):
  return int(round(boundary * num_examples / 100.))


def _rel_to_abs_instr(rel_instr, name2len):
  """Returns _AbsoluteInstruction instance for given RelativeInstruction.

  Args:
    rel_instr: RelativeInstruction instance.
    name2len: dict {split_name: num_examples}.
  """
  pct_to_abs = (_pct_to_abs_closest if rel_instr.rounding == 'closest'
                else _pct_to_abs_pct1)
  split = rel_instr.splitname
  if split not in name2len:
    raise ValueError('Unknown split "{}". Should be one of {}.'.format(
        split, list(name2len)))
  num_examples = name2len[split]
  from_ = rel_instr.from_
  to = rel_instr.to
  if rel_instr.unit == '%':
    from_ = 0 if from_ is None else pct_to_abs(from_, num_examples)
    to = num_examples if to is None else pct_to_abs(to, num_examples)
  else:
    from_ = 0 if from_ is None else from_
    to = num_examples if to is None else to
  if abs(from_) > num_examples or abs(to) > num_examples:
    msg = 'Requested slice [%s:%s] incompatible with %s examples.' % (
        from_ or '', to or '', num_examples)
    raise AssertionError(msg)
  if from_ < 0:
    from_ = num_examples + from_
  elif from_ == 0:
    from_ = None
  if to < 0:
    to = num_examples + to
  elif to == num_examples:
    to = None
  return _AbsoluteInstruction(split, from_, to)


class ReadInstruction(object):
  """Reading instruction for a dataset.

  Note: Due to the shards being read in parallel, order isn't guaranteed to be
  consistent between sub-splits. In other words reading `test[0:100]` followed
  by `test[100:200]` may yield examples in a different order than reading
  `test[:200]`.

  Examples of usage:

  ```
  # The following lines are equivalent:
  ds = tfds.load('mnist', split='test[:33%]')
  ds = tfds.load('mnist', split=tfds.core.ReadInstruction.from_spec(
      'test[:33%]'))
  ds = tfds.load('mnist', split=tfds.core.ReadInstruction(
      'test', to=33, unit='%'))
  ds = tfds.load('mnist', split=tfds.core.ReadInstruction(
      'test', from_=0, to=33, unit='%'))

  # The following lines are equivalent:
  ds = tfds.load('mnist', split='test[:33%]+train[1:-1]')
  ds = tfds.load('mnist', split=tfds.core.ReadInstruction.from_spec(
      'test[:33%]+train[1:-1]'))
  ds = tfds.load('mnist', split=(
      tfds.core.ReadInstruction('test', to=33, unit='%') +
      tfds.core.ReadInstruction('train', from_=1, to=-1, unit='abs')))

  # 10-fold validation:
  tests = tfds.load(
      'mnist',
      [tfds.core.ReadInstruction('train', from_=k, to=k+10, unit='%')
       for k in range(0, 100, 10)])
  trains = tfds.load(
      'mnist',
      [tfds.core.ReadInstruction('train', to=k, unit='%') +
       tfds.core.ReadInstruction('train', from_=k+10, unit='%')
       for k in range(0, 100, 10)])
  ```

  """

  def _init(self, relative_instructions):
    # Private initializer.
    self._relative_instructions = relative_instructions

  @classmethod
  def _read_instruction_from_relative_instructions(cls, relative_instructions):
    """Returns ReadInstruction obj initialized with relative_instructions."""
    # Use __new__ to bypass __init__ used by public API and not conveniant here.
    result = cls.__new__(cls)
    result._init(relative_instructions)  # pylint: disable=protected-access
    return result

  def __init__(
      self,
      split_name,
      *,
      rounding='closest',
      from_=None,
      to=None,
      unit=None,
  ):
    """Initialize ReadInstruction.

    Args:
      split_name (str): name of the split to read. Eg: 'train'.
      rounding (str): The rounding behaviour to use when percent slicing is
        used. Ignored when slicing with absolute indices.
        Possible values:
         - 'closest' (default): The specified percentages are rounded to the
           closest value. Use this if you want specified percents to be as
           much exact as possible.
         - 'pct1_dropremainder': the specified percentages are treated as
           multiple of 1%. Use this option if you want consistency. Eg:
             len(5%) == 5 * len(1%).
           Using this option, one might not be able to use the full set of
           examples, if the number of those is not a multiple of 100.
      from_ (int):
      to (int): alternative way of specifying slicing boundaries. If any of
        {from_, to, unit} argument is used, slicing cannot be specified as
        string.
      unit (str): optional, one of:
        '%': to set the slicing unit as percents of the split size.
        'abs': to set the slicing unit as absolute numbers.
    """
    # Unit is optional only if the full dataset is read, otherwise, will
    # `_RelativeInstruction` validator will fail.
    if from_ is None and to is None and unit is None:
      unit = '%'
    # This constructor is not always called. See factory method
    # `_read_instruction_from_relative_instructions`. Common init instructions
    # MUST be placed in the _init method.
    self._init(
        [_RelativeInstruction(split_name, from_, to, unit, rounding)])

  @classmethod
  def from_spec(cls, spec):
    """Creates a ReadInstruction instance out of a string spec.

    Args:
      spec (str): split(s) + optional slice(s) to read. A slice can be
            specified, using absolute numbers (int) or percentages (int). E.g.
              `test`: test split.
              `test + validation`: test split + validation split.
              `test[10:]`: test split, minus its first 10 records.
              `test[:10%]`: first 10% records of test split.
              `test[:-5%]+train[40%:60%]`: first 95% of test + middle 20% of
                                           train.

    Returns:
      ReadInstruction instance.
    """
    spec = str(spec)  # Need to convert to str in case of `Split` instance.
    subs = _ADDITION_SEP_RE.split(spec)
    if not subs:
      raise AssertionError('No instructions could be built out of %s' % spec)
    instruction = _str_to_relative_instruction(subs[0])
    return sum([_str_to_relative_instruction(sub) for sub in subs[1:]],
               instruction)

  def __add__(self, other):
    """Returns a new ReadInstruction obj, result of appending other to self."""
    if not isinstance(other, ReadInstruction):
      msg = 'ReadInstruction can only be added to another ReadInstruction obj.'
      raise AssertionError(msg)
    other_ris = other._relative_instructions  # pylint: disable=protected-access
    if self._relative_instructions[0].rounding != other_ris[0].rounding:
      raise AssertionError('It is forbidden to sum ReadInstruction instances '
                           'with different rounding values.')
    return self._read_instruction_from_relative_instructions(
        self._relative_instructions + other_ris)

  def __str__(self):
    return 'ReadInstruction(%s)' % self._relative_instructions

  def to_absolute(self, name2len):
    """Translate instruction into a list of absolute instructions.

    Those absolute instructions are then to be added together.

    Args:
      name2len: dict associating split names to number of examples.

    Returns:
      list of _AbsoluteInstruction instances (corresponds to the + in spec).
    """
    return [_rel_to_abs_instr(rel_instr, name2len)
            for rel_instr in self._relative_instructions]
