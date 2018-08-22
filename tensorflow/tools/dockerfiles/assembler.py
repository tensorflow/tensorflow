# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Assemble common TF Dockerfiles from many parts.

This script constructs TF's Dockerfiles by aggregating partial
Dockerfiles. See README.md for usage examples.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import errno
import os
import os.path
import re
import shutil
import textwrap

from absl import app
from absl import flags
import cerberus
import yaml

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'dry_run', False, 'Do not actually generate Dockerfiles', short_name='n')

flags.DEFINE_string(
    'spec_file',
    './spec.yml',
    'Path to a YAML specification file',
    short_name='s')

flags.DEFINE_string(
    'output_dir',
    './dockerfiles', ('Path to an output directory for Dockerfiles. '
                      'Will be created if it doesn\'t exist.'),
    short_name='o')

flags.DEFINE_string(
    'partial_dir',
    './partials',
    'Path to a directory containing foo.partial.Dockerfile partial files.',
    short_name='p')

flags.DEFINE_boolean(
    'quiet_dry_run',
    True,
    'Do not print contents of dry run Dockerfiles.',
    short_name='q')

flags.DEFINE_boolean(
    'validate', True, 'Validate generated Dockerfiles', short_name='c')

# Schema to verify the contents of spec.yml with Cerberus.
# Must be converted to a dict from yaml to work.
# Note: can add python references with e.g.
# !!python/name:builtins.str
# !!python/name:__main__.funcname
SCHEMA_TEXT = """
header:
  type: string

partials:
  type: dict
  keyschema:
    type: string
  valueschema:
    type: dict
    schema:
      desc:
        type: string
      args:
        type: dict
        keyschema:
          type: string
        valueschema:
          anyof:
            - type: [ boolean, number, string ]
            - type: dict
              schema:
                 default:
                    type: [ boolean, number, string ]
                 desc:
                    type: string
                 options:
                    type: list
                    schema:
                       type: string

images:
  keyschema:
    type: string
  valueschema:
    type: dict
    schema:
      desc:
        type: string
      arg-defaults:
        type: list
        schema:
          anyof:
            - type: dict
              keyschema:
                type: string
                arg_in_use: true
              valueschema:
                type: string
            - type: string
              isimage: true
      create-dockerfile:
        type: boolean
      partials:
        type: list
        schema:
          anyof:
            - type: dict
              keyschema:
                type: string
                regex: image
              valueschema:
                type: string
                isimage: true
            - type: string
              ispartial: true
"""


class TfDockerValidator(cerberus.Validator):
  """Custom Cerberus validator for TF dockerfile spec.

  Note: Each _validate_foo function's docstring must end with a segment
  describing its own validation schema, e.g. "The rule's arguments are...". If
  you add a new validator, you can copy/paste that section.
  """

  def _validate_ispartial(self, ispartial, field, value):
    """Validate that a partial references an existing partial spec.

    Args:
      ispartial: Value of the rule, a bool
      field: The field being validated
      value: The field's value

    The rule's arguments are validated against this schema:
    {'type': 'boolean'}
    """
    if ispartial and value not in self.root_document.get('partials', dict()):
      self._error(field, '{} is not an existing partial.'.format(value))

  def _validate_isimage(self, isimage, field, value):
    """Validate that an image references an existing partial spec.

    Args:
      isimage: Value of the rule, a bool
      field: The field being validated
      value: The field's value

    The rule's arguments are validated against this schema:
    {'type': 'boolean'}
    """
    if isimage and value not in self.root_document.get('images', dict()):
      self._error(field, '{} is not an existing image.'.format(value))

  def _validate_arg_in_use(self, arg_in_use, field, value):
    """Validate that an arg references an existing partial spec's args.

    Args:
      arg_in_use: Value of the rule, a bool
      field: The field being validated
      value: The field's value

    The rule's arguments are validated against this schema:
    {'type': 'boolean'}
    """
    if arg_in_use:
      for partial in self.root_document.get('partials', dict()).values():
        if value in partial.get('args', tuple()):
          return

      self._error(field, '{} is not an arg used in any partial.'.format(value))


def build_partial_description(partial_spec):
  """Create the documentation lines for a specific partial.

  Generates something like this:

    # This is the partial's description, from spec.yml.
    # --build-arg ARG_NAME=argdefault
    #    this is one of the args.
    # --build-arg ANOTHER_ARG=(some|choices)
    #    another arg.

  Args:
    partial_spec: A dict representing one of the partials from spec.yml. Doesn't
      include the name of the partial; is a dict like { desc: ..., args: ... }.

  Returns:
    A commented string describing this partial.
  """

  # Start from linewrapped desc field
  lines = []
  wrapper = textwrap.TextWrapper(
      initial_indent='# ', subsequent_indent='# ', width=80)
  description = wrapper.fill(partial_spec.get('desc', '( no comments )'))
  lines.extend(['#', description])

  # Document each arg
  for arg, arg_data in partial_spec.get('args', dict()).items():
    # Wrap arg description with comment lines
    desc = arg_data.get('desc', '( no description )')
    desc = textwrap.fill(
        desc,
        initial_indent='#    ',
        subsequent_indent='#    ',
        width=80,
        drop_whitespace=False)

    # Document (each|option|like|this)
    if 'options' in arg_data:
      arg_options = ' ({})'.format('|'.join(arg_data['options']))
    else:
      arg_options = ''

    # Add usage sample
    arg_use = '# --build-arg {}={}{}'.format(arg,
                                             arg_data.get('default', '(unset)'),
                                             arg_options)
    lines.extend([arg_use, desc])

  return '\n'.join(lines)


def construct_contents(partial_specs, image_spec):
  """Assemble the dockerfile contents for an image spec.

  It assembles a concrete list of partial references into a single, large
  string.
  Also expands argument defaults, so that the resulting Dockerfile doesn't have
  to be configured with --build-arg=... every time. That is, any ARG directive
  will be updated with a new default value.

  Args:
    partial_specs: The dict from spec.yml["partials"].
    image_spec: One of the dict values from spec.yml["images"].

  Returns:
    A string containing a valid Dockerfile based on the partials listed in
    image_spec.
  """
  processed_partial_strings = []
  for partial_name in image_spec['partials']:
    # Apply image arg-defaults to existing arg defaults
    partial_spec = copy.deepcopy(partial_specs[partial_name])
    args = partial_spec.get('args', dict())
    for k_v in image_spec.get('arg-defaults', []):
      arg, value = list(k_v.items())[0]
      if arg in args:
        args[arg]['default'] = value

    # Read partial file contents
    filename = partial_spec.get('file', partial_name)
    partial_path = os.path.join(FLAGS.partial_dir,
                                '{}.partial.Dockerfile'.format(filename))
    with open(partial_path, 'r') as f_partial:
      partial_contents = f_partial.read()

    # Replace ARG FOO=BAR with ARG FOO=[new-default]
    for arg, arg_data in args.items():
      if 'default' in arg_data and arg_data['default']:
        default = '={}'.format(arg_data['default'])
      else:
        default = ''
      partial_contents = re.sub(r'ARG {}.*'.format(arg), 'ARG {}{}'.format(
          arg, default), partial_contents)

    # Store updated partial contents
    processed_partial_strings.append(partial_contents)

  # Join everything together
  return '\n'.join(processed_partial_strings)


def mkdir_p(path):
  """Create a directory and its parents, even if it already exists."""
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise


def construct_documentation(header, partial_specs, image_spec):
  """Assemble all of the documentation for a single dockerfile.

  Builds explanations of included partials and available build args.

  Args:
    header: The string from spec.yml["header"]; will be commented and wrapped.
    partial_specs: The dict from spec.yml["partials"].
    image_spec: The spec for the dockerfile being built.

  Returns:
    A string containing a commented header that documents the contents of the
    dockerfile.

  """
  # Comment and wrap header and image description
  commented_header = '\n'.join(
      [('# ' + l).rstrip() for l in header.splitlines()])
  commented_desc = '\n'.join(
      ['# ' + l for l in image_spec.get('desc', '').splitlines()])
  partial_descriptions = []

  # Build documentation for each partial in the image
  for partial in image_spec['partials']:
    # Copy partial data for default args unique to this image
    partial_spec = copy.deepcopy(partial_specs[partial])
    args = partial_spec.get('args', dict())

    # Overwrite any existing arg defaults
    for k_v in image_spec.get('arg-defaults', []):
      arg, value = list(k_v.items())[0]
      if arg in args:
        args[arg]['default'] = value

    # Build the description from new args
    partial_description = build_partial_description(partial_spec)
    partial_descriptions.append(partial_description)

  contents = [commented_header, '#', commented_desc] + partial_descriptions
  return '\n'.join(contents) + '\n'


def normalize_partial_args(partial_specs):
  """Normalize the shorthand form of a partial's args specification.

  Turns this:

    partial:
      args:
        SOME_ARG: arg_value

  Into this:

    partial:
       args:
         SOME_ARG:
            default: arg_value

  Args:
    partial_specs: The dict from spec.yml["partials"]. This dict is modified in
      place.

  Returns:
    The modified contents of partial_specs.

  """
  for _, partial in partial_specs.items():
    args = partial.get('args', dict())
    for arg, value in args.items():
      if not isinstance(value, dict):
        new_value = {'default': value}
        args[arg] = new_value

  return partial_specs


def flatten_args_references(image_specs):
  """Resolve all default-args in each image spec to a concrete dict.

  Turns this:

    example-image:
      arg-defaults:
        - MY_ARG: ARG_VALUE

    another-example:
      arg-defaults:
        - ANOTHER_ARG: ANOTHER_VALUE
        - example_image

  Into this:

    example-image:
      arg-defaults:
        - MY_ARG: ARG_VALUE

    another-example:
      arg-defaults:
        - ANOTHER_ARG: ANOTHER_VALUE
        - MY_ARG: ARG_VALUE

  Args:
    image_specs: A dict of image_spec dicts; should be the contents of the
      "images" key in the global spec.yaml. This dict is modified in place and
      then returned.

  Returns:
    The modified contents of image_specs.
  """
  for _, image_spec in image_specs.items():
    too_deep = 0
    while str in map(type, image_spec.get('arg-defaults', [])) and too_deep < 5:
      new_args = []
      for arg in image_spec['arg-defaults']:
        if isinstance(arg, str):
          new_args.extend(image_specs[arg]['arg-defaults'])
        else:
          new_args.append(arg)

      image_spec['arg-defaults'] = new_args
      too_deep += 1

  return image_specs


def flatten_partial_references(image_specs):
  """Resolve all partial references in each image spec to a concrete list.

  Turns this:

    example-image:
      partials:
        - foo

    another-example:
      partials:
        - bar
        - image: example-image
        - bat

  Into this:

    example-image:
      partials:
        - foo

    another-example:
      partials:
        - bar
        - foo
        - bat
  Args:
    image_specs: A dict of image_spec dicts; should be the contents of the
      "images" key in the global spec.yaml. This dict is modified in place and
      then returned.

  Returns:
    The modified contents of image_specs.
  """
  for _, image_spec in image_specs.items():
    too_deep = 0
    while dict in map(type, image_spec['partials']) and too_deep < 5:
      new_partials = []
      for partial in image_spec['partials']:
        if isinstance(partial, str):
          new_partials.append(partial)
        else:
          new_partials.extend(image_specs[partial['image']]['partials'])

      image_spec['partials'] = new_partials
      too_deep += 1

  return image_specs


def construct_dockerfiles(tf_spec):
  """Generate a mapping of {"cpu": <cpu dockerfile contents>, ...}.

  Args:
    tf_spec: The full spec.yml loaded as a python object.

  Returns:
    A string:string dict of short names ("cpu-devel") to Dockerfile contents.
  """
  names_to_contents = dict()
  image_specs = tf_spec['images']
  image_specs = flatten_partial_references(image_specs)
  image_specs = flatten_args_references(image_specs)
  partial_specs = tf_spec['partials']
  partial_specs = normalize_partial_args(partial_specs)

  for name, image_spec in image_specs.items():
    if not image_spec.get('create-dockerfile', True):
      continue
    documentation = construct_documentation(tf_spec['header'], partial_specs,
                                            image_spec)
    contents = construct_contents(partial_specs, image_spec)
    names_to_contents[name] = '\n'.join([documentation, contents])

  return names_to_contents


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unexpected command line args found: {}'.format(argv))

  with open(FLAGS.spec_file, 'r') as spec_file:
    tf_spec = yaml.load(spec_file)

  # Abort if spec.yaml is invalid
  if FLAGS.validate:
    schema = yaml.load(SCHEMA_TEXT)
    v = TfDockerValidator(schema)
    if not v.validate(tf_spec):
      print('>> ERROR: {} is an invalid spec! The errors are:'.format(
          FLAGS.spec_file))
      print(yaml.dump(v.errors, indent=2))
      exit(1)
  else:
    print('>> WARNING: Not validating {}'.format(FLAGS.spec_file))

  # Generate mapping of { "cpu-devel": "<cpu-devel dockerfile contents>", ... }
  names_to_contents = construct_dockerfiles(tf_spec)

  # Write each completed Dockerfile
  if not FLAGS.dry_run:
    print('>> Emptying destination dir "{}"'.format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir, ignore_errors=True)
    mkdir_p(FLAGS.output_dir)
  else:
    print('>> Skipping creation of {} (dry run)'.format(FLAGS.output_dir))
  for name, contents in names_to_contents.items():
    path = os.path.join(FLAGS.output_dir, name + '.Dockerfile')
    if FLAGS.dry_run:
      print('>> Skipping writing contents of {} (dry run)'.format(path))
      print(contents)
    else:
      mkdir_p(FLAGS.output_dir)
      print('>> Writing {}'.format(path))
      with open(path, 'w') as f:
        f.write(contents)


if __name__ == '__main__':
  app.run(main)
