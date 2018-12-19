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
# ============================================================================
"""Multipurpose TensorFlow Docker Helper.

- Assembles Dockerfiles
- Builds images (and optionally runs image tests)
- Pushes images to Docker Hub (provided with credentials)

Logs are written to stderr; the list of successfully built images is
written to stdout.

Read README.md (in this directory) for instructions!
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import errno
import itertools
import multiprocessing
import os
import re
import shutil
import sys

from absl import app
from absl import flags
import cerberus
import docker
import yaml

FLAGS = flags.FLAGS

flags.DEFINE_string('hub_username', None,
                    'Dockerhub username, only used with --upload_to_hub')

flags.DEFINE_string(
    'hub_password', None,
    ('Dockerhub password, only used with --upload_to_hub. Use from an env param'
     ' so your password isn\'t in your history.'))

flags.DEFINE_integer('hub_timeout', 3600,
                     'Abort Hub upload if it takes longer than this.')

flags.DEFINE_string(
    'repository', 'tensorflow',
    'Tag local images as {repository}:tag (in addition to the '
    'hub_repository, if uploading to hub)')

flags.DEFINE_string(
    'hub_repository', None,
    'Push tags to this Docker Hub repository, e.g. tensorflow/tensorflow')

flags.DEFINE_boolean(
    'upload_to_hub',
    False,
    ('Push built images to Docker Hub (you must also provide --hub_username, '
     '--hub_password, and --hub_repository)'),
    short_name='u',
)

flags.DEFINE_boolean(
    'construct_dockerfiles', False, 'Do not build images', short_name='d')

flags.DEFINE_boolean(
    'keep_temp_dockerfiles',
    False,
    'Retain .temp.Dockerfiles created while building images.',
    short_name='k')

flags.DEFINE_boolean(
    'build_images', False, 'Do not build images', short_name='b')

flags.DEFINE_string(
    'run_tests_path', None,
    ('Execute test scripts on generated Dockerfiles before pushing them. '
     'Flag value must be a full path to the "tests" directory, which is usually'
     ' $(realpath ./tests). A failed tests counts the same as a failed build.'))

flags.DEFINE_string(
    'write_tags_to', None,
    'Write the list of tagged images to a file. Useful for parallelizing tests.'
)

flags.DEFINE_boolean(
    'stop_on_failure', False,
    ('Stop processing tags if any one build fails. If False or not specified, '
     'failures are reported but do not affect the other images.'))

flags.DEFINE_boolean(
    'dry_run',
    False,
    'Do not build or deploy anything at all.',
    short_name='n',
)

flags.DEFINE_string(
    'exclude_tags_matching',
    None,
    ('Regular expression that skips processing on any tag it matches. Must '
     'match entire string, e.g. ".*gpu.*" ignores all GPU tags.'),
    short_name='x')

flags.DEFINE_string(
    'only_tags_matching',
    None,
    ('Regular expression that skips processing on any tag it does not match. '
     'Must match entire string, e.g. ".*gpu.*" includes only GPU tags.'),
    short_name='i')

flags.DEFINE_string(
    'dockerfile_dir',
    './dockerfiles', 'Path to an output directory for Dockerfiles.'
    ' Will be created if it doesn\'t exist.'
    ' Existing files in this directory will be deleted when new Dockerfiles'
    ' are made.',
    short_name='o')

flags.DEFINE_string(
    'partial_dir',
    './partials',
    'Path to a directory containing foo.partial.Dockerfile partial files.'
    ' can have subdirectories, e.g. "bar/baz.partial.Dockerfile".',
    short_name='p')

flags.DEFINE_multi_string(
    'release', [],
    'Set of releases to build and tag. Defaults to every release type.',
    short_name='r')

flags.DEFINE_multi_string(
    'arg', [],
    ('Extra build arguments. These are used for expanding tag names if needed '
     '(e.g. --arg _TAG_PREFIX=foo) and for using as build arguments (unused '
     'args will print a warning).'),
    short_name='a')

flags.DEFINE_boolean(
    'nocache', False,
    'Disable the Docker build cache; identical to "docker build --no-cache"')

flags.DEFINE_string(
    'spec_file',
    './spec.yml',
    'Path to the YAML specification file',
    short_name='s')

# Schema to verify the contents of tag-spec.yml with Cerberus.
# Must be converted to a dict from yaml to work.
# Note: can add python references with e.g.
# !!python/name:builtins.str
# !!python/name:__main__.funcname
SCHEMA_TEXT = """
header:
  type: string

slice_sets:
  type: dict
  keyschema:
    type: string
  valueschema:
     type: list
     schema:
        type: dict
        schema:
           add_to_name:
             type: string
           dockerfile_exclusive_name:
             type: string
           dockerfile_subdirectory:
             type: string
           partials:
             type: list
             schema:
               type: string
               ispartial: true
           test_runtime:
             type: string
             required: false
           tests:
             type: list
             default: []
             schema:
               type: string
           args:
             type: list
             default: []
             schema:
               type: string
               isfullarg: true

releases:
  type: dict
  keyschema:
    type: string
  valueschema:
    type: dict
    schema:
      is_dockerfiles:
        type: boolean
        required: false
        default: false
      upload_images:
        type: boolean
        required: false
        default: true
      tag_specs:
        type: list
        required: true
        schema:
          type: string
"""


class TfDockerTagValidator(cerberus.Validator):
  """Custom Cerberus validator for TF tag spec.

  Note: Each _validate_foo function's docstring must end with a segment
  describing its own validation schema, e.g. "The rule's arguments are...". If
  you add a new validator, you can copy/paste that section.
  """

  def __init__(self, *args, **kwargs):
    # See http://docs.python-cerberus.org/en/stable/customize.html
    if 'partials' in kwargs:
      self.partials = kwargs['partials']
    super(cerberus.Validator, self).__init__(*args, **kwargs)

  def _validate_ispartial(self, ispartial, field, value):
    """Validate that a partial references an existing partial spec.

    Args:
      ispartial: Value of the rule, a bool
      field: The field being validated
      value: The field's value
    The rule's arguments are validated against this schema:
    {'type': 'boolean'}
    """
    if ispartial and value not in self.partials:
      self._error(field,
                  '{} is not present in the partials directory.'.format(value))

  def _validate_isfullarg(self, isfullarg, field, value):
    """Validate that a string is either a FULL=arg or NOT.

    Args:
      isfullarg: Value of the rule, a bool
      field: The field being validated
      value: The field's value
    The rule's arguments are validated against this schema:
    {'type': 'boolean'}
    """
    if isfullarg and '=' not in value:
      self._error(field, '{} should be of the form ARG=VALUE.'.format(value))
    if not isfullarg and '=' in value:
      self._error(field, '{} should be of the form ARG (no =).'.format(value))


def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, flush=True, **kwargs)


def aggregate_all_slice_combinations(spec, slice_set_names):
  """Figure out all of the possible slice groupings for a tag spec."""
  slice_sets = copy.deepcopy(spec['slice_sets'])

  for name in slice_set_names:
    for slice_set in slice_sets[name]:
      slice_set['set_name'] = name

  slices_grouped_but_not_keyed = [slice_sets[name] for name in slice_set_names]
  all_slice_combos = list(itertools.product(*slices_grouped_but_not_keyed))
  return all_slice_combos


def build_name_from_slices(format_string, slices, args, is_dockerfile=False):
  """Build the tag name (cpu-devel...) from a list of slices."""
  name_formatter = copy.deepcopy(args)
  name_formatter.update({s['set_name']: s['add_to_name'] for s in slices})
  name_formatter.update({
      s['set_name']: s['dockerfile_exclusive_name']
      for s in slices
      if is_dockerfile and 'dockerfile_exclusive_name' in s
  })
  name = format_string.format(**name_formatter)
  return name


def update_args_dict(args_dict, updater):
  """Update a dict of arg values with more values from a list or dict."""
  if isinstance(updater, list):
    for arg in updater:
      key, sep, value = arg.partition('=')
      if sep == '=':
        args_dict[key] = value
  if isinstance(updater, dict):
    for key, value in updater.items():
      args_dict[key] = value
  return args_dict


def get_slice_sets_and_required_args(slice_sets, tag_spec):
  """Extract used-slice-sets and required CLI arguments from a spec string.

  For example, {FOO}{bar}{bat} finds FOO, bar, and bat. Assuming bar and bat
  are both named slice sets, FOO must be specified on the command line.

  Args:
     slice_sets: Dict of named slice sets
     tag_spec: The tag spec string, e.g. {_FOO}{blep}

  Returns:
     (used_slice_sets, required_args), a tuple of lists
  """
  required_args = []
  used_slice_sets = []

  extract_bracketed_words = re.compile(r'\{([^}]+)\}')
  possible_args_or_slice_set_names = extract_bracketed_words.findall(tag_spec)
  for name in possible_args_or_slice_set_names:
    if name in slice_sets:
      used_slice_sets.append(name)
    else:
      required_args.append(name)

  return (used_slice_sets, required_args)


def gather_tag_args(slices, cli_input_args, required_args):
  """Build a dictionary of all the CLI and slice-specified args for a tag."""
  args = dict()

  for s in slices:
    args = update_args_dict(args, s['args'])

  args = update_args_dict(args, cli_input_args)
  for arg in required_args:
    if arg not in args:
      eprint(('> Error: {} is not a valid slice_set, and also isn\'t an arg '
              'provided on the command line. If it is an arg, please specify '
              'it with --arg. If not, check the slice_sets list.'.format(arg)))
      exit(1)

  return args


def gather_slice_list_items(slices, key):
  """For a list of slices, get the flattened list of all of a certain key."""
  return list(itertools.chain(*[s[key] for s in slices if key in s]))


def find_first_slice_value(slices, key):
  """For a list of slices, get the first value for a certain key."""
  for s in slices:
    if key in s and s[key] is not None:
      return s[key]
  return None


def assemble_tags(spec, cli_args, enabled_releases, all_partials):
  """Gather all the tags based on our spec.

  Args:
    spec: Nested dict containing full Tag spec
    cli_args: List of ARG=foo arguments to pass along to Docker build
    enabled_releases: List of releases to parse. Empty list = all
    all_partials: Dict of every partial, for reference

  Returns:
    Dict of tags and how to build them
  """
  tag_data = collections.defaultdict(list)

  for name, release in spec['releases'].items():
    for tag_spec in release['tag_specs']:
      if enabled_releases and name not in enabled_releases:
        eprint('> Skipping release {}'.format(name))
        continue

      used_slice_sets, required_cli_args = get_slice_sets_and_required_args(
          spec['slice_sets'], tag_spec)

      slice_combos = aggregate_all_slice_combinations(spec, used_slice_sets)
      for slices in slice_combos:

        tag_args = gather_tag_args(slices, cli_args, required_cli_args)
        tag_name = build_name_from_slices(tag_spec, slices, tag_args,
                                          release['is_dockerfiles'])
        used_partials = gather_slice_list_items(slices, 'partials')
        used_tests = gather_slice_list_items(slices, 'tests')
        test_runtime = find_first_slice_value(slices, 'test_runtime')
        dockerfile_subdirectory = find_first_slice_value(
            slices, 'dockerfile_subdirectory')
        dockerfile_contents = merge_partials(spec['header'], used_partials,
                                             all_partials)

        tag_data[tag_name].append({
            'release': name,
            'tag_spec': tag_spec,
            'is_dockerfiles': release['is_dockerfiles'],
            'upload_images': release['upload_images'],
            'cli_args': tag_args,
            'dockerfile_subdirectory': dockerfile_subdirectory or '',
            'partials': used_partials,
            'tests': used_tests,
            'test_runtime': test_runtime,
            'dockerfile_contents': dockerfile_contents,
        })

  return tag_data


def merge_partials(header, used_partials, all_partials):
  """Merge all partial contents with their header."""
  used_partials = list(used_partials)
  return '\n'.join([header] + [all_partials[u] for u in used_partials])


def upload_in_background(hub_repository, dock, image, tag):
  """Upload a docker image (to be used by multiprocessing)."""
  image.tag(hub_repository, tag=tag)
  print(dock.images.push(hub_repository, tag=tag))


def mkdir_p(path):
  """Create a directory and its parents, even if it already exists."""
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise


def gather_existing_partials(partial_path):
  """Find and read all available partials.

  Args:
    partial_path (string): read partials from this directory.

  Returns:
    Dict[string, string] of partial short names (like "ubuntu/python" or
      "bazel") to the full contents of that partial.
  """
  partials = dict()
  for path, _, files in os.walk(partial_path):
    for name in files:
      fullpath = os.path.join(path, name)
      if '.partial.Dockerfile' not in fullpath:
        eprint(('> Probably not a problem: skipping {}, which is not a '
                'partial.').format(fullpath))
        continue
      # partial_dir/foo/bar.partial.Dockerfile -> foo/bar
      simple_name = fullpath[len(partial_path) + 1:-len('.partial.dockerfile')]
      with open(fullpath, 'r') as f:
        partial_contents = f.read()
      partials[simple_name] = partial_contents
  return partials


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Read the full spec file, used for everything
  with open(FLAGS.spec_file, 'r') as spec_file:
    tag_spec = yaml.load(spec_file)

  # Get existing partial contents
  partials = gather_existing_partials(FLAGS.partial_dir)

  # Abort if spec.yaml is invalid
  schema = yaml.load(SCHEMA_TEXT)
  v = TfDockerTagValidator(schema, partials=partials)
  if not v.validate(tag_spec):
    eprint('> Error: {} is an invalid spec! The errors are:'.format(
        FLAGS.spec_file))
    eprint(yaml.dump(v.errors, indent=2))
    exit(1)
  tag_spec = v.normalized(tag_spec)

  # Assemble tags and images used to build them
  all_tags = assemble_tags(tag_spec, FLAGS.arg, FLAGS.release, partials)

  # Empty Dockerfile directory if building new Dockerfiles
  if FLAGS.construct_dockerfiles:
    eprint('> Emptying Dockerfile dir "{}"'.format(FLAGS.dockerfile_dir))
    shutil.rmtree(FLAGS.dockerfile_dir, ignore_errors=True)
    mkdir_p(FLAGS.dockerfile_dir)

  # Set up Docker helper
  dock = docker.from_env()

  # Login to Docker if uploading images
  if FLAGS.upload_to_hub:
    if not FLAGS.hub_username:
      eprint('> Error: please set --hub_username when uploading to Dockerhub.')
      exit(1)
    if not FLAGS.hub_repository:
      eprint(
          '> Error: please set --hub_repository when uploading to Dockerhub.')
      exit(1)
    if not FLAGS.hub_password:
      eprint('> Error: please set --hub_password when uploading to Dockerhub.')
      exit(1)
    dock.login(
        username=FLAGS.hub_username,
        password=FLAGS.hub_password,
    )

  # Each tag has a name ('tag') and a definition consisting of the contents
  # of its Dockerfile, its build arg list, etc.
  failed_tags = []
  succeeded_tags = []
  for tag, tag_defs in all_tags.items():
    for tag_def in tag_defs:
      eprint('> Working on {}'.format(tag))

      if FLAGS.exclude_tags_matching and re.match(FLAGS.exclude_tags_matching,
                                                  tag):
        eprint('>> Excluded due to match against "{}".'.format(
            FLAGS.exclude_tags_matching))
        continue

      if FLAGS.only_tags_matching and not re.match(FLAGS.only_tags_matching,
                                                   tag):
        eprint('>> Excluded due to failure to match against "{}".'.format(
            FLAGS.only_tags_matching))
        continue

      # Write releases marked "is_dockerfiles" into the Dockerfile directory
      if FLAGS.construct_dockerfiles and tag_def['is_dockerfiles']:
        path = os.path.join(FLAGS.dockerfile_dir,
                            tag_def['dockerfile_subdirectory'],
                            tag + '.Dockerfile')
        eprint('>> Writing {}...'.format(path))
        if not FLAGS.dry_run:
          mkdir_p(os.path.dirname(path))
          with open(path, 'w') as f:
            f.write(tag_def['dockerfile_contents'])

      # Don't build any images for dockerfile-only releases
      if not FLAGS.build_images:
        continue

      # Generate a temporary Dockerfile to use to build, since docker-py
      # needs a filepath relative to the build context (i.e. the current
      # directory)
      dockerfile = os.path.join(FLAGS.dockerfile_dir, tag + '.temp.Dockerfile')
      if not FLAGS.dry_run:
        with open(dockerfile, 'w') as f:
          f.write(tag_def['dockerfile_contents'])
      eprint('>> (Temporary) writing {}...'.format(dockerfile))

      repo_tag = '{}:{}'.format(FLAGS.repository, tag)
      eprint('>> Building {} using build args:'.format(repo_tag))
      for arg, value in tag_def['cli_args'].items():
        eprint('>>> {}={}'.format(arg, value))

      # Note that we are NOT using cache_from, which appears to limit
      # available cache layers to those from explicitly specified layers. Many
      # of our layers are similar between local builds, so we want to use the
      # implied local build cache.
      tag_failed = False
      image, logs = None, []
      if not FLAGS.dry_run:
        try:
          image, logs = dock.images.build(
              timeout=FLAGS.hub_timeout,
              path='.',
              nocache=FLAGS.nocache,
              dockerfile=dockerfile,
              buildargs=tag_def['cli_args'],
              tag=repo_tag)

          # Print logs after finishing
          log_lines = [l.get('stream', '') for l in logs]
          eprint(''.join(log_lines))

          # Run tests if requested, and dump output
          # Could be improved by backgrounding, but would need better
          # multiprocessing support to track failures properly.
          if FLAGS.run_tests_path:
            if not tag_def['tests']:
              eprint('>>> No tests to run.')
            for test in tag_def['tests']:
              eprint('>> Testing {}...'.format(test))
              container, = dock.containers.run(
                  image,
                  '/tests/' + test,
                  working_dir='/',
                  log_config={'type': 'journald'},
                  detach=True,
                  stderr=True,
                  stdout=True,
                  volumes={
                      FLAGS.run_tests_path: {
                          'bind': '/tests',
                          'mode': 'ro'
                      }
                  },
                  runtime=tag_def['test_runtime']),
              ret = container.wait()
              code = ret['StatusCode']
              out = container.logs(stdout=True, stderr=False)
              err = container.logs(stdout=False, stderr=True)
              container.remove()
              if out:
                eprint('>>> Output stdout:')
                eprint(out.decode('utf-8'))
              else:
                eprint('>>> No test standard out.')
              if err:
                eprint('>>> Output stderr:')
                eprint(out.decode('utf-8'))
              else:
                eprint('>>> No test standard err.')
              if code != 0:
                eprint('>> {} failed tests with status: "{}"'.format(
                    repo_tag, code))
                failed_tags.append(tag)
                tag_failed = True
                if FLAGS.stop_on_failure:
                  eprint('>> ABORTING due to --stop_on_failure!')
                  exit(1)
              else:
                eprint('>> Tests look good!')

        except docker.errors.BuildError as e:
          eprint('>> {} failed to build with message: "{}"'.format(
              repo_tag, e.msg))
          eprint('>> Build logs follow:')
          log_lines = [l.get('stream', '') for l in e.build_log]
          eprint(''.join(log_lines))
          failed_tags.append(tag)
          tag_failed = True
          if FLAGS.stop_on_failure:
            eprint('>> ABORTING due to --stop_on_failure!')
            exit(1)

        # Clean temporary dockerfiles if they were created earlier
        if not FLAGS.keep_temp_dockerfiles:
          os.remove(dockerfile)

      # Upload new images to DockerHub as long as they built + passed tests
      if FLAGS.upload_to_hub:
        if not tag_def['upload_images']:
          continue
        if tag_failed:
          continue

        eprint('>> Uploading to {}:{}'.format(FLAGS.hub_repository, tag))
        if not FLAGS.dry_run:
          p = multiprocessing.Process(
              target=upload_in_background,
              args=(FLAGS.hub_repository, dock, image, tag))
          p.start()

      if not tag_failed:
        succeeded_tags.append(tag)

  if failed_tags:
    eprint(
        '> Some tags failed to build or failed testing, check scrollback for '
        'errors: {}'.format(','.join(failed_tags)))
    exit(1)

  eprint('> Writing built{} tags to standard out.'.format(
      ' and tested' if FLAGS.run_tests_path else ''))
  for tag in succeeded_tags:
    print('{}:{}'.format(FLAGS.repository, tag))


if __name__ == '__main__':
  app.run(main)
