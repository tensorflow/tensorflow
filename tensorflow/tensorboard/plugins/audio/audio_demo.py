# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Sample data exhibiting audio summaries, via a waveform generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('logdir', '/tmp/audio_demo',
                       'Directory into which to write TensorBoard data.')

tf.flags.DEFINE_integer('steps', 500,
                        'Number of frequencies of each waveform to generate.')

# Parameters for the audio output.
tf.flags.DEFINE_integer('sample_rate', 44100, 'Sample rate, in Hz.')
tf.flags.DEFINE_float('duration', 2.0, 'Duration of each waveform, in s.')


def _samples():
  """Compute how many samples should be included in each waveform."""
  return int(FLAGS.sample_rate * FLAGS.duration)


def run(logdir, run_name, wave_name, wave_constructor):
  """Generate wave data of the given form.

  The provided function `wave_constructor` should accept a scalar tensor
  of type float32, representing the frequency (in Hz) at which to
  construct a wave, and return a tensor of shape [1, _samples(), `n`]
  representing audio data (for some number of channels `n`).

  Waves will be generated at frequencies ranging from A4 to A5.

  Arguments:
    logdir: the top-level directory into which to write summary data
    run_name: the name of this run; will be created as a subdirectory
      under logdir
    wave_name: the name of the wave being generated
    wave_constructor: see above
  """
  tf.reset_default_graph()
  tf.set_random_seed(0)

  # On each step `i`, we'll set this placeholder to `i`. This allows us
  # to know "what time it is" at each step.
  step_placeholder = tf.placeholder(tf.float32, shape=[])

  # We want to linearly interpolate a frequency between A4 (440 Hz) and
  # A5 (880 Hz).
  f_min = 440.0
  f_max = 880.0
  t = step_placeholder / (FLAGS.steps - 1)
  frequency = f_min * (1.0 - t) + f_max * t

  # Let's log this frequency, just so that we can make sure that it's as
  # expected.
  tf.summary.scalar('frequency', frequency)

  # Now, we pass this to the wave constructor to get our waveform. Doing
  # so within a name scope means that any summaries that the wave
  # constructor produces will be namespaced.
  with tf.name_scope(wave_name):
    waveform = wave_constructor(frequency)

  # Here's the crucial piece: we interpret this result as audio.
  tf.summary.audio('waveform', waveform, FLAGS.sample_rate)

  # Now, we can collect up all the summaries and begin the run.
  summ = tf.summary.merge_all()

  sess = tf.Session()
  writer = tf.summary.FileWriter(os.path.join(logdir, run_name))
  writer.add_graph(sess.graph)
  sess.run(tf.global_variables_initializer())
  for step in xrange(FLAGS.steps):
    s = sess.run(summ, feed_dict={step_placeholder: float(step)})
    writer.add_summary(s, global_step=step)
  writer.close()


# Now, let's take a look at the kinds of waves that we can generate.


def sine_wave(frequency):
  """Emit a sine wave at the given frequency."""
  xs = tf.reshape(tf.range(_samples(), dtype=tf.float32), [1, _samples(), 1])
  ts = xs / FLAGS.sample_rate
  return tf.sin(2 * math.pi * frequency * ts)


def square_wave(frequency):
  """Emit a square wave at the given frequency."""
  # The square is just the sign of the sine!
  return tf.sign(sine_wave(frequency))


def triangle_wave(frequency):
  """Emit a triangle wave at the given frequency."""
  xs = tf.reshape(tf.range(_samples(), dtype=tf.float32), [1, _samples(), 1])
  ts = xs / FLAGS.sample_rate
  #
  # A triangle wave looks like this:
  #
  #      /\      /\
  #     /  \    /  \
  #         \  /    \  /
  #          \/      \/
  #
  # If we look at just half a period (the first four slashes in the
  # diagram above), we can see that it looks like a transformed absolute
  # value function.
  #
  # Let's start by computing the times relative to the start of each
  # half-wave pulse (each individual "mountain" or "valley", of which
  # there are four in the above diagram).
  half_pulse_index = ts * (frequency * 2)
  half_pulse_angle = half_pulse_index % 1.0  # in [0, 1]
  #
  # Now, we can see that each positive half-pulse ("mountain") has
  # amplitude given by A(z) = 0.5 - abs(z - 0.5), and then normalized:
  absolute_amplitude = (0.5 - tf.abs(half_pulse_angle - 0.5)) / 0.5
  #
  # But every other half-pulse is negative, so we should invert these.
  half_pulse_parity = tf.sign(1 - (half_pulse_index % 2.0))
  amplitude = half_pulse_parity * absolute_amplitude
  #
  # This is precisely the desired result, so we're done!
  return amplitude


# If we want to get fancy, we can use our above waves as primitives to
# build more interesting waves.


def bisine_wave(frequency):
  """Emit two sine waves, in stereo at different octaves."""
  #
  # We can first our existing sine generator to generate two different
  # waves.
  f_hi = frequency
  f_lo = frequency / 2.0
  with tf.name_scope('hi'):
    sine_hi = sine_wave(f_hi)
  with tf.name_scope('lo'):
    sine_lo = sine_wave(f_lo)
  #
  # Now, we have two tensors of shape [1, _samples(), 1]. By concatenating
  # them along axis 2, we get a tensor of shape [1, _samples(), 2]---a
  # stereo waveform.
  return tf.concat([sine_lo, sine_hi], axis=2)


def bisine_wahwah_wave(frequency):
  """Emit two sine waves with balance oscillating left and right."""
  #
  # This is clearly intended to build on the bisine wave defined above,
  # so we can start by generating that.
  waves_a = bisine_wave(frequency)
  #
  # Then, by reversing axis 2, we swap the stereo channels. By mixing
  # this with `waves_a`, we'll be able to create the desired effect.
  waves_b = tf.reverse(waves_a, axis=[2])
  #
  # Let's have the balance oscillate from left to right four times.
  iterations = 4
  #
  # Now, we compute the balance for each sample: `ts` has values
  # in [0, 1] that indicate how much we should use `waves_a`.
  xs = tf.reshape(tf.range(_samples(), dtype=tf.float32), [1, _samples(), 1])
  thetas = xs / _samples() * iterations
  ts = (tf.sin(math.pi * 2 * thetas) + 1) / 2
  #
  # Finally, we can mix the two together, and we're done.
  return ts * waves_a + (1.0 - ts) * waves_b


def run_all(logdir, verbose=False):
  """Generate waves of the shapes defined above.

  Arguments:
    logdir: the directory into which to store all the runs' data
    verbose: if true, print out each run's name as it begins
  """
  waves = [sine_wave, square_wave, triangle_wave,
           bisine_wave, bisine_wahwah_wave]
  for (i, wave_constructor) in enumerate(waves):
    wave_name = wave_constructor.__name__
    run_name = 'wave:%02d,%s' % (i + 1, wave_name)
    if verbose:
      print('--- Running: %s' % run_name)
    run(logdir, run_name, wave_name, wave_constructor)


def main(unused_argv):
  print('Saving output to %s.' % FLAGS.logdir)
  run_all(FLAGS.logdir, verbose=True)
  print('Done. Output saved to %s.' % FLAGS.logdir)


if __name__ == '__main__':
  tf.app.run()
