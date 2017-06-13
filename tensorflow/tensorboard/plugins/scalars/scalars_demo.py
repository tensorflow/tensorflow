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
"""Sample data exhibiting scalar summaries, via a temperature simulation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Directory into which to write tensorboard data.
LOGDIR = '/tmp/scalars_demo'

# Duration of the simulation.
STEPS = 1000


def run(logdir, run_name,
        initial_temperature, ambient_temperature, heat_coefficient):
  """Run a temperature simulation.

  This will simulate an object at temperature `initial_temperature`
  sitting at rest in a large room at temperature `ambient_temperature`.
  The object has some intrinsic `heat_coefficient`, which indicates
  how much thermal conductivity it has: for instance, metals have high
  thermal conductivity, while the thermal conductivity of water is low.

  Over time, the object's temperature will adjust to match the
  temperature of its environment. We'll track the object's temperature,
  how far it is from the room's temperature, and how much it changes at
  each time step.

  Arguments:
    logdir: the top-level directory into which to write summary data
    run_name: the name of this run; will be created as a subdirectory
      under logdir
    initial_temperature: float; the object's initial temperature
    ambient_temperature: float; the temperature of the enclosing room
    heat_coefficient: float; a measure of the object's thermal
      conductivity
  """
  tf.reset_default_graph()
  tf.set_random_seed(0)

  with tf.name_scope('temperature'):
    # Create a mutable variable to hold the object's temperature, and
    # create a scalar summary to track its value over time. The name of
    # the summary will appear as "temperature/current" due to the
    # name-scope above.
    temperature = tf.Variable(tf.constant(initial_temperature),
                              name='temperature')
    tf.summary.scalar('current', temperature)

    # Compute how much the object's temperature differs from that of its
    # environment, and track this, too: likewise, as
    # "temperature/difference_to_ambient".
    ambient_difference = temperature - ambient_temperature
    tf.summary.scalar('difference_to_ambient', ambient_difference)

  # Newton suggested that the rate of change of the temperature of an
  # object is directly proportional to this `ambient_difference` above,
  # where the proportionality constant is what we called the heat
  # coefficient. But in real life, not everything is quite so clean, so
  # we'll add in some noise. (The value of 50 is arbitrary, chosen to
  # make the data look somewhat interesting. :-) )
  noise = 50 * tf.random_normal([])
  delta = -heat_coefficient * (ambient_difference + noise)
  tf.summary.scalar('delta', delta)

  # Now, augment the current temperature by this delta that we computed.
  update_step = temperature.assign_add(delta)

  # Collect all the scalars that we want to keep track of.
  summ = tf.summary.merge_all()

  sess = tf.Session()
  writer = tf.summary.FileWriter(os.path.join(logdir, run_name))
  writer.add_graph(sess.graph)
  sess.run(tf.global_variables_initializer())
  for step in xrange(STEPS):
    # By asking TensorFlow to compute the update step, we force it to
    # change the value of the temperature variable. We don't actually
    # care about this value, so we discard it; instead, we grab the
    # summary data computed along the way.
    (s, _) = sess.run([summ, update_step])
    writer.add_summary(s, global_step=step)
  writer.close()


def run_all(logdir, verbose=False):
  """Run simulations on a reasonable set of parameters.

  Arguments:
    logdir: the directory into which to store all the runs' data
    verbose: if true, print out each run's name as it begins
  """
  for initial_temperature in [270.0, 310.0, 350.0]:
    for final_temperature in [270.0, 310.0, 350.0]:
      for heat_coefficient in [0.001, 0.005]:
        run_name = 'temperature:t0=%g,tA=%g,kH=%g' % (
            initial_temperature, final_temperature, heat_coefficient)
        if verbose:
          print('--- Running: %s' % run_name)
        run(logdir, run_name,
            initial_temperature, final_temperature, heat_coefficient)


def main(unused_argv):
  print('Saving output to %s.' % LOGDIR)
  run_all(LOGDIR, verbose=True)
  print('Done. Output saved to %s.' % LOGDIR)


if __name__ == '__main__':
  tf.app.run()
