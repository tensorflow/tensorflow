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
"""Simple bash progress bar with estimated completion time."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
from datetime import timedelta

time_points = []
progress_points = []

max_points = 60
min_points = 4


def update_progress_bar(progress,
                        pre_msg='',
                        post_msg='',
                        size=40,
                        show_times=False,
                        c_return='\r'):
  """Function to display a pretty progress bar on the command line,
  along with a message and optional elapsed and estimated time
  remaining."""

  global time_points, progress_points
  global max_points, min_points

  # if the progress is less than the last progress point reset the points
  if progress_points == [] or progress_points[-1] > progress:
    time_points = [datetime.now()]
    progress_points = [progress]
  elif (datetime.now() - time_points[-1]).total_seconds() > 1.0:
    time_points += [datetime.now()]
    progress_points += [progress]

    if len(time_points) > max_points:
      time_points = time_points[-max_points:]
      progress_points = progress_points[-max_points:]

  progress_string = '%s [\033[92m' % pre_msg
  for i in range(1, size+1):
    if progress * size > i:
      progress_string = '%s#' % progress_string
    elif progress * size > i-0.25:
      progress_string = '%s=' % progress_string
    elif progress * size > i-0.5:
      progress_string = '%s~' % progress_string
    elif progress * size > i-0.75:
      progress_string = '%s-' % progress_string
    else:
      progress_string = '%s ' % progress_string
  progress_string = '%s\033[0m] %6.2f%% %s' %\
                    (progress_string, progress*100, post_msg)

  if show_times:
    if len(time_points) < min_points:
      progress_string += " ( ... )"
    else:
      recent_duration_secs = (time_points[-1] - time_points[0]).total_seconds()
      recent_progress = progress_points[-1] - progress_points[0]
      time_g = recent_progress / recent_duration_secs
      prog_remaining = 1 - progress
      seconds_remaining = prog_remaining / time_g
      time_remaining = seconds_remaining

      seconds = time_remaining - (int(time_remaining/60) * 60)
      time_remaining = int(time_remaining/60)

      minutes = time_remaining - (int(time_remaining/60) * 60)
      time_remaining = int(time_remaining/60)

      hours = time_remaining - (int(time_remaining/24) * 24)

      days = int(time_remaining/24)

      remaining_string = "%2ds" % seconds
      if minutes > 0 or hours > 0 or days > 0:
        remaining_string = "%2dm %s" % (minutes, remaining_string)
      if hours > 0 or days > 0:
        remaining_string = "%2dh %s" % (hours, remaining_string)
      if days > 0:
        remaining_string = "%2dd %s" % (days, remaining_string)

      progress_string += " ( %s )" % remaining_string

      if seconds_remaining > 3600:  # more than an hour then add ETC time
        etc = time_points[-1] + timedelta(seconds=seconds_remaining)
        progress_string += " ETC %s" % etc.strftime("%b %d %Y %H:%M:%S")

      progress_string += "      "

  print(progress_string, end=c_return, flush=True)


def finish_progress_bar(progress):

  update_progress_bar(progress, c_return='\n')
