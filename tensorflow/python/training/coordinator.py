# Copyright 2015 Google Inc. All Rights Reserved.
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
"""Coordinator to help multiple threads stop when requested."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import sys
import threading
import time

import six

from tensorflow.python.framework import errors
from tensorflow.python.platform import logging
from tensorflow.python.util import compat


class Coordinator(object):
  """A coordinator for threads.

  This class implements a simple mechanism to coordinate the termination of a
  set of threads.

  #### Usage:

  ```python
  # Create a coordinator.
  coord = Coordinator()
  # Start a number of threads, passing the coordinator to each of them.
  ...start thread 1...(coord, ...)
  ...start thread N...(coord, ...)
  # Wait for all the threads to terminate.
  coord.join(threads)
  ```

  Any of the threads can call `coord.request_stop()` to ask for all the threads
  to stop.  To cooperate with the requests, each thread must check for
  `coord.should_stop()` on a regular basis.  `coord.should_stop()` returns
  `True` as soon as `coord.request_stop()` has been called.

  A typical thread running with a coordinator will do something like:

  ```python
  while not coord.should_stop():
    ...do some work...
  ```

  #### Exception handling:

  A thread can report an exception to the coordinator as part of the
  `should_stop()` call.  The exception will be re-raised from the
  `coord.join()` call.

  Thread code:

  ```python
  try:
    while not coord.should_stop():
      ...do some work...
  except Exception as e:
    coord.request_stop(e)
  ```

  Main code:

  ```python
  try:
    ...
    coord = Coordinator()
    # Start a number of threads, passing the coordinator to each of them.
    ...start thread 1...(coord, ...)
    ...start thread N...(coord, ...)
    # Wait for all the threads to terminate.
    coord.join(threads)
  except Exception as e:
    ...exception that was passed to coord.request_stop()
  ```

  To simplify the thread implementation, the Coordinator provides a
  context handler `stop_on_exception()` that automatically requests a stop if
  an exception is raised.  Using the context handler the thread code above
  can be written as:

  ```python
  with coord.stop_on_exception():
    while not coord.should_stop():
      ...do some work...
  ```

  #### Grace period for stopping:

  After a thread has called `coord.request_stop()` the other threads have a
  fixed time to stop, this is called the 'stop grace period' and defaults to 2
  minutes.  If any of the threads is still alive after the grace period expires
  `coord.join()` raises a RuntimeException reporting the laggards.

  ```python
  try:
    ...
    coord = Coordinator()
    # Start a number of threads, passing the coordinator to each of them.
    ...start thread 1...(coord, ...)
    ...start thread N...(coord, ...)
    # Wait for all the threads to terminate, give them 10s grace period
    coord.join(threads, stop_grace_period_secs=10)
  except RuntimeException:
    ...one of the threads took more than 10s to stop after request_stop()
    ...was called.
  except Exception:
    ...exception that was passed to coord.request_stop()
  ```
  """

  def __init__(self):
    """Create a new Coordinator."""
    # Protects all attributes.
    self._lock = threading.Lock()
    # Event set when threads must stop.
    self._stop_event = threading.Event()
    # Python exc_info to report.
    # If not None, it should hold the returned value of sys.exc_info(), which is
    # a tuple containing exception (type, value, traceback).
    self._exc_info_to_raise = None

  def _filter_exception(self, ex):
    """Check if the exception indicated in 'ex' should be ignored.

    This method examines `ex` to check if it is an exception that should be
    reported to the users.  If yes, it returns `ex` as is, otherwise it returns
    None.

    The code returns None for exceptions that are used for control flow such as
    the OutOfRangeError raised by the dequeue operations to indicate that a
    queue was closed after its contents were dequeued.

    Args:
      ex: None, an `Exception`, or a Python `exc_info` tuple as returned by
        `sys.exc_info()`.

    Returns:
      ex or None.
    """
    if isinstance(ex, tuple):
      ex2 = ex[1]
    else:
      ex2 = ex
    # OutOfRangeError is used to indicate "end of input".  We do not want to
    # report an exception for it.  TODO(touts): Likely also need to ignore
    # some of the Aborted and Cancelled exceptions raised by queue ops after
    # queues are closed, but this can only be done after these exceptions have
    # been clearly identified.
    if isinstance(ex2, (errors.OutOfRangeError)):
      # Ignore the exception.
      ex = None
    return ex

  def request_stop(self, ex=None):
    """Request that the threads stop.

    After this is called, calls to `should_stop()` will return `True`.

    Note: If an exception is being passed in, in must be in the context of
    handling the exception (i.e. `try: ... except Exception as ex: ...`) and not
    a newly created one.

    Args:
      ex: Optional `Exception`, or Python `exc_info` tuple as returned by
        `sys.exc_info()`.  If this is the first call to `request_stop()` the
        corresponding exception is recorded and re-raised from `join()`.
    """
    ex = self._filter_exception(ex)
    with self._lock:
      if not self._stop_event.is_set():
        if ex and self._exc_info_to_raise is None:
          if isinstance(ex, tuple):
            logging.info("Error reported to Coordinator: %s",
                         compat.as_str_any(ex[1]))
            self._exc_info_to_raise = ex
          else:
            logging.info("Error reported to Coordinator: %s",
                         compat.as_str_any(ex))
            self._exc_info_to_raise = sys.exc_info()
          # self._exc_info_to_raise should contain a tuple containing exception
          # (type, value, traceback)
          if (len(self._exc_info_to_raise) != 3 or
              not self._exc_info_to_raise[0] or
              not self._exc_info_to_raise[1]):
            # Raise, catch and record the exception here so that error happens
            # where expected.
            try:
              raise ValueError(
                  "ex must be a tuple or sys.exc_info must return the current "
                  "exception: %s"
                  % self._exc_info_to_raise)
            except ValueError:
              # Record this error so it kills the coordinator properly.
              self._exc_info_to_raise = sys.exc_info()

        self._stop_event.set()

  def clear_stop(self):
    """Clears the stop flag.

    After this is called, calls to `should_stop()` will return `False`.
    """
    with self._lock:
      if self._stop_event.is_set():
        self._stop_event.clear()

  def should_stop(self):
    """Check if stop was requested.

    Returns:
      True if a stop was requested.
    """
    return self._stop_event.is_set()

  @contextlib.contextmanager
  def stop_on_exception(self):
    """Context manager to request stop when an Exception is raised.

    Code that uses a coordinator must catch exceptions and pass
    them to the `request_stop()` method to stop the other threads
    managed by the coordinator.

    This context handler simplifies the exception handling.
    Use it as follows:

    ```python
    with coord.stop_on_exception():
      # Any exception raised in the body of the with
      # clause is reported to the coordinator before terminating
      # the execution of the body.
      ...body...
    ```

    This is completely equivalent to the slightly longer code:

    ```python
    try:
      ...body...
    exception Exception as ex:
      coord.request_stop(ex)
    ```

    Yields:
      nothing.
    """
    # pylint: disable=broad-except
    try:
      yield
    except Exception as ex:
      self.request_stop(ex)
    # pylint: enable=broad-except

  def wait_for_stop(self, timeout=None):
    """Wait till the Coordinator is told to stop.

    Args:
      timeout: Float.  Sleep for up to that many seconds waiting for
        should_stop() to become True.

    Returns:
      True if the Coordinator is told stop, False if the timeout expired.
    """
    return self._stop_event.wait(timeout)

  def join(self, threads, stop_grace_period_secs=120):
    """Wait for threads to terminate.

    Blocks until all `threads` have terminated or `request_stop()` is called.

    After the threads stop, if an `exc_info` was passed to `request_stop`, that
    exception is re-raised.

    Grace period handling: When `request_stop()` is called, threads are given
    'stop_grace_period_secs' seconds to terminate.  If any of them is still
    alive after that period expires, a `RuntimeError` is raised.  Note that if
    an `exc_info` was passed to `request_stop()` then it is raised instead of
    that `RuntimeError`.

    Args:
      threads: List of `threading.Threads`. The started threads to join.
      stop_grace_period_secs: Number of seconds given to threads to stop after
        `request_stop()` has been called.

    Raises:
      RuntimeError: If any thread is still alive after `request_stop()`
        is called and the grace period expires.
    """
    # Wait for all threads to stop or for request_stop() to be called.
    while any(t.is_alive() for t in threads) and not self.wait_for_stop(1.0):
      pass

    # If any thread is still alive, wait for the grace period to expire.
    while any(t.is_alive() for t in threads) and stop_grace_period_secs >= 0.0:
      stop_grace_period_secs -= 1.0
      time.sleep(1.0)

    # List the threads still alive after the grace period.
    stragglers = [t.name for t in threads if t.is_alive()]

    # Terminate with an exception if appropriate.
    with self._lock:
      if self._exc_info_to_raise:
        six.reraise(*self._exc_info_to_raise)
      elif stragglers:
        raise RuntimeError("Coordinator stopped with threads still running: %s",
                           " ".join(stragglers))


# Threads for the standard services.
class LooperThread(threading.Thread):
  """A thread that runs code repeatedly, optionally on a timer.

  This thread class is intended to be used with a `Coordinator`.  It repeatedly
  runs code specified either as `target` and `args` or by the `run_loop()`
  method.

  Before each run the thread checks if the coordinator has requested stop.  In
  that case the looper thread terminates immediately.

  If the code being run raises an exception, that exception is reported to the
  coordinator and the thread terminates.  The coordinator will then request all
  the other threads it coordinates to stop.

  You typically pass looper threads to the supervisor `Join()` method.
  """

  def __init__(self, coord, timer_interval_secs, target=None, args=None):
    """Create a LooperThread.

    Args:
      coord: A Coordinator.
      timer_interval_secs: Time boundaries at which to call Run(), or None
        if it should be called back to back.
      target: Optional callable object that will be executed in the thread.
      args: Optional arguments to pass to `target` when calling it.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
    if not isinstance(coord, Coordinator):
      raise ValueError("'coord' argument must be a Coordinator: %s" % coord)
    super(LooperThread, self).__init__()
    self.daemon = True
    self._coord = coord
    self._timer_interval_secs = timer_interval_secs
    self._target = target
    if self._target:
      if args is None:
        self._args = ()
      else:
        self._args = args
    elif args:
      raise ValueError("'args' argument require that you also pass 'target'")

  @staticmethod
  def loop(coord, timer_interval_secs, target, args=None):
    """Start a LooperThread that calls a function periodically.

    If `timer_interval_secs` is None the thread calls `target(args)`
    repeatedly.  Otherwise `target(args)` is called every `timer_interval_secs`
    seconds.  The thread terminates when a stop of the coordinator is
    requested.

    Args:
      coord: A Coordinator.
      timer_interval_secs: Number. Time boundaries at which to call `target`.
      target: A callable object.
      args: Optional arguments to pass to `target` when calling it.

    Returns:
      The started thread.
    """
    looper = LooperThread(coord, timer_interval_secs, target=target, args=args)
    looper.start()
    return looper

  def run(self):
    with self._coord.stop_on_exception():
      self.start_loop()
      if self._timer_interval_secs is None:
        # Call back-to-back.
        while not self._coord.should_stop():
          self.run_loop()
      else:
        # Next time at which to call run_loop(), starts as 'now'.
        next_timer_time = time.time()
        while not self._coord.wait_for_stop(next_timer_time - time.time()):
          next_timer_time += self._timer_interval_secs
          self.run_loop()
      self.stop_loop()

  def start_loop(self):
    """Called when the thread starts."""
    pass

  def stop_loop(self):
    """Called when the thread stops."""
    pass

  def run_loop(self):
    """Called at 'timer_interval_secs' boundaries."""
    if self._target:
      self._target(*self._args)
