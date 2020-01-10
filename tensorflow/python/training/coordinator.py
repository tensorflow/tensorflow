"""Coordinator to help multiple threads stop when requested."""
import sys
import threading
import time

from tensorflow.python.platform import logging


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

  A typical thread running with a Coordinator will do something like:

  ```python
  while not coord.should_stop():
     ...do some work...
  ```

  #### Exception handling:

  A thread can report an exception to the Coordinator as part of the
  `should_stop()` call.  The exception will be re-raised from the
  `coord.join()` call.

  Thread code:

  ```python
  try:
    while not coord.should_stop():
      ...do some work...
  except Exception, e:
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
  except Exception, e:
    ...exception that was passed to coord.request_stop()
  ```

  #### Grace period for stopping:

  After a thread has called `coord.request_stop()` the other threads have a
  fixed time to stop, this is called the 'stop grace period' and defaults to 2
  minutes.  If any of the threads is still alive after the grace period expires
  `coord.join()` raises a RuntimeException reporting the laggards.

  ```
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
    self._exc_info_to_raise = None

  def request_stop(self, ex=None):
    """Request that the threads stop.

    After this is called, calls to should_stop() will return True.

    Args:
      ex: Optional Exception, or Python 'exc_info' tuple as returned by
        sys.exc_info().  If this is the first call to request_stop() the
        corresponding exception is recorded and re-raised from join().
    """
    with self._lock:
      if not self._stop_event.is_set():
        if ex and self._exc_info_to_raise is None:
          if isinstance(ex, tuple):
            logging.info("Error reported to Coordinator: %s", str(ex[1]))
            self._exc_info_to_raise = ex
          else:
            logging.info("Error reported to Coordinator: %s", str(ex))
            self._exc_info_to_raise = sys.exc_info()
        self._stop_event.set()

  def should_stop(self):
    """Check if stop was requested.

    Returns:
      True if a stop was requested.
    """
    return self._stop_event.is_set()

  def wait_for_stop(self, timeout=None):
    """Wait till the Coordinator is told to stop.

    Args:
      timeout: float.  Sleep for up to that many seconds waiting for
        should_stop() to become True.

    Returns:
      True if the Coordinator is told stop, False if the timeout expired.
    """
    return self._stop_event.wait(timeout)

  def join(self, threads, stop_grace_period_secs=120):
    """Wait for threads to terminate.

    Blocks until all 'threads' have terminated or request_stop() is called.

    After the threads stop, if an 'exc_info' was passed to request_stop, that
    exception is re-reaised.

    Grace period handling: When request_stop() is called, threads are given
    'stop_grace_period_secs' seconds to terminate.  If any of them is still
    alive after that period expires, a RuntimeError is raised.  Note that if
    an 'exc_info' was passed to request_stop() then it is raised instead of
    that RuntimeError.

    Args:
      threads: List threading.Threads. The started threads to join.
      stop_grace_period_secs: Number of seconds given to threads to stop after
        request_stop() has been called.

    Raises:
      RuntimeError: If any thread is still alive after request_stop()
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
        exc_info = self._exc_info_to_raise
        raise exc_info[0], exc_info[1], exc_info[2]
      elif stragglers:
        raise RuntimeError("Coordinator stopped with threads still running: %s",
                           " ".join(stragglers))
