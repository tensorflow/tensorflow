# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Framework of debug wrapper sessions.

A debug wrapper session is a wrapper around a TensorFlow Python Session.
The wrapper preserves the Session interface, most importantly the run() method,
while providing abilities to:
a) Intercept a run() call to a wrapped session and insert debug tensor watches
   according to externally-specified debug URLs.

b) Release control to an external (i.e., non-Session) object before and after
   the run() call, so that the external object can perform actions such as
   launching a UI to let users inspect the intermediate tensors and partition
   graphs from the run() call.

c) (To be implemented) Intercept a run() call and give control to DebugStepper
   to let it perform stepping / continuing-to actions on the graph.

b) (To be implemented in a future CL) Enter an instruction loop to let an
   external object (e.g., remote client) launch run() and cont() calls
   remotely.

*** The lifetime of a debug wrapper session: ***

1) The wrapper session is created by calling the constructor with a
   wrapped (normal) session as the argument:
     wrapper = FooDebugWrapperSession(sess)
   wherein FooDebugWrapperSession is a concrete subclass implementing the
   abstract BaseDebugWrapperSession class below.

2) Near the end of the constructor call, the on_session_init() callback is
   invoked, with a OnSessionInitRequest object as the argument. The object
   carries the wrapped (normal) session object.

3) The callback handles the request and returns a OnSessionInitResponse
   object with an action field, directing the wrapper session what to do next.

If the action field in the OnSessionInitResponse is PROCEED, the constuctor
returns. Control is released back to the caller of the constructor, which can
invoke run() method of wrapper session with the same syntax as a non-wrapped
session, e.g.,:
  wrapper.run(fetches, feed_dict=feeds, options=run_options)

Below, A1 - A2 is the lifetime of a wrapper run() call if the action is
PROCEED:

A1) Right at the start of each run() call, the on_run_start() callback is
    invoked, with an OnRunStartRequest object carrying information such as
    the fetches, the feed dict, the run options and run metadata used in
    this run call, along with a count of how many run calls has occurred
    on this wrapper session. The callback then returns an OnRunStartResponse
    object, of which the action field directs what the wrapper session
    actually will do of the run() call.

    If the action is DEBUG_RUN, a debugged (tensor-watched) run will ensue,
    with the debug URLs supplied in the debug_urls field of the response.
    These can be file:// or grpc:// URLs, for example.

    If the action is NON_DEBUG_RUN, a non-debug (normal) run will ensue.

    If the action is INVOKE_STEPPER, no run() call will be issued to the
    wrapped session. But instead, a DebugStepper (i.e., "continuation
    debugger") will be used to perform stepping / continue-to actions on
    the graph.

TODO(cais): The event loop for the DebugStepper will request additional
   callbacks including on_cont_start() and on_cont_end(). Add those.

A2) Right before the run() returns, the on_run_end() callback is invoked,
    with an OnRunEndRequest object as the argument, which carries information
    including the actual action performed in the warpper run() call and the
    run_metadata from the run() call.

However, if the action field in OnSessionInitResponse is
REMOTE_INSTR_LOOP, the constructor will automatically invoke an instruction loop
that gives the control to a remote caller.

In the remote instruction loop, the following steps will happen:

B1) Callback on_instr_start() is invoked. The callback will return an
    OnInstrStartResponse object with an action field which can order one of
    the following actions:
        i) a run() call with fetches, feeds and debug_urls specified.
       ii) a DebugStepper cont() call with target specified.
      iii) value overrides in the cached tensors from the DebugStepper.
       iv) exit the instruction loop.

B2) The wrapper session carries out the action specified above.

B3) If still in the instruction loop, the wrapper session invokes the
    on_instr_end() callback. After the on_instr_end() callback returns, jump
    back to B1.

TODO(cais): Implemented the instruction loop in B1 - B3.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug import debug_utils


# Helper function.
def _check_type(obj, expected_type):
  """Check if an object is of the expected type.

  Args:
    obj: The object being checked.
    expected_type: (type) The expected type of obj.

  Raises:
      TypeError: If obj is not an instance of expected_type.
  """
  if not isinstance(obj, expected_type):
    raise TypeError("Expected type %s; got type %s" %
                    (expected_type, type(obj)))


class OnSessionInitRequest(object):
  """Request to an on-session-init callback.

  This callback is invoked during the __init__ call to a debug-wrapper session.
  """

  def __init__(self, sess):
    """Constructor.

    Args:
      sess: A tensorflow Session object.
    """

    _check_type(sess, session.Session)
    self.session = sess


class OnSessionInitAction(object):
  """Enum-like values for possible action to take on session init."""

  # Proceed, without special actions, in the wrapper session initializaton. What
  # action the wrapper session performs next is determined by the caller of the
  # wrapper session. E.g., it can call run().
  PROCEED = "proceed"

  # Instead of letting the caller of the wrapper session determine what actions
  # the wrapper session will perform next, enter a loop to receive instructions
  # from a remote client.
  # For example, TensorBoard visual debugger can use this action so that it can
  # launch session.run() calls remotely.
  REMOTE_INSTR_LOOP = "remote_instr_loop"


class OnSessionInitResponse(object):
  """Response from an on-session-init callback."""

  def __init__(self, action):
    """Constructor.

    Args:
      action: (OnSessionInitAction) Debugger action to take on session init.
    """
    _check_type(action, str)
    self.action = action


class OnRunStartRequest(object):
  """Request to an on-run-start callback.

  This callback is invoked during a run() call of the debug-wrapper
  session, immediately after the run() call counter is incremented.
  """

  def __init__(self, fetches, feed_dict, run_options, run_metadata,
               run_call_count):
    """Constructor of OnRunStartRequest.

    Args:
      fetches: Fetch targets of the run() call.
      feed_dict: The feed dictionary to the run() call.
      run_options: RunOptions input to the run() call.
      run_metadata: RunMetadata input to the run() call.
        The above four arguments are identical to the input arguments to the
        run() method of a non-wrapped TensorFlow session.
      run_call_count: 1-based count of how many run calls (including this one)
        has been invoked.
    """
    self.fetches = fetches
    self.feed_dict = feed_dict
    self.run_options = run_options
    self.run_metadata = run_metadata
    self.run_call_count = run_call_count


class OnRunStartAction(object):
  """Enum-like values for possible action to take on start of a run() call."""

  # Run once with debug tensor-watching.
  DEBUG_RUN = "debug_run"

  # Run without debug tensor-watching.
  NON_DEBUG_RUN = "non_debug_run"

  # Instead of running the fetches as a whole, as would normally happen, invoke
  # the (to-be-implemented) debug stepper.
  # TODO(cais): Remove "to-be-implemented".
  INVOKE_STEPPER = "invoke_stepper"


class OnRunStartResponse(object):
  """Request from an on-run-start callback.

  The caller of the callback can use this response object to specify what
  action the debug-wrapper session actually takes on the run() call.
  """

  def __init__(self, action, debug_urls):
    """Constructor of OnRunStartResponse.

    Args:
      action: (OnRunStartAction) the action actually taken by the wrapped
        session for the run() call.
      debug_urls: (list of str) debug_urls used in watching the tensors during
        the run() call.
    """

    _check_type(action, str)
    self.action = action

    _check_type(debug_urls, list)
    self.debug_urls = debug_urls


class OnRunEndRequest(object):
  """Request to an on-run-end callback.

  The callback is invoked immediately before the wrapped run() call ends.
  """

  def __init__(self, performed_action, run_metadata=None):
    """Constructor for OnRunEndRequest.

    Args:
      performed_action: (OnRunStartAction) Actually-performed action by the
        debug-wrapper session.
      run_metadata: run_metadata output from the run() call (if any).
    """

    _check_type(performed_action, str)
    self.performed_action = performed_action

    if run_metadata is not None:
      _check_type(run_metadata, config_pb2.RunMetadata)
    self.run_metadata = run_metadata


class OnRunEndResponse(object):
  """Response from an on-run-end callback."""

  def __init__(self):

    # Currently only a placeholder.
    pass


class BaseDebugWrapperSession(object):
  """Base class of debug-wrapper session classes.

  Concrete classes that inherit from this class need to implement the abstract
  methods such as on_session_init, on_run_start and on_run_end.
  """

  # TODO(cais): Add on_cont_start and on_cont_end callbacks once the stepper is
  # is available.

  def __init__(self, sess):
    """Constructor of BaseDebugWrapperSession.

    Args:
      sess: An (unwrapped) TensorFlow session instance.

    Raises:
      ValueError: On invalid OnSessionInitAction value.
    """

    _check_type(sess, session.Session)

    # The session being wrapped.
    self._sess = sess

    # Keeps track of number of run calls that have been performed on this
    # debug-wrapper session.
    self._run_call_count = 0

    # Invoke on-session-init callback.
    response = self.on_session_init(OnSessionInitRequest(self._sess))
    _check_type(response, OnSessionInitResponse)

    if response.action == OnSessionInitAction.PROCEED:
      pass
    elif response.action == OnSessionInitAction.REMOTE_INSTR_LOOP:
      # TODO(cais): Implement REMOTE_INSTR_LOOP
      raise NotImplementedError(
          "OnSessionInitAction REMOTE_INSTR_LOOP has not been "
          "implemented.")
    else:
      raise ValueError(
          "Invalid OnSessionInitAction value: %s" % response.action)

  @property
  def session(self):
    return self._sess

  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    """Wrapper around Session.run() that inserts tensor watch options.

    Args:
      fetches: Same as the fetches arg to regular Session.run()
      feed_dict: Same as the feed_dict arg to regular Session.run()
      options: Same as the options arg to regular Session.run()
      run_metadata: Same as the run_metadata to regular Session.run()

    Returns:
      Simply forwards the output of the wrapped Session.run() call.

    Raises:
      ValueError: On invalid OnRunStartAction value.
    """

    self._run_call_count += 1

    # Invoke on-run-start callback and obtain response.
    run_start_resp = self.on_run_start(
        OnRunStartRequest(fetches, feed_dict, options, run_metadata,
                          self._run_call_count))
    _check_type(run_start_resp, OnRunStartResponse)

    if run_start_resp.action == OnRunStartAction.DEBUG_RUN:
      # Decorate RunOption to fill in debugger tensor watch specifications.
      decorated_run_options = options or config_pb2.RunOptions()
      run_metadata = run_metadata or config_pb2.RunMetadata()

      self._decorate_run_options(decorated_run_options,
                                 run_start_resp.debug_urls)

      # Invoke the run() method of the wrapped Session.
      retvals = self._sess.run(
          fetches,
          feed_dict=feed_dict,
          options=decorated_run_options,
          run_metadata=run_metadata)

      # Prepare arg for the on-run-end callback.
      run_end_req = OnRunEndRequest(
          run_start_resp.action, run_metadata=run_metadata)
    elif run_start_resp.action == OnRunStartAction.NON_DEBUG_RUN:
      # Invoke run() method of the wrapped session.
      retvals = self._sess.run(
          fetches,
          feed_dict=feed_dict,
          options=options,
          run_metadata=run_metadata)

      # Prepare arg for the on-run-end callback.
      run_end_req = OnRunEndRequest(run_start_resp.action)
    elif run_start_resp.action == OnRunStartAction.INVOKE_STEPPER:
      # TODO(cais): Implement stepper loop.
      raise NotImplementedError(
          "OnRunStartAction INVOKE_STEPPER has not been implemented.")
    else:
      raise ValueError(
          "Invalid OnRunStartAction value: %s" % run_start_resp.action)

    # Invoke on-run-end callback and obtain response.
    run_end_resp = self.on_run_end(run_end_req)
    _check_type(run_end_resp, OnRunEndResponse)
    # Currently run_end_resp is only a placeholder. No action is taken on it.

    return retvals

  def partial_run(self, handle, fetches, feed_dict=None):
    raise NotImplementedError(
        "partial_run is not implemented for debug-wrapper sessions.")

  def _decorate_run_options(self, run_options, debug_urls):
    """Modify a RunOptions object for debug tensor watching.

    Specifies request for outputting partition graphs. Adds
    debug_tensor_watch_opts with proper debug URLs.

    Args:
      run_options: (RunOptions) the modified RunOptions object.
      debug_urls: (list of str) debug URLs to be entered in run_options.
        debug_tensor_watch_opts.
    """

    run_options.output_partition_graphs = True
    debug_utils.watch_graph(
        run_options, self._sess.graph, debug_urls=debug_urls)

  @abc.abstractmethod
  def on_session_init(self, request):
    """Callback invoked during construction of the debug-wrapper session.

    This is a blocking callback.
    The invocation happens right before the constructor ends.

    Args:
      request: (OnSessionInitRequest) callback request carrying information
        such as the session being wrapped.

    Returns:
      An instance of OnSessionInitResponse.
    """
    pass

  @abc.abstractmethod
  def on_run_start(self, request):
    """Callback invoked on run() calls to the debug-wrapper session.

    This is a blocking callback.
    The invocation happens after the wrapper's run() call is entered,
    after an increment of run call counter.

    Args:
      request: (OnRunStartRequest) callback request object carrying information
        about the run call such as the fetches, feed dict, run options, run
        metadata, and how many run() calls to this wrapper session has occurred.

    Returns:
      An instance of OnRunStartResponse, carrying information to
        1) direct the wrapper session to perform a specified action (e.g., run
          with or without debug tensor watching, invoking the stepper.)
        2) debug URLs used to watch the tensors.
    """
    pass

  @abc.abstractmethod
  def on_run_end(self, request):
    """Callback invoked on run() calls to the debug-wrapper session.

    This is a blocking callback.
    The invocation happens right before the wrapper exits its run() call.

    Args:
      request: (OnRunEndRequest) callback request object carrying information
        such as the actual action performed by the session wrapper for the
        run() call.

    Returns:
      An instance of OnRunStartResponse.
    """
    pass

  # TODO(cais): Add _node_name_regex_whitelist and
  #   _node_op_type_regex_whitelist.
