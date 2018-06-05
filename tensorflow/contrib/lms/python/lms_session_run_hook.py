# (C) Copyright IBM Corp. 2018. All Rights Reserved.
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
from tensorflow.python.framework import ops
from tensorflow.python.training import session_run_hook
from tensorflow.contrib.lms.python.lms import LMS


class LMSHook(session_run_hook.SessionRunHook):
    ''' This hook is to modify the input graph for Large Model Support
    by adding swap operations.
    '''
    def __init__(self, optimizer_scopes, **kwargs):
        """Create an LMSHook object to edit the graph for supporting large model.

        Args:
          optimizer_scopes: a set of scopes for the optimizers/solvers.
          kwargs: the kwargs to pass to LMS. Note, the `graph` argument is
                  removed from the kwargs before initializing LMS because
                  the graph is obtained automatically by the SessionRunHook and
                  is generally not available at hook initilization time.
        """
        kwargs.pop('graph', None)
        self.lms_obj = LMS(optimizer_scopes, **kwargs)

    def begin(self):
        self.lms_obj.run(ops.get_default_graph())
