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

from tensorflow.python.keras.callbacks import Callback

from tensorflow.contrib.lms import LMS
from tensorflow.python.framework import ops


class LMSKerasCallback(Callback):
    """This callback is to modify the input graph for Large Model Support
    during Keras training / fit by adding swap operations.
    """

    def __init__(self, optimizer_scopes_override=None, **kwargs):
        """Create an LMSKerasCallback object to edit the graph for
           supporting large model tensor swapping when using TensorFlow Keras.

        Args:
          optimizer_scopes_override: by default the LMSKerasCallback will
                automatically discover the optimizer scopes from the Keras
                model. This parameter allows overriding that automatic
                discovery with a set of optimizer scope names.
          kwargs: the kwargs to pass to LMS. Note, the `graph` argument is
                  removed from the kwargs and not used for initializing LMS
                  because the graph is obtained automatically by the
                  Keras callback during the set_model method.
        """
        self._optimizer_scopes = optimizer_scopes_override
        self._lms_args = kwargs
        self._lms_args.pop('graph', None)

    def set_model(self, model):
        self.model = model
        optimizer_scopes = self._optimizer_scopes
        if not self._optimizer_scopes:
            optimizer_name = self.model.optimizer.__class__.__name__
            optimizer_scopes = {'training/'+optimizer_name+'/gradients'}

        lmsMod = LMS(optimizer_scopes,
                     graph=ops.get_default_graph(),
                     **self._lms_args)
        lmsMod.run()
