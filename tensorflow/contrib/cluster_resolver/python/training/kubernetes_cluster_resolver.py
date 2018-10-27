# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Cluster Resolvers for Kubernetes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.cluster_resolver.python.training.cluster_resolver import ClusterResolver
from tensorflow.python.training import server_lib

_KUBERNETES_API_CLIENT_INSTALLED = True
try:
  from kubernetes import client as k8sclient  # pylint: disable=g-import-not-at-top
  from kubernetes import config as k8sconfig  # pylint: disable=g-import-not-at-top
except ImportError:
  _KUBERNETES_API_CLIENT_INSTALLED = False


class KubernetesClusterResolver(ClusterResolver):
  """Cluster Resolver for Kubernetes.

  This is an implementation of cluster resolvers for Kubernetes. When given the
  the Kubernetes namespace and label selector for pods, we will retrieve the
  pod IP addresses of all running pods matching the selector, and return a
  ClusterSpec based on that information.
  """

  def __init__(self,
               job_to_label_mapping=None,
               tf_server_port=8470,
               override_client=None):
    """Initializes a new KubernetesClusterResolver.

    This initializes a new Kubernetes Cluster Resolver. The Cluster Resolver
    will attempt to talk to the Kubernetes master to retrieve all the instances
    of pods matching a label selector.

    Args:
      job_to_label_mapping: A mapping of TensorFlow jobs to label selectors.
        This allows users to specify many TensorFlow jobs in one Cluster
        Resolver, and each job can have pods belong with different label
        selectors. For example, a sample mapping might be
        ```
        {'worker': ['job-name=worker-cluster-a', 'job-name=worker-cluster-b'],
         'ps': ['job-name=ps-1', 'job-name=ps-2']}
        ```
      tf_server_port: The port the TensorFlow server is listening on.
      override_client: The Kubernetes client (usually automatically retrieved
        using `from kubernetes import client as k8sclient`). If you pass this
        in, you are responsible for setting Kubernetes credentials manually.

    Raises:
      ImportError: If the Kubernetes Python client is not installed and no
        `override_client` is passed in.
    """
    if _KUBERNETES_API_CLIENT_INSTALLED:
      k8sconfig.load_kube_config()

    if not job_to_label_mapping:
      job_to_label_mapping = {'worker': ['job-name=tensorflow']}

    if not override_client and not _KUBERNETES_API_CLIENT_INSTALLED:
      raise ImportError('The Kubernetes Python client must be installed before'
                        'using the Kubernetes Cluster Resolver. To install the'
                        'Kubernetes Python client, run `pip install '
                        'kubernetes` on your command line.')

    self._job_to_label_mapping = job_to_label_mapping
    self._tf_server_port = tf_server_port
    self._override_client = override_client

  def master(self):
    # TODO(frankchn): Figure out a standard way to pass in the current task type
    # and task id via Kubernetes.
    pass

  def get_master(self):
    return self.master()

  def get_job_name(self):
    return self._job_name

  def cluster_spec(self):
    """Returns a ClusterSpec object based on the latest info from Kubernetes.

    We retrieve the information from the Kubernetes master every time this
    method is called.

    Returns:
      A ClusterSpec containing host information returned from Kubernetes.

    Raises:
      RuntimeError: If any of the pods returned by the master is not in the
        `Running` phase.
    """
    if not self._override_client:
      k8sconfig.load_kube_config()

    client = self._override_client or k8sclient.CoreV1Api()
    cluster_map = {}

    for tf_job in self._job_to_label_mapping:
      all_pods = []
      for selector in self._job_to_label_mapping[tf_job]:
        ret = client.list_pod_for_all_namespaces(label_selector=selector)
        selected_pods = []

        # Sort the list by the name to make sure it doesn't change call to call.
        for pod in sorted(ret.items, key=lambda x: x.metadata.name):
          if pod.status.phase == 'Running':
            selected_pods.append(
                '%s:%s' % (pod.status.host_ip, self._tf_server_port))
          else:
            raise RuntimeError('Pod "%s" is not running; phase: "%s"' %
                               (pod.metadata.name, pod.status.phase))
        all_pods.extend(selected_pods)
      cluster_map[tf_job] = all_pods

    return server_lib.ClusterSpec(cluster_map)
