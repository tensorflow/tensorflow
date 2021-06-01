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

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training import server_lib
from tensorflow.python.util.tf_export import tf_export


@tf_export('distribute.cluster_resolver.KubernetesClusterResolver')
class KubernetesClusterResolver(ClusterResolver):
  """ClusterResolver for Kubernetes.

  This is an implementation of cluster resolvers for Kubernetes. When given the
  the Kubernetes namespace and label selector for pods, we will retrieve the
  pod IP addresses of all running pods matching the selector, and return a
  ClusterSpec based on that information.

  Note: it cannot retrieve `task_type`, `task_id` or `rpc_layer`. To use it
  with some distribution strategies like
  `tf.distribute.experimental.MultiWorkerMirroredStrategy`, you will need to
  specify `task_type` and `task_id` by setting these attributes.

  Usage example with tf.distribute.Strategy:

    ```Python
    # On worker 0
    cluster_resolver = KubernetesClusterResolver(
        {"worker": ["job-name=worker-cluster-a", "job-name=worker-cluster-b"]})
    cluster_resolver.task_type = "worker"
    cluster_resolver.task_id = 0
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)

    # On worker 1
    cluster_resolver = KubernetesClusterResolver(
        {"worker": ["job-name=worker-cluster-a", "job-name=worker-cluster-b"]})
    cluster_resolver.task_type = "worker"
    cluster_resolver.task_id = 1
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)
    ```
  """

  def __init__(self,
               job_to_label_mapping=None,
               tf_server_port=8470,
               rpc_layer='grpc',
               override_client=None):
    """Initializes a new KubernetesClusterResolver.

    This initializes a new Kubernetes ClusterResolver. The ClusterResolver
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
      rpc_layer: (Optional) The RPC layer TensorFlow should use to communicate
        between tasks in Kubernetes. Defaults to 'grpc'.
      override_client: The Kubernetes client (usually automatically retrieved
        using `from kubernetes import client as k8sclient`). If you pass this
        in, you are responsible for setting Kubernetes credentials manually.

    Raises:
      ImportError: If the Kubernetes Python client is not installed and no
        `override_client` is passed in.
      RuntimeError: If autoresolve_task is not a boolean or a callable.
    """
    try:
      from kubernetes import config as k8sconfig  # pylint: disable=g-import-not-at-top

      k8sconfig.load_kube_config()
    except ImportError:
      if not override_client:
        raise ImportError('The Kubernetes Python client must be installed '
                          'before using the Kubernetes Cluster Resolver. '
                          'To install the Kubernetes Python client, run '
                          '`pip install kubernetes` on your command line.')

    if not job_to_label_mapping:
      job_to_label_mapping = {'worker': ['job-name=tensorflow']}

    self._job_to_label_mapping = job_to_label_mapping
    self._tf_server_port = tf_server_port
    self._override_client = override_client

    self.task_type = None
    self.task_id = None
    self.rpc_layer = rpc_layer

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    """Returns the master address to use when creating a session.

    You must have set the task_type and task_id object properties before
    calling this function, or pass in the `task_type` and `task_id`
    parameters when using this function. If you do both, the function parameters
    will override the object properties.

    Note: this is only useful for TensorFlow 1.x.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC protocol for the given cluster.

    Returns:
      The name or URL of the session master.
    """
    task_type = task_type if task_type is not None else self.task_type
    task_id = task_id if task_id is not None else self.task_id

    if task_type is not None and task_id is not None:
      return format_master_url(
          self.cluster_spec().task_address(task_type, task_id),
          rpc_layer or self.rpc_layer)

    return ''

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
    if self._override_client:
      client = self._override_client
    else:
      from kubernetes import config as k8sconfig  # pylint: disable=g-import-not-at-top
      from kubernetes import client as k8sclient  # pylint: disable=g-import-not-at-top

      k8sconfig.load_kube_config()
      client = k8sclient.CoreV1Api()

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
