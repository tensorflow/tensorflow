# Testing Distributed Runtime in TensorFlow

This folder contains tools and test suites for GRPC-based and Allreduce-based
distributed runtimes in TensorFlow.

There are three general modes of testing:

**1) Launch a docker container and run parameters servers and workers as
 separate processes therein.**

For example:

    ./local_test.sh

By default, local_test.sh runs the MNIST-with-replicas model as a test.
However, you can use the --model_name flag to run the tf-learn/wide&deep
cesnsu model:

    ./local_test.sh --model_name CENSUS_WIDENDEEP

You can test specify version of TensorFlow:

```shell
./local_test.sh ${whl_file_url}
```

For example, you can find these TensorFlow python package URLs from [here](https://www.tensorflow.org/install/pip) for Ubuntu.

**2) Launch a remote k8s cluster on Google Kubernetes Engine (GKE) and run the
test suite on it**

For example:

    export TF_DIST_GCLOUD_PROJECT="tensorflow-testing"
    export TF_DIST_GCLOUD_COMPUTE_ZONE="us-central1-f"
    export TF_DIST_CONTAINER_CLUSTER="test-cluster-1"
    export TF_DIST_GCLOUD_KEY_FILE="/var/gcloud-secrets/my-gcloud-key.json"
    ./remote_test.sh

Here you specify the Google Compute Engine (GCE) project, compute zone and
container cluster with the first three environment variables, in that order.
The environment variable "TF_DIST_GCLOUD_KEY_FILE_DIR" is a directory in which
the JSON service account key file named "tensorflow-testing.json" is located.
You can use the flag "--setup-cluster-only" to perform only the cluster setup
step and skip the testing step:

    ./remote_test.sh --setup_cluster_only

**3) Run the test suite on an existing k8s TensorFlow cluster**

For example:

    export TF_DIST_GRPC_SERVER_URL="grpc://11.22.33.44:2222"
    ./remote_test.sh

The IP address above is a dummy example. Such a cluster may have been set up
using the command described at the end of the previous section.


**Asynchronous and synchronous parameter updates**

There are two modes for the coordination of the parameters from multiple
workers: asynchronous and synchronous.

In the asynchronous mode, the parameter updates (gradients) from the workers
are applied to the parameters without any explicit coordination. This is the
default mode in the tests.

In the synchronous mode, a certain number of parameter updates are aggregated
from the model replicas before the update is applied to the model parameters.
To use this mode, do:

    # For remote testing
    ./remote_test.sh --sync_replicas

    # For local testing
    ./local_test.sh --sync_replicas


**Specifying the number of workers**

You can specify the number of workers by using the --num-workers option flag,
e.g.,

    # For remote testing
    ./remote_test.sh --num_workers 4

    # For local testing
    ./local_test.sh --num_workers 4


**Building the GRPC server Docker image**

To build the Docker image for a test server of TensorFlow distributed runtime,
run:

    ./build_server.sh <docker_image_name>

**Using the GRPC server Docker image**
To launch a container as a TensorFlow GRPC server, do as the following example:

    docker run tensorflow/tf_grpc_server --cluster_spec="worker|localhost:2222;foo:2222,ps|bar:2222;qux:2222" --job_name=worker --task_id=0

**Generating configuration file for TensorFlow k8s clusters**

The script at "scripts/k8s_tensorflow.py" can be used to generate yaml
configuration files for a TensorFlow k8s cluster consisting of a number of
workers and parameter servers. For example:

    scripts/k8s_tensorflow.py \
        --num_workers 2 \
        --num_parameter_servers 2 \
        --grpc_port 2222 \
        --request_load_balancer true \
        --docker_image "tensorflow/tf_grpc_server" \
        > tf-k8s-with-lb.yaml

The yaml configuration file generated in the previous step can be used to a
create a k8s cluster running the specified numbers of worker and parameter
servers. For example:

    kubectl create -f tf-k8s-with-lb.yaml

See [Kubernetes kubectl documentation](http://kubernetes.io/docs/user-guide/kubectl-overview/)
for more details.

**Create allreduce-based Tensorflow k8s deployment**

The allreduce-based Tensorflow, Horovod, is an open source distributed deep
learning framework for TensorFlow, detailed information can be found in
https://arxiv.org/pdf/1802.05799.pdf.

The script "scripts_allreduce/k8s_deploy_tensorflow.sh" can be used to create or
delete an allreduce-based Tensorflow k8s deployment with specified number of
containers.

Create a deployment containing a number of containers and enable passwordless
ssh between the containers (optional: enable host network mode with --hostnet
and --port <container_ssh_port>):

    scripts_allreduce/k8s_deploy_tensorflow.sh \
        --num_containers <num_of_containers> \
        --image <docker_image> \
        --deployment <deployment_name> \
        --config_map <config_map>

Delete a deployment and config_map in k8s cluster:

    scripts_allreduce/k8s_deploy_tensorflow.sh \
        --deployment <deployment_name> \
        --config_map <config_map> \
        --delete

Upload file or directory to all the containers of a deployment:

    scripts_allreduce/k8s_deploy_tensorflow.sh \
        --cp --src <path_to_local_directory> \
        --dest <path_to_directory_on_containers> \
        --deployment <deployment_name>
