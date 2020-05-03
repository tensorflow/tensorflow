# Cluster Resolvers

Cluster Resolvers are a new way of specifying cluster information for distributed execution. Built on top of existing `ClusterSpec` framework, Cluster Resolvers allow users to simply specify a configuration and a cluster management service and a `ClusterResolver` will automatically fetch the relevant information from the service and populate `ClusterSpec`s.

`ClusterResolvers` are designed to work well with `ManagedTrainingSession` and `ClusterSpec` propagation so that distributed training sessions remain robust in the face of node and network failures.
