# Performance

## Get an XProf session

Follow the instructions in the [XProf
documentation](https://openxla.org/xprof/capturing_profiles) to generate an
XProf trace for your problematic run.

## Check for shortage of mapped DMA buffers

The Megascale XLA runtime needs to register host memory before it can be used
for DMAs to and from the TPU. This happens early after the process starts. If
you see these registrations (`MapDmaBuffer` calls) at steady-state then it
indicates that something is wrong. Look for the presence of these calls in XProf
Trace Viewer. See the screenshot below for reference.

**Tip:** Search for the exact worker name, because there can be other workers
with similar or close names. You also search for the term “MapDmaBuffer” on the
page.

![Example xprof trace showing MapDmaBuffer
calls](./images/map_dma_buffer_example_trace.png)

If the issue is observed then try to increase the size of the premapped memory
region by increasing the value of `--megascale_grpc_premap_memory_bytes`,
restarting the job, then checking again.

## Check for memory copies during network transfers

Megascale XLA network transfers are zero-copy by design. However, there are
cases where memory copies will occur and cause degraded performance. Look for
memory copies in Megascale's "Communication Transport" traces as shown in the
example screenshot below.

![Example "Megascale: Memory Copy" trace during network
receive](./images/memory_copy_example_trace.png)

If the issue is observed then try to increase the size of the premapped memory
region by increasing the value of `--megascale_grpc_premap_memory_bytes`,
restarting the job, then checking again.

## Network Analysis

MegaScale also provides a Colab notebook to help analyze network performance
using an XProf trace.

TODO: Upload network analysis colab and link to it here.

### Collective Slack Too Small

One indicator that your workload is not optimized for compute/communication
overlap is seeing small slack times for a subset of collectives. This can
manifest as longer than expected `recv-done` traces in the trace viewer, or as
collectives with zero or near-zero
[slack](https://cloud.google.com/tpu/docs/troubleshooting/troubleshoot-multislice#slack_time).

If this is the case, look towards identifying bottlenecks in your workload that
may be causing parts of your program to not overlap compute and network
communication.

### High Network Bandwidth Demand

If you are observing long `recv-done` op latencies within your model XProf, this
could be an indication that the model is 'Bandwidth Bound' in those portions of
the step function (is blocked by available network bandwidth in the system).

You can generate a timeline of network usage for your model. If you see
consistently high network usage throughout the step, or specific regions with
large spikes, then your model may be bandwidth bound in those regions.

To mitigate bandwidth bound models, you can:

1.  Check the [Collective Slack](#collective-slack-too-small) of your model.
    Models with many collectives with low slack will have bandwidth bound
    regions.
2.  Confirm that the network settings are optimized.
3.  Examine your model structure and data sharding to see if there are ways to
    increase computation/communication overlap.
4.  (Data parallel models) Confirm that you have sufficient batch size in each
    local replica to overlap with the communication.

### High Network Latency

If the bandwidth is not saturated, you may want to generate the RPC latency
timeline. If you see high RPC latencies is constantly or sporadically high, this
means that there are some issues with the MXLA RPCs.

High RPC latencies on Cloud environment are often caused by suboptimal TCP
configuration. Please confirm if all TCP parameters are configured properly
within the container.

If any of the TCP parameters are not correctly configured, consult Google Cloud
ML Compute Services (CMCS) team on how to configure them properly.

### HLO Dump

Please follow [these steps](https://openxla.org/xla/hlo_dumps) to dump HLO to
the local filesystem on the TPU worker. You may need to upload the dump to GCS
in order to share them with the XLA or Megascale team.
