#ifndef TENSORFLOW_CONTRIB_MPI_H_
#define TENSORFLOW_CONTRIB_MPI_H_

#ifdef TENSORFLOW_USE_MPI

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

#if GOOGLE_CUDA
#include "cuda_runtime.h"
#endif

// Needed to avoid header issues with C++-supporting MPI implementations
#define OMPI_SKIP_MPICXX
#include "third_party/mpi/mpi.h"

#define TAG_TENSOR 12

namespace tensorflow {
namespace contrib {
namespace mpi {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// Convert from templated types to values we can pass to MPI.
template<typename T>
MPI_Datatype MPIType();

// Convert from templated types to TensorFlow data types.
template<typename T>
DataType TensorFlowDataType();

#define MPI_REQUIRES_OK(MPI_STATUS)                                     \
    if ((MPI_STATUS) != MPI_SUCCESS) {                                   \
        return errors::Unknown("MPI operation failed unexpectedly.");   \
    }

// Copy data from one tensor to another tensor.
// This uses a custom CUDA stream on GPU, which is necessary to overlay the
// backpropagation computations with the allreduce.
template <typename Device>
void CopyTensorData(void* destination, void* source, size_t size);

// Add a tensor into another tensor, accumulating in place.
// This uses a custom CUDA stream on GPU, which is necessary to overlay the
// backpropagation computations with the allreduce.
template <typename Device, typename T>
void AccumulateTensorData(T* destination, T* source, size_t size);

// We need to get the right stream for doing CUDA memory transfers and
// operations, which is possibly different from the standard TensorFlow stream.
#if GOOGLE_CUDA
cudaStream_t CudaStreamForMPI();
#endif

/* Perform a ring allreduce on the data. Allocate the necessary output tensor and
 * store it in the output parameter.
 *
 * Assumes that all MPI processes are doing an allreduce of the same tensor,
 * with the same dimensions.
 *
 * A ring allreduce is a bandwidth-optimal way to do an allreduce. To do the allreduce,
 * the nodes involved are arranged in a ring:
 *
 *                   .--0--.
 *                  /       \
 *                 3         1
 *                  \       /
 *                   *--2--*
 *
 *  Each node always sends to the next clockwise node in the ring, and receives
 *  from the previous one.
 *
 *  The allreduce is done in two parts: a scatter-reduce and an allgather. In
 *  the scatter reduce, a reduction is done, so that each node ends up with a
 *  chunk of the final output tensor which has contributions from all other
 *  nodes.  In the allgather, those chunks are distributed among all the nodes,
 *  so that all nodes have the entire output tensor.
 *
 *  Both of these operations are done by dividing the input tensor into N
 *  evenly sized chunks (where N is the number of nodes in the ring).
 *
 *  The scatter-reduce is done in N-1 steps. In the ith step, node j will send
 *  the (j - i)th chunk and receive the (j - i - 1)th chunk, adding it in to
 *  its existing data for that chunk. For example, in the first iteration with
 *  the ring depicted above, you will have the following transfers:
 *
 *      Segment 0:  Node 0 --> Node 1
 *      Segment 1:  Node 1 --> Node 2
 *      Segment 2:  Node 2 --> Node 3
 *      Segment 3:  Node 3 --> Node 0
 *
 *  In the second iteration, you'll have the following transfers:
 *
 *      Segment 0:  Node 1 --> Node 2
 *      Segment 1:  Node 2 --> Node 3
 *      Segment 2:  Node 3 --> Node 0
 *      Segment 3:  Node 0 --> Node 1
 *
 *  After this iteration, Node 2 has 3 of the four contributions to Segment 0.
 *  The last iteration has the following transfers:
 *
 *      Segment 0:  Node 2 --> Node 3
 *      Segment 1:  Node 3 --> Node 0
 *      Segment 2:  Node 0 --> Node 1
 *      Segment 3:  Node 1 --> Node 2
 *
 *  After this iteration, Node 3 has the fully accumulated Segment 0; Node 0
 *  has the fully accumulated Segment 1; and so on. The scatter-reduce is complete.
 *
 *  Next, the allgather distributes these fully accumululated chunks across all nodes.
 *  Communication proceeds in the same ring, once again in N-1 steps. At the ith step,
 *  node j will send chunk (j - i + 1) and receive chunk (j - i). For example, at the
 *  first iteration, the following transfers will occur:
 *
 *      Segment 0:  Node 3 --> Node 0
 *      Segment 1:  Node 0 --> Node 1
 *      Segment 2:  Node 1 --> Node 2
 *      Segment 3:  Node 2 --> Node 3
 *
 * After the first iteration, Node 0 will have a fully accumulated Segment 0
 * (from Node 3) and Segment 1. In the next iteration, Node 0 will send its
 * just-received Segment 0 onward to Node 1, and receive Segment 3 from Node 3.
 * After this has continued for N - 1 iterations, all nodes will have a the fully
 * accumulated tensor.
 *
 * Each node will do (N-1) sends for the scatter-reduce and (N-1) sends for the allgather.
 * Each send will contain K / N bytes, if there are K bytes in the original tensor on every node.
 * Thus, each node sends and receives 2K(N - 1)/N bytes of data, and the performance of the allreduce
 * (assuming no latency in connections) is constrained by the slowest interconnect between the nodes.
 *
 */
template<typename Device, typename T>
Status RingAllreduce(OpKernelContext* context, const Tensor* input,
                     Tensor* temp, Tensor* output) {
    // Acquire MPI size and rank
    int n, r;
    MPI_REQUIRES_OK(MPI_Comm_size(MPI_COMM_WORLD, &n));
    MPI_REQUIRES_OK(MPI_Comm_rank(MPI_COMM_WORLD, &r));

    T* buffer = (T*) output->tensor_data().data();

    CopyTensorData<Device>((void*) buffer,
            (void*) input->tensor_data().data(),
            output->tensor_data().size());

    // Calculate segment sizes and segment ends
    const size_t elements_to_reduce = input->NumElements();
    const size_t segment_size = elements_to_reduce / n;
    std::vector<size_t> segment_sizes(n, segment_size);

    const size_t residual = elements_to_reduce % n;
    for (size_t i = 0; i < residual; ++i) {
        segment_sizes[i]++;
    }

    std::vector<size_t> segment_starts(n);
    segment_starts[0] = 0;
    for (size_t i = 1; i < segment_starts.size(); ++i) {
        segment_starts[i] = segment_starts[i-1] + segment_sizes[i-1];
    }

    assert(segment_starts[n-1] + segment_sizes[n-1] == elements_to_reduce);

    T* segment_recv = (T*) temp->tensor_data().data();

    // Receive from your left neighbor with wrap-around
    const size_t recv_from = ((r - 1) + n) % n;

    // Send to your right neighbor with wrap-around
    const size_t send_to = (r + 1) % n;

    MPI_Status recv_status;
    MPI_Request recv_req;

    // Now start ring. At every step, for every rank, we iterate through
    // segments with wraparound and send and recv from our neighbors and reduce
    // locally. At the i'th iteration, rank r, sends segment (r-i) and receives
    // segment (r-i-1).
    for (int i = 0; i < n - 1; i++) {
        const size_t send_seg_id = ((r-i) + n) % n;
        const size_t recv_seg_id = ((r-i-1) + n) % n;

        T* segment_send = &(buffer[segment_starts[send_seg_id]]);

        MPI_REQUIRES_OK(MPI_Irecv(segment_recv, segment_sizes[recv_seg_id],
                                  MPIType<T>(), recv_from, TAG_TENSOR,
                                  MPI_COMM_WORLD, &recv_req));

        MPI_REQUIRES_OK(MPI_Send(segment_send, segment_sizes[send_seg_id],
                                 MPIType<T>(), send_to, TAG_TENSOR,
                                 MPI_COMM_WORLD));

        T *segment_update = &(buffer[segment_starts[recv_seg_id]]);

        // Wait for recv to complete before reduction
        MPI_REQUIRES_OK(MPI_Wait(&recv_req, &recv_status));

        const size_t recv_seg_size = segment_sizes[recv_seg_id];
        AccumulateTensorData<Device, T>(
                segment_update, segment_recv, recv_seg_size);
    }

    // Now start pipelined ring allgather. At every step, for every rank, we
    // iterate through segments with wraparound and send and recv from our
    // neighbors. At the i'th iteration, rank r, sends segment (r-i+1) and
    // receives segment (r-i).
    for (size_t i = 0; i < n - 1; ++i) {
        const size_t send_seg_id = ((r-i+1) + n) % n;
        const size_t recv_seg_id = ((r-i) + n) % n;

        // Segment to send - at every iteration we send segment (r-i+1)
        T* segment_send = &(buffer[segment_starts[send_seg_id]]);

        // Segment to recv - at every iteration we receive segment (r-i)
        T* segment_recv = &(buffer[segment_starts[recv_seg_id]]);

        MPI_REQUIRES_OK(MPI_Sendrecv(segment_send, segment_sizes[send_seg_id],
                 MPIType<T>(), send_to, TAG_TENSOR, segment_recv,
                 segment_sizes[recv_seg_id], MPIType<T>(), recv_from,
                 TAG_TENSOR, MPI_COMM_WORLD, &recv_status));
    }

    return Status::OK();
}

// Perform a ring allgather on a Tensor. Other ranks may allgather with a
// tensor which differs in the first dimension only; all other dimensions must
// be the same.
//
// For more information on the ring allgather, read the documentation for the
// ring allreduce, which includes a ring allgather.
template<typename Device, typename T>
Status RingAllgather(OpKernelContext* context, const Tensor* input,
                     const std::vector<size_t>& sizes, Tensor* output) {
    // Acquire MPI size and rank
    int n, r;
    MPI_REQUIRES_OK(MPI_Comm_size(MPI_COMM_WORLD, &n));
    MPI_REQUIRES_OK(MPI_Comm_rank(MPI_COMM_WORLD, &r));

    assert(sizes.size() == n);
    assert(input->dim_size(0) == sizes[r]);

    // Compute number of elements in every "row". We can't compute number of
    // elements in every chunks, because those chunks are variable length.
    size_t elements_per_row = 1;
    for (int i = 1; i < input->shape().dims(); i++) {
        elements_per_row *= input->dim_size(i);
    }

    // Copy data from input tensor to correct place in output tensor.
    std::vector<size_t> segment_starts(n);
    segment_starts[0] = 0;
    for (int i = 1; i < n; i++) {
        segment_starts[i] = segment_starts[i - 1] + elements_per_row * sizes[i - 1];
    }
    size_t offset = segment_starts[r];

    // Copy data to the right offset for this rank.
    T* buffer = (T*) output->tensor_data().data();
    CopyTensorData<Device>((void*) (buffer + offset),
            (void*) input->tensor_data().data(),
            elements_per_row * sizes[r] * sizeof(T));

    // Receive from your left neighbor with wrap-around
    const size_t recv_from = ((r - 1) + n) % n;

    // Send to your right neighbor with wrap-around
    const size_t send_to = (r + 1) % n;

    // Perform a ring allgather. At every step, for every rank, we iterate
    // through segments with wraparound and send and recv from our neighbors.
    // At the i'th iteration, rank r, sends segment (r-i) and receives segment
    // (r-1-i).
    MPI_Status recv_status;
    for (size_t i = 0; i < n - 1; ++i) {
        const size_t send_seg_id = ((r-i) + n) % n;
        const size_t recv_seg_id = ((r-i-1) + n) % n;

        // Segment to send - at every iteration we send segment (r-i)
        size_t offset_send = segment_starts[send_seg_id];
        size_t rows_send = sizes[send_seg_id];
        T* segment_send = &(buffer[offset_send]);

        // Segment to recv - at every iteration we receive segment (r-1-i)
        size_t offset_recv = segment_starts[recv_seg_id];
        size_t rows_recv = sizes[recv_seg_id];
        T* segment_recv = &(buffer[offset_recv]);

        MPI_REQUIRES_OK(MPI_Sendrecv(
                    segment_send, elements_per_row * rows_send,
                    MPIType<T>(), send_to, TAG_TENSOR, segment_recv,
                    elements_per_row * rows_recv, MPIType<T>(), recv_from,
                    TAG_TENSOR, MPI_COMM_WORLD, &recv_status));
    }

    return Status::OK();
}

}
}
}

#endif // TENSORFLOW_USE_MPI

#undef TENSORFLOW_CONTRIB_MPI_H_
#endif // TENSORFLOW_CONTRIB_MPI_H_
