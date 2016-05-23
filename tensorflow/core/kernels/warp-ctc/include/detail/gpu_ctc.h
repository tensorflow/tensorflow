#pragma once

#include "ctc_helper.h"
#include "gpu_ctc_kernels.h"
#include "reduce.h"

template <typename ProbT>
class GpuCTC {
    public:
        GpuCTC(int alphabet_size,
               int minibatch,
               void *workspace,
               CUstream stream) :
            out_dim_(alphabet_size), minibatch_(minibatch),
            gpu_workspace_(workspace), stream_(stream) {};

        // Noncopyable
        GpuCTC(const GpuCTC&) = delete;
        GpuCTC& operator=(const GpuCTC&) = delete;

        ctcStatus_t
        cost_and_grad(const ProbT* const activations,
                      ProbT* grads,
                      ProbT* costs,
                      const int* const flat_labels,
                      const int* const label_lengths,
                      const int* const input_lengths);

        ctcStatus_t
        score_forward(const ProbT* const activations,
                      ProbT* costs,
                      const int* const flat_labels,
                      const int* const label_lengths,
                      const int* const input_lengths);

    private:

        template<int NT, int VT>
        ctcStatus_t launch_alpha_beta_kernels(const ProbT* const probs,
                                              ProbT *grads,
                                              bool compute_alpha,
                                              bool compute_beta);

        ctcStatus_t
        launch_gpu_kernels(const ProbT* const probs,
                           ProbT *grads,
                           size_t config,
                           bool launch_alpha,
                           bool launch_beta);

        ctcStatus_t
        setup_gpu_metadata(const int* const flat_labels,
                           const int* const label_lengths,
                           const int* const input_lengths);

        ctcStatus_t
        create_metadata_and_choose_config(const int* const label_lengths,
                                          const int* const flat_labels,
                                          const int* const input_lengths,
                                          size_t& best_config);

        ctcStatus_t
        compute_probs(const ProbT* const activations);

        ctcStatus_t
        compute_cost_and_score(const ProbT* const activations,
                               ProbT* grads,
                               ProbT* costs,
                               const int* const flat_labels,
                               const int* const label_lengths,
                               const int* const input_lengths,
                               bool compute_alpha,
                               bool compute_betas_and_grad);


        int out_dim_; // Number of characters plus blank
        int minibatch_;

        int S_;
        int T_;

        int activation_cols_; // Number of columns in activations

        CUstream stream_;

        void *gpu_workspace_; // Buffer for all temporary GPU memory
        int *utt_length_; // T
        int *label_sizes_; // L
        int *repeats_; // repeats_
        int *label_offsets_;
        int *labels_without_blanks_;
        int *labels_with_blanks_;
        ProbT *alphas_;
        ProbT *nll_forward_;
        ProbT *nll_backward_;
        ProbT *denoms_; // Temporary storage for denoms for softmax
        ProbT *probs_; // Temporary storage for probabilities (softmax output)
};

template<typename ProbT>
ctcStatus_t
GpuCTC<ProbT>::setup_gpu_metadata(const int* const flat_labels,
                                  const int* const label_lengths,
                                  const int* const input_lengths)
{
    size_t gpu_bytes_used = 0;

    nll_forward_ =
        reinterpret_cast<ProbT *>(static_cast<char*>(gpu_workspace_) +
                                  gpu_bytes_used);
    gpu_bytes_used += minibatch_ * sizeof(ProbT);


    nll_backward_ =
        reinterpret_cast<ProbT *>(static_cast<char*>(gpu_workspace_) +
                                  gpu_bytes_used);
    gpu_bytes_used += minibatch_ * sizeof(ProbT);


    repeats_ =
        reinterpret_cast<int *>(static_cast<char*>(gpu_workspace_) +
                                gpu_bytes_used);
    gpu_bytes_used += minibatch_ * sizeof(int);

    label_offsets_ =
        reinterpret_cast<int *>(static_cast<char*>(gpu_workspace_) +
                                gpu_bytes_used);
    gpu_bytes_used += minibatch_ * sizeof(int);


    // This is the max of all S and T for all valid examples in the minibatch.
    // A valid example is one for which L + repeats <= T
    S_ = 0;
    T_ = 0;

    // This is the max of all timesteps, valid or not. Needed to compute offsets
    int Tmax = 0;

    // This is the max of all labels, valid or not. Needed to compute offsets
    int Lmax = 0;
    int total_label_length = 0;

    constexpr int cpu_buffer_size = 64;
    int repeats[cpu_buffer_size];
    int label_offsets[cpu_buffer_size];

    const int num_passes = ctc_helper::div_up(minibatch_, cpu_buffer_size);

    cudaError_t cuda_status;

    for (int pass = 0; pass < num_passes; ++pass) {

        const int start_idx = pass * cpu_buffer_size;
        const int end_idx = std::min(minibatch_, (pass+1) * cpu_buffer_size);

        for (int j = start_idx; j < end_idx; ++j) {
            const int L = label_lengths[j];
            const int local_T = input_lengths[j];
            const int *label_ptr = &(flat_labels[total_label_length]);

            label_offsets[j % cpu_buffer_size] = total_label_length;
            total_label_length += L;

            int repeat_counter = 0;

            for (int i = 1; i < L; ++i)
                repeat_counter += (label_ptr[i] == label_ptr[i-1]);

            repeats[j % cpu_buffer_size] = repeat_counter;
            const bool valid_label = ((L + repeat_counter) <= local_T);

            // Only update S and T if label is valid
            S_ = (valid_label) ? std::max(S_, L) : S_;
            T_ = (valid_label) ? std::max(T_, local_T) : T_;

            Tmax = std::max(Tmax, local_T);
            Lmax = std::max(Lmax, L);
        }

        cuda_status = cudaMemcpyAsync(&(repeats_[start_idx]), repeats,
                                      (end_idx - start_idx) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_);
        if (cuda_status != cudaSuccess)
            return CTC_STATUS_MEMOPS_FAILED;


        cuda_status = cudaMemcpyAsync(&(label_offsets_[start_idx]), label_offsets,
                                      (end_idx - start_idx) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_);
        if (cuda_status != cudaSuccess)
            return CTC_STATUS_MEMOPS_FAILED;
    }

    S_ = 2 * S_ + 1;
    const int Smax = 2 * Lmax + 1;

    activation_cols_ = minibatch_ * Tmax;

    // Allocate memory for T
    utt_length_ =
        reinterpret_cast<int *>(static_cast<char*>(gpu_workspace_) +
                                gpu_bytes_used);
    gpu_bytes_used += minibatch_  * sizeof(int);

    cuda_status = cudaMemcpyAsync(utt_length_, input_lengths,
                                  minibatch_ * sizeof(int),
                                  cudaMemcpyHostToDevice, stream_);
    if (cuda_status != cudaSuccess)
        return CTC_STATUS_MEMOPS_FAILED;

    label_sizes_ =
        reinterpret_cast<int *>(static_cast<char*>(gpu_workspace_) +
                                gpu_bytes_used);
    gpu_bytes_used += minibatch_ * sizeof(int);
    cuda_status = cudaMemcpyAsync(label_sizes_, label_lengths,
                                  minibatch_ * sizeof(int),
                                  cudaMemcpyHostToDevice, stream_);
    if (cuda_status != cudaSuccess)
        return CTC_STATUS_MEMOPS_FAILED;

    labels_without_blanks_ =
        reinterpret_cast<int *>(static_cast<char*>(gpu_workspace_) +
                                gpu_bytes_used);
    gpu_bytes_used += Lmax * minibatch_ * sizeof(int);
    cuda_status = cudaMemcpyAsync(labels_without_blanks_, flat_labels,
                                  total_label_length * sizeof(int),
                                  cudaMemcpyHostToDevice, stream_);
    if (cuda_status != cudaSuccess)
        return CTC_STATUS_MEMOPS_FAILED;

    labels_with_blanks_ =
        reinterpret_cast<int *>(static_cast<char*>(gpu_workspace_) +
                                gpu_bytes_used);
    gpu_bytes_used += Smax * minibatch_ * sizeof(int);
    cuda_status = cudaMemsetAsync(labels_with_blanks_, ctc_helper::BLANK,
                                  S_ * minibatch_ * sizeof(int), stream_);
    if (cuda_status != cudaSuccess)
        return CTC_STATUS_MEMOPS_FAILED;

    alphas_ =
        reinterpret_cast<ProbT *>(static_cast<char*>(gpu_workspace_) +
                                  gpu_bytes_used);
    gpu_bytes_used += (S_ * T_) * minibatch_ * sizeof(ProbT);


    denoms_ =
        reinterpret_cast<ProbT *>(static_cast<char*>(gpu_workspace_) +
                                  gpu_bytes_used);
    gpu_bytes_used += activation_cols_ * sizeof(ProbT);

    probs_ =
        reinterpret_cast<ProbT *>(static_cast<char*>(gpu_workspace_) +
                                  gpu_bytes_used);
    gpu_bytes_used += out_dim_ * activation_cols_ * sizeof(ProbT);

    return CTC_STATUS_SUCCESS;
}

template<typename ProbT>
template<int NT, int VT>
ctcStatus_t GpuCTC<ProbT>::launch_alpha_beta_kernels(const ProbT* const probs,
                                                     ProbT* grads,
                                                     bool compute_alpha,
                                                     bool compute_beta ) {

    // One thread block per utterance
    const int grid_size = minibatch_;

    // The data is laid out so that the next timestep is minibatch entries
    // away
    const int stride = minibatch_;

    if (compute_alpha)
        compute_alpha_kernel<ProbT, NT, VT><<<grid_size, NT, 0, stream_>>>
            (probs, label_sizes_, utt_length_,
             repeats_, labels_without_blanks_, label_offsets_,
             labels_with_blanks_, alphas_, nll_forward_,
             stride, out_dim_, S_, T_);


    if (compute_beta) {
        compute_betas_and_grad_kernel<ProbT, NT, VT><<<grid_size, NT, 0, stream_>>>
            (probs, label_sizes_, utt_length_, repeats_,
             labels_with_blanks_, alphas_, nll_forward_, nll_backward_,
             grads, stride, out_dim_, S_, T_);

        cudaStreamSynchronize(stream_);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return CTC_STATUS_EXECUTION_FAILED;

    return CTC_STATUS_SUCCESS;
}

template<typename ProbT>
ctcStatus_t
GpuCTC<ProbT>::create_metadata_and_choose_config(const int* const flat_labels,
                                                 const int* const label_lengths,
                                                 const int* const input_lengths,
                                                 size_t& best_config) {

    // Setup the metadata for GPU
    ctcStatus_t status = setup_gpu_metadata(flat_labels, label_lengths, input_lengths);
    if (status != CTC_STATUS_SUCCESS)
        return status;

    constexpr int num_configs = 12;

    int config_NT[num_configs] =
        {32, 64, 128, 64, 128, 32, 64, 128, 64, 128, 128, 128};
    int config_VT[num_configs] =
        { 1,  1,   1,  3,   2,  9,  6,   4,  9,   6,   9,  10};

    best_config = 0;

    for (int i = 0; i < num_configs; ++i) {
        if ((config_NT[i]* config_VT[i]) >= S_)
            break;
        else
            best_config++;
    }

    if (best_config >= num_configs)
        return CTC_STATUS_UNKNOWN_ERROR;

    return CTC_STATUS_SUCCESS;
}

template<typename ProbT>
ctcStatus_t
GpuCTC<ProbT>::launch_gpu_kernels(const ProbT* const probs,
                                  ProbT* grads,
                                  size_t config,
                                  bool l_a,
                                  bool l_b) {

    switch(config) {
        case 0:   {return launch_alpha_beta_kernels<32,   1>(probs, grads, l_a, l_b);}
        case 1:   {return launch_alpha_beta_kernels<64,   1>(probs, grads, l_a, l_b);}
        case 2:   {return launch_alpha_beta_kernels<128,  1>(probs, grads, l_a, l_b);}
        case 3:   {return launch_alpha_beta_kernels<64,   3>(probs, grads, l_a, l_b);}
        case 4:   {return launch_alpha_beta_kernels<128,  2>(probs, grads, l_a, l_b);}
        case 5:   {return launch_alpha_beta_kernels<32,   9>(probs, grads, l_a, l_b);}
        case 6:   {return launch_alpha_beta_kernels<64,   6>(probs, grads, l_a, l_b);}
        case 7:   {return launch_alpha_beta_kernels<128,  4>(probs, grads, l_a, l_b);}
        case 8:   {return launch_alpha_beta_kernels<64,   9>(probs, grads, l_a, l_b);}
        case 9:   {return launch_alpha_beta_kernels<128,  6>(probs, grads, l_a, l_b);}
        case 10:  {return launch_alpha_beta_kernels<128,  9>(probs, grads, l_a, l_b);}
        case 11:  {return launch_alpha_beta_kernels<128, 10>(probs, grads, l_a, l_b);}
    }

    return CTC_STATUS_EXECUTION_FAILED;
}

template<typename ProbT>
ctcStatus_t
GpuCTC<ProbT>::compute_probs(const ProbT* const activations) {

    cudaError_t cuda_status;
    cuda_status =
        cudaMemcpyAsync(probs_, activations,
                        activation_cols_ * out_dim_ *sizeof(ProbT),
                        cudaMemcpyDeviceToDevice, stream_);
    if (cuda_status != cudaSuccess)
        return CTC_STATUS_MEMOPS_FAILED;

    // Numerically stable SM
    ctcStatus_t ctc_status =
        reduce_max(probs_, denoms_, out_dim_,
                   activation_cols_, 1, stream_);
    if (ctc_status != CTC_STATUS_SUCCESS)
        return ctc_status;

    // Kernel launch to subtract maximum
    const int NT = 128;
    const int VT = 1;
    const int NV = NT * VT;
    const int num_elements = out_dim_ * activation_cols_;
    const int grid_size = ctc_helper::div_up(num_elements, NV);

    prepare_stable_SM_kernel<ProbT, VT> <<< grid_size, NT, 0, stream_>>>
       (ctc_helper::identity<ProbT>(), probs_,
        denoms_, out_dim_, num_elements);

    // Reduce along columns to calculate denominator
    ctc_status =
        reduce_exp(probs_, denoms_, out_dim_,
                   activation_cols_, 1, stream_);
    if (ctc_status != CTC_STATUS_SUCCESS)
        return ctc_status;

    // Kernel launch to calculate probabilities
    compute_probs_kernel<ProbT, VT><<<grid_size, NT, 0, stream_>>>
        (ctc_helper::exponential<ProbT>(), probs_,
         denoms_, out_dim_, num_elements);

    return CTC_STATUS_SUCCESS;
}

template<typename ProbT>
ctcStatus_t
GpuCTC<ProbT>::compute_cost_and_score(const ProbT* const activations,
                                      ProbT* grads,
                                      ProbT* costs,
                                      const int* const flat_labels,
                                      const int* const label_lengths,
                                      const int* const input_lengths,
                                      bool compute_alpha,
                                      bool compute_betas_and_grad) {

    size_t best_config;
    ctcStatus_t status = create_metadata_and_choose_config(flat_labels,
                                                           label_lengths,
                                                           input_lengths,
                                                           best_config);
    if (status != CTC_STATUS_SUCCESS)
        return status;

    status = compute_probs(activations);
    if (status != CTC_STATUS_SUCCESS)
        return status;

    launch_gpu_kernels(probs_, grads, best_config,
                       compute_alpha, compute_betas_and_grad);

    cudaError_t cuda_status_mem, cuda_status_sync;
    cuda_status_mem = cudaMemcpyAsync(costs, nll_forward_,
                                      sizeof(ProbT) * minibatch_,
                                      cudaMemcpyDeviceToHost, stream_);
    cuda_status_sync = cudaStreamSynchronize(stream_);
    if (cuda_status_mem != cudaSuccess || cuda_status_sync != cudaSuccess)
        return CTC_STATUS_MEMOPS_FAILED;

    return CTC_STATUS_SUCCESS;
}

template<typename ProbT>
ctcStatus_t
GpuCTC<ProbT>::cost_and_grad(const ProbT* const activations,
                             ProbT* grads,
                             ProbT* costs,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths) {
    if (activations == nullptr ||
        grads == nullptr ||
        costs == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr
        )
        return CTC_STATUS_INVALID_VALUE;

    return compute_cost_and_score(activations, grads, costs, flat_labels,
                                  label_lengths, input_lengths, true, true);
}

template<typename ProbT>
ctcStatus_t
GpuCTC<ProbT>::score_forward(const ProbT* const activations,
                             ProbT* costs,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths) {
    if (activations == nullptr ||
        costs == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr
        )
        return CTC_STATUS_INVALID_VALUE;

    return compute_cost_and_score(activations, nullptr, costs, flat_labels,
                                  label_lengths, input_lengths, true, false);
}

