#pragma once

#include <tuple>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

#if !defined(CTC_DISABLE_OMP) && !defined(APPLE)
#include <omp.h>
#endif

#include "ctc_helper.h"


template<typename ProbT>
class CpuCTC {
public:
    // Noncopyable
    CpuCTC(int alphabet_size, int minibatch, void* workspace, int num_threads) :
            alphabet_size_(alphabet_size), minibatch_(minibatch),
            num_threads_(num_threads), workspace_(workspace) {
#if defined(CTC_DISABLE_OMP) || defined(APPLE)
#else
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        } else {
            num_threads_ = omp_get_max_threads();
        }
#endif
    };

    CpuCTC(const CpuCTC&) = delete;
    CpuCTC& operator=(const CpuCTC&) = delete;

    ctcStatus_t cost_and_grad(const ProbT* const activations,
                              ProbT *grads,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);


    ctcStatus_t score_forward(const ProbT* const activations,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);

private:

    class CpuCTC_metadata {

    private:
        int setup_labels(const int* const labels, int L, int S);

    public:
        CpuCTC_metadata(int L, int S, int T, int mb, int alphabet_size,
                        void* workspace, size_t bytes_used,
                        const int* const labels);

        ProbT* alphas;
        ProbT* betas;
        int* labels_w_blanks;
        int* e_inc;
        int* s_inc;
        ProbT* output;
        int repeats;
    };

    int alphabet_size_; // Number of characters plus blank
    int minibatch_;
    int num_threads_;
    void* workspace_;

    void softmax(const ProbT* const activations, ProbT* probs,
                 const int* const input_lengths);

    std::tuple<ProbT, bool>
            cost_and_grad_kernel(ProbT *grad, const ProbT* const probs,
                                 const int* const labels, int T, int L,
                                 int mb, size_t bytes_used);

    ProbT compute_alphas(const ProbT* probs, int repeats, int S, int T,
                         const int* const e_inc,
                         const int* const s_inc,
                         const int* const labels,
                         ProbT* alphas);

    ProbT compute_betas_and_grad(ProbT* grad, const ProbT* const probs,
                                 ProbT log_partition, int repeats,
                                 int S, int T, const int* const e_inc,
                                 const int* const s_inc,
                                 const int* const labels,
                                 ProbT* alphas,
                                 ProbT* betas,
                                 ProbT* output);
};

template<typename ProbT>
CpuCTC<ProbT>::CpuCTC_metadata::CpuCTC_metadata(int L, int S, int T, int mb,
                                                int alphabet_size,
                                                void* workspace, size_t bytes_used,
                                                const int* const labels) {

    alphas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * S * T;
    std::fill(alphas, alphas + S * T, ctc_helper::neg_inf<ProbT>());
    betas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * S;
    std::fill(betas, betas + S, ctc_helper::neg_inf<ProbT>());
    labels_w_blanks = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(int) * S;
    e_inc = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(int) * S;
    s_inc = reinterpret_cast<int *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(int) * S;
    output = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * alphabet_size;

    repeats = setup_labels(labels, L, S);
}

template<typename ProbT>
int CpuCTC<ProbT>::CpuCTC_metadata::setup_labels(const int* const labels,
                                                 int L, int S) {
    int e_counter = 0;
    int s_counter = 0;

    s_inc[s_counter++] = 1;

    int repeats = 0;

    for (int i = 1; i < L; ++i) {
        if (labels[i-1] == labels[i]) {
            s_inc[s_counter++] = 1;
            s_inc[s_counter++] = 1;
            e_inc[e_counter++] = 1;
            e_inc[e_counter++] = 1;
            ++repeats;
        }
        else {
            s_inc[s_counter++] = 2;
            e_inc[e_counter++] = 2;
        }
    }
    e_inc[e_counter++] = 1;

    for (int i = 0; i < L; ++i) {
        labels_w_blanks[2 * i] = ctc_helper::BLANK;
        labels_w_blanks[2 * i + 1] = labels[i];
    }
    labels_w_blanks[S - 1] = ctc_helper::BLANK;

    return repeats;
}

template<typename ProbT>
void
CpuCTC<ProbT>::softmax(const ProbT* const activations, ProbT* probs,
                       const int* const input_lengths) {
#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        for(int c = 0; c < input_lengths[mb]; ++c) {
            int col_offset = (mb + minibatch_ * c) * alphabet_size_;
            ProbT max_activation = -std::numeric_limits<ProbT>::infinity();
            for(int r = 0; r < alphabet_size_; ++r)
                max_activation = std::max(max_activation, activations[r + col_offset]);

            ProbT denom = ProbT(0.);
            for(int r = 0; r < alphabet_size_; ++r)
                denom += std::exp(activations[r + col_offset] - max_activation);

            for(int r = 0; r < alphabet_size_; ++r) {
                probs[r + col_offset] = std::exp(activations[r + col_offset] - max_activation) / denom;
            }
        }
    }
}

template<typename ProbT>
std::tuple<ProbT, bool>
CpuCTC<ProbT>::cost_and_grad_kernel(ProbT *grad, const ProbT* const probs,
                                    const int* const labels,
                                    int T, int L, int mb, size_t bytes_used) {

    const int S = 2*L + 1; // Number of labels with blanks

    CpuCTC_metadata ctcm(L, S, T, mb, alphabet_size_, workspace_, bytes_used, labels);

    bool over_threshold = false;

    if (L + ctcm.repeats > T) {
        return std::make_tuple(ProbT(0), over_threshold); // TODO, not right to return 0
    }

    ProbT llForward = compute_alphas(probs, ctcm.repeats, S, T, ctcm.e_inc,
                                     ctcm.s_inc, ctcm.labels_w_blanks,
                                     ctcm.alphas);

    ProbT llBackward = compute_betas_and_grad(grad, probs, llForward, ctcm.repeats,
                                              S, T, ctcm.e_inc, ctcm.s_inc,
                                              ctcm.labels_w_blanks,
                                              ctcm.alphas,
                                              ctcm.betas,
                                              ctcm.output);

    ProbT diff = std::abs(llForward - llBackward);
    if (diff > ctc_helper::threshold) {
        over_threshold = true;
    }

    return std::make_tuple(-llForward, over_threshold);
}

// Computes forward probabilities
template<typename ProbT>
ProbT CpuCTC<ProbT>::compute_alphas(const ProbT* probs, int repeats, int S, int T,
                                    const int* const e_inc,
                                    const int* const s_inc,
                                    const int* const labels,
                                    ProbT* alphas) {

    int start =  (((S /2) + repeats - T) < 0) ? 0 : 1,
            end = S > 1 ? 2 : 1;

    for (int i = start; i < end; ++i) {
        alphas[i] = std::log(probs[labels[i]]);
    }

    for(int t = 1; t < T; ++t) {
        int remain = (S / 2) + repeats - (T - t);
        if(remain >= 0)
            start += s_inc[remain];
        if(t <= (S / 2) + repeats)
            end += e_inc[t - 1];
        int startloop = start;
        int idx1 = t * S, idx2 = (t - 1) * S, idx3 = t * (alphabet_size_ * minibatch_);

        if (start == 0) {
            alphas[idx1] = alphas[idx2] + std::log(probs[ctc_helper::BLANK + idx3]);
            startloop += 1;
        }

        for(int i = startloop; i < end; ++i) {
            ProbT prev_sum = ctc_helper::log_plus<ProbT>()(alphas[i + idx2], alphas[(i-1) + idx2]);

            // Skip two if not on blank and not on repeat.
            if (labels[i] != ctc_helper::BLANK && i != 1 && labels[i] != labels[i-2])
                prev_sum = ctc_helper::log_plus<ProbT>()(prev_sum, alphas[(i-2) + idx2]);

            alphas[i + idx1] = prev_sum + std::log(probs[labels[i] + idx3]);
        }
    }

    ProbT loglike = ctc_helper::neg_inf<ProbT>();
    for(int i = start; i < end; ++i) {
        loglike = ctc_helper::log_plus<ProbT>()(loglike, alphas[i + (T - 1) * S]);
    }

    return loglike;
}

// Starting from T, we sweep backward over the alpha array computing one column
// of betas as we go.  At each position we can update product alpha * beta and then
// sum into the gradient associated with each label.
// NOTE computes gradient w.r.t UNNORMALIZED final layer activations.
// Assumed passed in grads are already zeroed!
template<typename ProbT>
ProbT CpuCTC<ProbT>::compute_betas_and_grad(ProbT* grad, const ProbT* const probs,
                                            ProbT log_partition, int repeats,
                                            int S, int T, const int* const e_inc,
                                            const int* const s_inc,
                                            const int* const labels,
                                            ProbT* alphas,
                                            ProbT* betas,
                                            ProbT* output) {
    int start = S > 1 ? (S - 2) : 0,
            end = (T > (S / 2) + repeats) ? S : S-1;

    std::fill(output, output + alphabet_size_, ctc_helper::neg_inf<ProbT>());

    //set the starting values in the beta column at the very right edge
    for (int i = start; i < end; ++i) {
        betas[i] = std::log(probs[labels[i] + (T - 1) * (alphabet_size_ * minibatch_)]);

        //compute alpha * beta in log space at this position in (S, T) space
        alphas[i + (T - 1) * S] += betas[i];

        //update the gradient associated with this label
        //essentially performing a reduce-by-key in a sequential manner
        output[labels[i]] =
                ctc_helper::log_plus<ProbT>()(alphas[i + (T - 1) * S], output[labels[i]]);
    }

    //update the gradient wrt to each unique label
    for (int i = 0; i < alphabet_size_; ++i) {
        int idx3 = (T - 1) * alphabet_size_ * minibatch_ + i;

        if (output[i] == 0.0 || output[i] == ctc_helper::neg_inf<ProbT>() ||
            probs[idx3] == 0.0) {
            grad[idx3] = probs[idx3];
        } else {
            grad[idx3] = probs[idx3] - std::exp(output[i] -
                                                std::log(probs[idx3]) - log_partition);
        }
    }

    //loop from the second to last column all the way to the left
    for(int t = T - 2; t >= 0; --t) {
        int remain = (S / 2) + repeats - (T - t);
        if(remain >= -1)
            start -= s_inc[remain + 1];
        if(t < (S / 2) + repeats)
            end -= e_inc[t];

        int endloop = end == S ? end - 1 : end;
        int idx1 = t * S, idx3 = t * (alphabet_size_ * minibatch_);

        std::fill(output, output + alphabet_size_, ctc_helper::neg_inf<ProbT>());

        for(int i = start; i < endloop; ++i) {
            ProbT next_sum = ctc_helper::log_plus<ProbT>()(betas[i], betas[(i+1)]);
            // Skip two if not on blank and not on repeat.
            if (labels[i] != ctc_helper::BLANK && i != (S-2) && labels[i] != labels[i+2]){
                next_sum = ctc_helper::log_plus<ProbT>()(next_sum, betas[(i+2)]);
            }
            betas[i] = next_sum + std::log(probs[labels[i] + idx3]);

            //compute alpha * beta in log space
            alphas[i + idx1] += betas[i];

            //update the gradient associated with this label
            output[labels[i]] =
                    ctc_helper::log_plus<ProbT>()(alphas[i + idx1], output[labels[i]]);
        }

        if (end == S) {
            betas[(S-1)] = betas[(S-1)] + std::log(probs[ctc_helper::BLANK + idx3]);
            alphas[(S-1) + idx1] += betas[(S-1)];

            output[labels[S-1]] =
                    ctc_helper::log_plus<ProbT>()(alphas[S-1 + idx1], output[labels[S-1]]);
        }

        //go over the unique labels and compute the final grad
        // wrt to each one at this time step
        for (int i = 0; i < alphabet_size_; ++i) {

            if (output[i] == 0.0 || output[i] == ctc_helper::neg_inf<ProbT>() ||
                probs[idx3] == 0.0) {
                grad[idx3] = probs[idx3];
            } else {
                grad[idx3] = probs[idx3] - std::exp(output[i] -
                                                    std::log(probs[idx3]) - log_partition);
            }
            ++idx3;
        }
    }

    ProbT loglike = ctc_helper::neg_inf<ProbT>();
    for(int i = start; i < end; ++i) {
        loglike = ctc_helper::log_plus<ProbT>()(loglike, betas[i]);
    }

    return loglike;
}

template<typename ProbT>
ctcStatus_t
CpuCTC<ProbT>::cost_and_grad(const ProbT* const activations,
                             ProbT *grads,
                             ProbT *costs,
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

    ProbT* probs = static_cast<ProbT *>(workspace_);

    int maxT = *std::max_element(input_lengths, input_lengths + minibatch_);

    size_t bytes_used = sizeof(ProbT) * minibatch_ * alphabet_size_ * maxT;

    //per minibatch memory
    size_t per_minibatch_bytes = 0;

    int maxL = *std::max_element(label_lengths, label_lengths + minibatch_);;
    int maxS = 2 * maxL + 1;

    //output
    per_minibatch_bytes += sizeof(float) * alphabet_size_;

    //alphas
    per_minibatch_bytes += sizeof(float) * maxS * maxT;

    //betas
    per_minibatch_bytes += sizeof(float) * maxS;

    //labels w/blanks, e_inc, s_inc
    per_minibatch_bytes += 3 * sizeof(int) * maxS;

    softmax(activations, probs, input_lengths);

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb]; // Length of utterance (time)
        const int L = label_lengths[mb]; // Number of labels in transcription

        bool mb_status;

        std::tie(costs[mb], mb_status) =
                cost_and_grad_kernel(grads + mb * alphabet_size_,
                                     probs + mb * alphabet_size_,
                                     flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0),
                                     T, L, mb,
                                     bytes_used + mb * per_minibatch_bytes);
    }

    return CTC_STATUS_SUCCESS;
}

template<typename ProbT>
ctcStatus_t CpuCTC<ProbT>::score_forward(const ProbT* const activations,
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

    ProbT* probs = static_cast<ProbT *>(workspace_);

    int maxT = *std::max_element(input_lengths, input_lengths + minibatch_);

    size_t bytes_used = sizeof(ProbT) * minibatch_ * alphabet_size_ * maxT;

    //per minibatch memory
    size_t per_minibatch_bytes = 0;

    int maxL = *std::max_element(label_lengths, label_lengths + minibatch_);
    int maxS = 2 * maxL + 1;

    //output
    per_minibatch_bytes += sizeof(float) * alphabet_size_;

    //alphas
    per_minibatch_bytes += sizeof(float) * maxS * maxT;

    //betas
    per_minibatch_bytes += sizeof(float) * maxS;

    //labels w/blanks, e_inc, s_inc
    per_minibatch_bytes += 3 * sizeof(int) * maxS;

    softmax(activations, probs, input_lengths);

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb]; // Length of utterance (time)
        const int L = label_lengths[mb]; // Number of labels in transcription
        const int S = 2*L + 1; // Number of labels with blanks

        CpuCTC_metadata ctcm(L, S, T, mb, alphabet_size_, workspace_,
                             bytes_used + mb * per_minibatch_bytes,
                             flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0));


        if (L + ctcm.repeats > T)
            costs[mb] = ProbT(0);
        else {
            costs[mb] = -compute_alphas(probs + mb * alphabet_size_, ctcm.repeats, S, T,
                                        ctcm.e_inc, ctcm.s_inc, ctcm.labels_w_blanks,
                                        ctcm.alphas);
        }

    }

    return CTC_STATUS_SUCCESS;
}
