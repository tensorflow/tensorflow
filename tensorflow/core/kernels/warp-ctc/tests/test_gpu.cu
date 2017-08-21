#include <cmath>
#include <random>
#include <tuple>
#include <vector>
#include <iostream>

#include "tensorflow/core/kernels/warp-ctc/include/ctc.h"

#include "test.h"
#include "test_gpu.h"

bool small_test() {
    const int alphabet_size = 5;
    const int T = 2;

    std::vector<float> activations = {0.1, 0.6, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.6, 0.1, 0.1};

    // Calculate the score analytically
    float expected_score;
    {
        std::vector<float> probs(activations.size());
        softmax(activations.data(), alphabet_size, T, probs.data());

        // Score calculation is specific to the given activations above
        expected_score = probs[1] * probs[7];
    }

    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
                   "cudaStreamCreate");

    float *activations_gpu;
    throw_on_error(cudaMalloc(&activations_gpu, 
                   activations.size() * sizeof(float)),
                   "cudaMalloc");
    throw_on_error(cudaMemcpyAsync(activations_gpu, activations.data(),
                                   activations.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");

    std::vector<int> labels = {1, 2};
    std::vector<int> label_lengths = {2};

    std::vector<int> lengths;
    lengths.push_back(T);

    float score;

    ctcComputeInfo info;
    info.loc = CTC_GPU;
    info.stream = stream;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), lengths.data(),
                                      alphabet_size, lengths.size(), info,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in small_test");

    char *ctc_gpu_workspace;
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");

    throw_on_error(compute_ctc_loss(activations_gpu, NULL,
                                    labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    &score,
                                    ctc_gpu_workspace,
                                    info),
                   "Error: compute_ctc_loss in small_test");

    score = std::exp(-score);
    const float eps = 1e-6;

    const float lb = expected_score - eps;
    const float ub = expected_score + eps;

    throw_on_error(cudaFree(activations_gpu),
                   "cudaFree");
    throw_on_error(cudaFree(ctc_gpu_workspace),
                   "cudaFree");
    throw_on_error(cudaStreamDestroy(stream),
                   "cudaStreamDestroy");

    return (score > lb && score < ub);
}

bool inf_test() {
    const int alphabet_size = 15;
    const int T = 50;
    const int L = 10;
    const int minibatch = 1;

    std::vector<int> labels = genLabels(alphabet_size, L);
    labels[0] = 2;
    std::vector<int> label_lengths = {L};

    std::vector<float> acts = genActs(alphabet_size * T * minibatch);

    for (int i = 0; i < T; ++i)
        acts[alphabet_size * i + 2] = -1e30;

    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
                   "cudaStreamCreate");

    float *acts_gpu;
    throw_on_error(cudaMalloc(&acts_gpu, acts.size() * sizeof(float)),
                   "cudaMalloc");
    throw_on_error(cudaMemcpyAsync(acts_gpu, acts.data(), 
                                   acts.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");

    std::vector<int> lengths;
    lengths.push_back(T);

    float *grads_gpu;
    throw_on_error(cudaMalloc(&grads_gpu, (alphabet_size * T) * sizeof(float)),
                   "cudaMalloc");

    float cost;

    ctcComputeInfo info;
    info.loc = CTC_GPU;
    info.stream = stream;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), lengths.data(),
                                      alphabet_size, lengths.size(), info,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in inf_test");

    char *ctc_gpu_workspace;
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");

    throw_on_error(compute_ctc_loss(acts_gpu, grads_gpu,
                                    labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    &cost,
                                    ctc_gpu_workspace,
                                    info),
                   "Error: compute_ctc_loss in inf_test");

    bool status = std::isinf(cost);

    std::vector<float> grads(alphabet_size * T);
    throw_on_error(cudaMemcpyAsync(grads.data(), grads_gpu, 
                                   grads.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync");
    throw_on_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    for (int i = 0; i < alphabet_size * T; ++i)
        status &= !std::isnan(grads[i]);

    throw_on_error(cudaFree(acts_gpu), "cudaFree");
    throw_on_error(cudaFree(grads_gpu), "cudaFree");
    throw_on_error(cudaFree(ctc_gpu_workspace), "cudaFree");
    throw_on_error(cudaStreamDestroy(stream), "cudaStreamDestroy");

    return status;
}

float grad_check(int T, int alphabet_size,
                  std::vector<float>& acts,
                  const std::vector<std::vector<int>>& labels,
                  const std::vector<int>& lengths) {

    float epsilon = 1e-2;

    const int minibatch = labels.size();

    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream),
                   "cudaStreamCreate");

    float *acts_gpu;
    throw_on_error(cudaMalloc(&acts_gpu, acts.size() * sizeof(float)),
                   "cudaMalloc");
    throw_on_error(cudaMemcpyAsync(acts_gpu, acts.data(),
                                   acts.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");

    std::vector<int> flat_labels;
    std::vector<int> label_lengths;
    for (const auto& l : labels) {
        flat_labels.insert(flat_labels.end(), l.begin(), l.end());
        label_lengths.push_back(l.size());
    }

    std::vector<float> costs(minibatch);

    float *grads_gpu;
    throw_on_error(cudaMalloc(&grads_gpu, acts.size() * sizeof(float)),
                   "cudaMalloc");

    ctcComputeInfo info;
    info.loc = CTC_GPU;
    info.stream = stream;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(),
                                      lengths.data(),
                                      alphabet_size,
                                      lengths.size(),
                                      info,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in grad_check");

    char *ctc_gpu_workspace;
    throw_on_error(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");

    throw_on_error(compute_ctc_loss(acts_gpu, grads_gpu,
                                    flat_labels.data(), 
                                    label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    minibatch,
                                    costs.data(),
                                    ctc_gpu_workspace,
                                    info),
                   "Error: compute_ctc_loss (0) in grad_check");

    std::vector<float> grads(acts.size());
    throw_on_error(cudaMemcpyAsync(grads.data(), 
                                   grads_gpu, grads.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync");
    throw_on_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    std::vector<float> num_grad(grads.size());

    //perform 2nd order central differencing
    for (int i = 0; i < T * alphabet_size * minibatch; ++i) {
        acts[i] += epsilon;

        throw_on_error(cudaMemcpyAsync(acts_gpu, acts.data(), 
                                       acts.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync");

        std::vector<float> costsP1(minibatch);
        std::vector<float> costsP2(minibatch);

        throw_on_error(compute_ctc_loss(acts_gpu, NULL,
                                        flat_labels.data(), 
                                        label_lengths.data(),
                                        lengths.data(),
                                        alphabet_size,
                                        minibatch,
                                        costsP1.data(),
                                        ctc_gpu_workspace,
                                        info),
                       "Error: compute_ctc_loss (1) in grad_check");

        acts[i] -= 2 * epsilon;
        throw_on_error(cudaMemcpyAsync(acts_gpu, acts.data(),
                                       acts.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync");

        throw_on_error(compute_ctc_loss(acts_gpu, NULL,
                                        flat_labels.data(),
                                        label_lengths.data(),
                                        lengths.data(),
                                        alphabet_size,
                                        minibatch,
                                        costsP2.data(),
                                        ctc_gpu_workspace,
                                        info),
                       "Error: compute_ctc_loss (2) in grad_check");

        float costP1 = std::accumulate(costsP1.begin(), costsP1.end(), 0.);
        float costP2 = std::accumulate(costsP2.begin(), costsP2.end(), 0.);

        acts[i] += epsilon;

        num_grad[i] = (costP1 - costP2) / (2 * epsilon);
    }

    float diff = rel_diff(grads, num_grad);

    throw_on_error(cudaFree(acts_gpu),
                   "cudaFree");
    throw_on_error(cudaFree(grads_gpu),
                   "cudaFree");
    throw_on_error(cudaFree(ctc_gpu_workspace),
                   "cudaFree");
    throw_on_error(cudaStreamDestroy(stream),
                   "cudaStreamDestroy");

    return diff;
}

bool run_tests() {
    std::vector<std::tuple<int, int, int, int, float>> problem_sizes =
        { std::make_tuple(28, 50, 15, 1, 1e-5) };

    bool status = true;
    for (auto problem : problem_sizes) {
        int alphabet_size, T, L, minibatch;
        float tol;
        std::tie(alphabet_size, T, L, minibatch, tol) = problem;

        std::vector<float> acts = genActs(alphabet_size * T * minibatch);

        std::vector<std::vector<int>> labels;
        std::vector<int> sizes;
        for (int mb = 0; mb < minibatch; ++mb) {
            int actual_length = L;
            labels.push_back(genLabels(alphabet_size, actual_length));
            sizes.push_back(T);
        }

        float diff = grad_check(T, alphabet_size, acts, labels, sizes);
        status &= (diff < tol);
    }

    return status;
}

int main(void) {
    std::cout << "Running GPU tests" << std::endl;
    throw_on_error(cudaSetDevice(0), "cudaSetDevice");

    bool status = true;
    status &= small_test();
    status &= inf_test();
    status &= run_tests();

    if (status)
        std::cout << "Tests pass" << std::endl;
    else
        std::cout << "Some or all tests fail" << std::endl;
}
