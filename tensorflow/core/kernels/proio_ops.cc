#include <float.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

#include "external/proio_archive/cpp-proio/src/event.h"
#include "external/proio_archive/model/eic.pb.h"

namespace model = proio::model::eic;

namespace tensorflow {
namespace {

const double sqrt3 = sqrt(3.);

class ExtractEnergyDepsOp : public OpKernel {
   public:
    explicit ExtractEnergyDepsOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("event_tag", &event_tag_));
        OP_REQUIRES_OK(context, context->GetAttr("n_samples", &n_samples_));
        OP_REQUIRES(context, n_samples_ > 0, errors::InvalidArgument("n_samples should be greater than 0"));
        OP_REQUIRES_OK(context, context->GetAttr("max_n_deps", &max_n_deps_));
        OP_REQUIRES(context, max_n_deps_ > 0, errors::InvalidArgument("max_n_deps should be greater than 0"));
    }

    void Compute(OpKernelContext *context) override {
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat_outer_dims<std::string, 2>();

        int n_events = input.dimension(0);
        int n_merge_events = input.dimension(1);
        int max_event_edeps = 1;

        std::vector<std::pair<model::EnergyDep *, std::vector<std::pair<int, uint64_t>>>> edeps[n_events];
        std::map<model::EnergyDep *, uint64_t> id_map;
        std::unique_ptr<proio::Event> events[n_events][n_merge_events];

        for (int event_index = 0; event_index < n_events; event_index++) {
            std::vector<std::pair<model::EnergyDep *, std::vector<std::pair<int, uint64_t>>>> &event_edeps =
                edeps[event_index];

            for (int merge_event_index = 0; merge_event_index < n_merge_events; merge_event_index++) {
                std::unique_ptr<proio::Event> &event = events[event_index][merge_event_index];

                event.reset(new proio::Event(input(event_index, merge_event_index)));

                auto edep_ids = event->TaggedEntries(event_tag_);

                for (auto id : edep_ids) {
                    auto edep = dynamic_cast<model::EnergyDep *>(event->GetEntry(id));
                    if (!edep) continue;

                    std::vector<std::pair<int, uint64_t>> parts;
                    for (auto source_id : edep->source()) {
                        auto part_id = source_id;
                        if (!dynamic_cast<model::Particle *>(event->GetEntry(part_id))) {
                            auto sim_hit = dynamic_cast<model::SimHit *>(event->GetEntry(source_id));
                            if (sim_hit)
                                part_id = sim_hit->particle();
                            else
                                assert(0);
                        }

                        parts.push_back(std::pair<int, uint64_t>(merge_event_index, part_id));
                    }

                    event_edeps.push_back(
                        std::pair<model::EnergyDep *, std::vector<std::pair<int, uint64_t>>>(edep, parts));
                    id_map[edep] = id;
                }
            }

            std::sort(event_edeps.begin(), event_edeps.end(),
                      [](const std::pair<model::EnergyDep *, std::vector<std::pair<int, uint64_t>>> a,
                         const std::pair<model::EnergyDep *, std::vector<std::pair<int, uint64_t>>> b) {
                          double t_sum = 0;
                          double weight_sum = 0;
                          for (auto pos : a.first->pos()) {
                              double weight = pos.weightmod() + 1;
                              t_sum += pos.mean().t() * weight;
                              weight_sum += weight;
                          }
                          double t_a = t_sum / weight_sum;
                          t_sum = 0;
                          weight_sum = 0;
                          for (auto pos : b.first->pos()) {
                              double weight = pos.weightmod() + 1;
                              t_sum += pos.mean().t() * weight;
                              weight_sum += weight;
                          }
                          double t_b = t_sum / weight_sum;
                          return t_a < t_b;
                      });

            if (event_edeps.size() > max_event_edeps) max_event_edeps = event_edeps.size();
        }

        random::PhiloxRandom gen((random::New64()));
        random::UniformDistribution<random::PhiloxRandom, double> flat_double;
        random::NormalDistribution<random::PhiloxRandom, double> normal_double;

        int max_n_deps = max_event_edeps;
        if (max_n_deps > max_n_deps_) max_n_deps = max_n_deps_;
        Tensor *edeps_tensor = NULL;
        Tensor *ids_tensor = NULL;
        Tensor *assoc_tensor = NULL;
        Tensor *assoc_inv_tensor = NULL;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, {n_events, n_samples_, max_n_deps, 14}, &edeps_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, {n_events, max_n_deps}, &ids_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(2, {n_events, max_n_deps, max_n_deps}, &assoc_tensor));
        OP_REQUIRES_OK(context,
                       context->allocate_output(3, {n_events, max_n_deps, max_n_deps}, &assoc_inv_tensor));
        auto edeps_eigen_tensor = edeps_tensor->tensor<float, 4>();
        auto ids_eigen_tensor = ids_tensor->tensor<float, 2>();
        auto assoc_eigen_tensor = assoc_tensor->tensor<float, 3>();
        auto assoc_inv_eigen_tensor = assoc_inv_tensor->tensor<float, 3>();

        auto edeps_flat = edeps_tensor->flat<float>();
        for (int i = 0; i < edeps_flat.size(); i++) edeps_flat(i) = 0;
        auto ids_flat = ids_tensor->flat<float>();
        for (int i = 0; i < ids_flat.size(); i++) ids_flat(i) = 0;
        auto assoc_flat = assoc_tensor->flat<float>();
        auto assoc_inv_flat = assoc_inv_tensor->flat<float>();
        for (int i = 0; i < assoc_flat.size(); i++) {
            assoc_flat(i) = 0;
            assoc_inv_flat(i) = 0;
        }

        for (int event_index = 0; event_index < n_events; event_index++) {
            std::vector<std::pair<model::EnergyDep *, std::vector<std::pair<int, uint64_t>>>> &event_edeps =
                edeps[event_index];

            int event_max_time = max_n_deps;
            if (event_max_time > event_edeps.size()) event_max_time = event_edeps.size();

            for (int t_index = 0; t_index < event_max_time; t_index++) {
                assoc_eigen_tensor(event_index, t_index, t_index) = 0;
                assoc_inv_eigen_tensor(event_index, t_index, t_index) = 0;
                for (int t_index2 = t_index + 1; t_index2 < event_max_time; t_index2++) {
                    bool sharedPart = false;
                    for (auto part : event_edeps[t_index].second) {
                        for (auto part2 : event_edeps[t_index2].second) {
                            if (part.first == part2.first && part.second == part2.second) {
                                sharedPart = true;
                                goto endpartloops;
                            }
                        }
                    }
                endpartloops:
                    if (sharedPart) {
                        assoc_eigen_tensor(event_index, t_index, t_index2) = 1;
                        assoc_eigen_tensor(event_index, t_index2, t_index) = 1;
                        assoc_inv_eigen_tensor(event_index, t_index, t_index2) = 0;
                        assoc_inv_eigen_tensor(event_index, t_index2, t_index) = 0;
                    } else {
                        assoc_eigen_tensor(event_index, t_index, t_index2) = 0;
                        assoc_eigen_tensor(event_index, t_index2, t_index) = 0;
                        assoc_inv_eigen_tensor(event_index, t_index, t_index2) = 1;
                        assoc_inv_eigen_tensor(event_index, t_index2, t_index) = 1;
                    }
                }

                auto edep = event_edeps[t_index].first;
                ids_eigen_tensor(event_index, t_index) = id_map[edep];

                int pos_index = 0;
                if (edep->pos_size() > 0) {
                    double total_weight = 0;
                    for (auto pos : edep->pos()) total_weight += pos.weightmod() + 1;
                    double weight_pick = flat_double(&gen)[0] * total_weight;
                    total_weight = 0;
                    for (int i = 0; i < edep->pos_size(); i++) {
                        total_weight += edep->pos()[i].weightmod() + 1;
                        if (weight_pick < total_weight) {
                            pos_index = i;
                            break;
                        }
                    }
                }
                auto pos = edep->pos()[pos_index];

                double x = pos.mean().x();
                double y = pos.mean().y();
                double z = pos.mean().z();
                double t = pos.mean().t();

                for (int sample = 0; sample < n_samples_; sample++) {
                    for (auto rand_var : pos.noise())
                        if (rand_var.dist() == model::RandVar::NORMAL) {
                            x += normal_double(&gen)[0] * rand_var.sigma().x();
                            y += normal_double(&gen)[0] * rand_var.sigma().y();
                            z += normal_double(&gen)[0] * rand_var.sigma().z();
                            t += normal_double(&gen)[0] * rand_var.sigma().t();
                        } else if (rand_var.dist() == model::RandVar::UNIFORM) {
                            x += (flat_double(&gen)[0] * 2 - 1) * rand_var.sigma().x() * sqrt3;
                            y += (flat_double(&gen)[0] * 2 - 1) * rand_var.sigma().y() * sqrt3;
                            z += (flat_double(&gen)[0] * 2 - 1) * rand_var.sigma().z() * sqrt3;
                            t += (flat_double(&gen)[0] * 2 - 1) * rand_var.sigma().t() * sqrt3;
                        }

                    double e = edep->mean() + normal_double(&gen)[0] * edep->noise();

                    double xx = x * x;
                    double yy = y * y;
                    double zz = z * z;
                    double xy = x * y;
                    double xz = x * z;
                    double yz = y * z;
                    double rho = sqrt(xx + yy);
                    double theta = atan(rho / z);
                    double phi = atan(y / x);

                    edeps_eigen_tensor(event_index, sample, t_index, 0) = x;
                    edeps_eigen_tensor(event_index, sample, t_index, 1) = y;
                    edeps_eigen_tensor(event_index, sample, t_index, 2) = z;
                    edeps_eigen_tensor(event_index, sample, t_index, 3) = xx;
                    edeps_eigen_tensor(event_index, sample, t_index, 4) = yy;
                    edeps_eigen_tensor(event_index, sample, t_index, 5) = zz;
                    edeps_eigen_tensor(event_index, sample, t_index, 6) = xy;
                    edeps_eigen_tensor(event_index, sample, t_index, 7) = xz;
                    edeps_eigen_tensor(event_index, sample, t_index, 8) = yz;
                    edeps_eigen_tensor(event_index, sample, t_index, 9) = rho;
                    edeps_eigen_tensor(event_index, sample, t_index, 10) = theta;
                    edeps_eigen_tensor(event_index, sample, t_index, 11) = phi;
                    edeps_eigen_tensor(event_index, sample, t_index, 12) = e;
                    edeps_eigen_tensor(event_index, sample, t_index, 13) = t;
                }
            }
        }
    }

   private:
    std::string event_tag_;
    int n_samples_;
    int max_n_deps_;
};

REGISTER_KERNEL_BUILDER(Name("ExtractEnergyDeps").Device(DEVICE_CPU), ExtractEnergyDepsOp);

class CompClusterOp : public OpKernel {
   public:
    explicit CompClusterOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        // Grab the input tensors
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat_inner_dims<std::string, 1>();
        int n_events = input.dimension(0);

        const Tensor &cross_entropy_tensor = context->input(1);
        auto cross_entropy = cross_entropy_tensor.flat_inner_dims<float, 3>();
        OP_REQUIRES(
            context, cross_entropy.dimension(0) == n_events,
            errors::InvalidArgument(
                "CompCluster requires that the first indices for all input tensors have the same size"));
        int n_hits = cross_entropy.dimension(1);
        OP_REQUIRES(
            context, cross_entropy.dimension(2) == n_hits,
            errors::InvalidArgument(
                "CompCluster requires that the last two cross-entropy tensor indices have equal size"));

        const Tensor &entropy_tensor = context->input(2);
        auto entropy = entropy_tensor.flat_inner_dims<float, 2>();
        OP_REQUIRES(
            context, entropy.dimension(0) == n_events,
            errors::InvalidArgument(
                "CompCluster requires that the first indices for all input tensors have the same size"));
        OP_REQUIRES(context, entropy.dimension(1) == n_hits,
                    errors::InvalidArgument("CompCluster requires that the last entropy tensor index has the "
                                            "same size as the last two cross-entropy tensor indices"));

        const Tensor &ids_tensor = context->input(3);
        auto ids = ids_tensor.flat_inner_dims<float, 2>();
        OP_REQUIRES(
            context, ids.dimension(0) == n_events,
            errors::InvalidArgument(
                "CompCluster requires that the first indices for all input tensors have the same size"));
        OP_REQUIRES(context, ids.dimension(1) == n_hits,
                    errors::InvalidArgument("CompCluster requires that the last hit id tensor index has the "
                                            "same size as the last two cross-entropy tensor indices"));

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
        auto output = output_tensor->flat_inner_dims<std::string, 1>();

        for (int i = 0; i < n_events; i++) {
            std::unique_ptr<proio::Event> event(new proio::Event(input(i)));

            int current_hit = 0;
            std::vector<std::map<int, int>> clusters;
            for (int j = 1; j < n_hits; j++) {
                if (clusters.size() == 0) clusters.push_back(std::map<int, int>());
                std::map<int, int> *cluster = &clusters.back();
                if (cluster->size() == 0) (*cluster)[current_hit] = 1;

                double minDist = DBL_MAX;
                int nextHit = -1;
                for (int k = 0; k < n_hits; k++) {
                    if (k == current_hit) continue;
                    double dist = DBL_MAX;
                    for (auto tmp_cluster : clusters)
                        if (tmp_cluster.count(k) > 0) goto skipHit;
                    dist = cross_entropy(i, k, current_hit);
                    if (dist < minDist) {
                        minDist = dist;
                        nextHit = k;
                    }
                skipHit:
                    continue;
                }

                current_hit = nextHit;
                if (minDist > entropy(i, current_hit)) {
                    clusters.push_back(std::map<int, int>());
                    cluster = &clusters.back();
                }
                (*cluster)[current_hit] = 1;
            }

            for (auto cluster : clusters) {
                auto track = new model::Track();
                for (auto keyValuePair : cluster) track->add_observation(ids(i, keyValuePair.first));
                event->AddEntry(track, "Reconstructed");
            }

            output(i) = "test";
            event->SerializeToString(&output(i));
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("CompCluster").Device(DEVICE_CPU), CompClusterOp);

}  // namespace
}  // namespace tensorflow
