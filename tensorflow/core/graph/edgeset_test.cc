#include "tensorflow/core/graph/edgeset.h"

#include <gtest/gtest.h>
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
class EdgeSetTest : public ::testing::Test {
 public:
  EdgeSetTest() : edges_(nullptr), eset_(nullptr) {}

  ~EdgeSetTest() override {
    delete eset_;
    delete[] edges_;
  }

  void MakeEdgeSet(int n) {
    delete eset_;
    delete[] edges_;
    edges_ = new Edge[n];
    eset_ = new EdgeSet;
    model_.clear();
    for (int i = 0; i < n; i++) {
      eset_->insert(&edges_[i]);
      model_.insert(&edges_[i]);
    }
  }

  void CheckSame() {
    EXPECT_EQ(model_.size(), eset_->size());
    EXPECT_EQ(model_.empty(), eset_->empty());
    std::vector<const Edge*> modelv(model_.begin(), model_.end());
    std::vector<const Edge*> esetv(eset_->begin(), eset_->end());
    std::sort(modelv.begin(), modelv.end());
    std::sort(esetv.begin(), esetv.end());
    EXPECT_EQ(modelv.size(), esetv.size());
    for (size_t i = 0; i < modelv.size(); i++) {
      EXPECT_EQ(modelv[i], esetv[i]) << i;
    }
  }

  Edge nonexistent_;
  Edge* edges_;
  EdgeSet* eset_;
  std::set<const Edge*> model_;
};

namespace {

TEST_F(EdgeSetTest, Ops) {
  for (int n : {0, 1, 2, 3, 4, 10}) {
    MakeEdgeSet(n);
    CheckSame();
    EXPECT_EQ((n == 0), eset_->empty());
    EXPECT_EQ(n, eset_->size());

    eset_->clear();
    model_.clear();
    CheckSame();

    eset_->insert(&edges_[0]);
    model_.insert(&edges_[0]);
    CheckSame();
  }
}

// Try insert/erase of existing elements at different positions.
TEST_F(EdgeSetTest, Exists) {
  for (int n : {0, 1, 2, 3, 4, 10}) {
    MakeEdgeSet(n);
    for (int pos = 0; pos < n; pos++) {
      MakeEdgeSet(n);
      auto p = eset_->insert(&edges_[pos]);
      EXPECT_FALSE(p.second);
      EXPECT_EQ(&edges_[pos], *p.first);

      EXPECT_EQ(1, eset_->erase(&edges_[pos]));
      model_.erase(&edges_[pos]);
      CheckSame();
    }
  }
}

// Try insert/erase of non-existent element.
TEST_F(EdgeSetTest, DoesNotExist) {
  for (int n : {0, 1, 2, 3, 4, 10}) {
    MakeEdgeSet(n);
    EXPECT_EQ(0, eset_->erase(&nonexistent_));
    auto p = eset_->insert(&nonexistent_);
    EXPECT_TRUE(p.second);
    EXPECT_EQ(&nonexistent_, *p.first);
  }
}

}  // namespace
}  // namespace tensorflow
