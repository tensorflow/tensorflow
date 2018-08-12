#include <vector>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>

#include "./hungarian/hungarian.hpp"
#include "./stitch_rects.hpp"

using std::vector;

void filter_rects(const vector<vector<vector<Rect> > >& all_rects,
                  vector<Rect>* stitched_rects,
                  float threshold,
                  float max_threshold,
                  float tau,
                  float conf_alpha) {
  const vector<Rect>& accepted_rects = *stitched_rects;
  for (int i = 0; i < (int)all_rects.size(); ++i) {
    for (int j = 0; j < (int)all_rects[0].size(); ++j) {
      vector<Rect> current_rects;
      for (int k = 0; k < (int)all_rects[i][j].size(); ++k) {
        if (all_rects[i][j][k].confidence_ * conf_alpha > threshold) {
          Rect r = Rect(all_rects[i][j][k]);
          r.confidence_ *= conf_alpha;
          r.true_confidence_ *= conf_alpha;
          current_rects.push_back(r);
        }
      }
            
      vector<Rect> relevant_rects;
      for (int k = 0; k < (int)accepted_rects.size(); ++k) {
          for (int l = 0; l < (int)current_rects.size(); ++l) {
              if (accepted_rects[k].overlaps(current_rects[l], tau)) {
                relevant_rects.push_back(Rect(accepted_rects[k]));
                break;
              }
          }
      }
      
      if (relevant_rects.size() == 0 || current_rects.size() == 0) {
          for (int k = 0; k < (int)current_rects.size(); ++k) {
            stitched_rects->push_back(Rect(current_rects[k]));
          }
          continue;
      }
      
      int num_pred = MAX(current_rects.size(), relevant_rects.size());

      int int_cost[num_pred * num_pred];
      for (int k = 0; k < num_pred * num_pred; ++k) { int_cost[k] = 0; }
      for (int k = 0; k < (int)current_rects.size(); ++k) {
        for (int l = 0; l < (int)relevant_rects.size(); ++l) {
          int idx = k * num_pred + l;
          int cost = 10000;
          if (current_rects[k].overlaps(relevant_rects[l], tau)) {
            cost -= 1000;
          }
          cost += (int)(current_rects[k].distance(relevant_rects[l]) / 10.);
          int_cost[idx] = cost;
        }
      }
      
      std::vector<int> assignment;

      hungarian_problem_t p;
      int** m = array_to_matrix(int_cost, num_pred, num_pred);
      hungarian_init(&p, m, num_pred, num_pred, HUNGARIAN_MODE_MINIMIZE_COST);
      hungarian_solve(&p);
      for (int i = 0; i < num_pred; ++i) {
        for (int j = 0; j < num_pred; ++j) {
          if (p.assignment[i][j] == HUNGARIAN_ASSIGNED) {
            assignment.push_back(j);
          }
        }
      }
      assert((int)assignment.size() == num_pred);
      hungarian_free(&p);
      
      for (int i = 0; i < num_pred; ++i) {
        free(m[i]);
      }
      free(m);

      vector<int> bad;
      for (int k = 0; k < (int)assignment.size(); ++k) {
        if (k < (int)current_rects.size() && assignment[k] < (int)relevant_rects.size()) {
          Rect& c = current_rects[k];
          Rect& a = relevant_rects[assignment[k]];
          if (c.confidence_ > max_threshold) {
            bad.push_back(k);
            continue;
          }
          if (c.overlaps(a, tau)) {
            if (c.confidence_ > a.confidence_ && c.iou(a) > 0.7) {
              c.true_confidence_ = a.confidence_;
              stitched_rects->erase(std::find(stitched_rects->begin(), stitched_rects->end(), a));
            } else {
              bad.push_back(k);
            }
          }
        }
      }

      for (int k = 0; k < (int)current_rects.size(); ++k) {
        bool bad_contains_k = false;
        for (int l = 0; l < (int)bad.size(); ++l) {
          if (k == bad[l]) {
            bad_contains_k = true;
            break;
          }
        }
        if (!bad_contains_k) {
          stitched_rects->push_back(Rect(current_rects[k]));
        }
      }
    }
  }
}
