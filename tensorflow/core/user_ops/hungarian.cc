#include <assert.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using std::vector;

#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#ifdef __cplusplus
extern "C" {
#endif
  
#define HUNGARIAN_NOT_ASSIGNED 0 
#define HUNGARIAN_ASSIGNED 1

#define HUNGARIAN_MODE_MINIMIZE_COST   0
#define HUNGARIAN_MODE_MAXIMIZE_UTIL 1

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

using std::vector;

class Rect {
 public:
  int cx_;
  int cy_;
  int width_;
  int height_;
  float confidence_;
  float true_confidence_;

  explicit Rect(int cx, int cy, int width, int height, float confidence) {
    cx_ = cx;
    cy_ = cy;
    width_ = width;
    height_ = height;
    confidence_ = confidence;
    true_confidence_ = confidence;
  }

  Rect(const Rect& other) {
    cx_ = other.cx_;
    cy_ = other.cy_;
    width_ = other.width_;
    height_ = other.height_;
    confidence_ = other.confidence_;
    true_confidence_ = other.true_confidence_;
  }

  bool overlaps(const Rect& other) const {
    if (fabs(cx_ - other.cx_) > (width_ + other.width_) / 1.5) {
      return false;
    } else if (fabs(cy_ - other.cy_) > (height_ + other.height_) / 2.0) {
      return false;
    } else {
      return iou(other) > 0.25;
    }
  }

  int distance(const Rect& other) const {
    return (fabs(cx_ - other.cx_) + fabs(cy_ - other.cy_) +
            fabs(width_ - other.width_) + fabs(height_ - other.height_));
  }

  float intersection(const Rect& other) const {
    int left = MAX(cx_ - width_ / 2., other.cx_ - other.width_ / 2.);
    int right = MIN(cx_ + width_ / 2., other.cx_ + other.width_ / 2.);
    int width = MAX(right - left, 0);

    int top = MAX(cy_ - height_ / 2., other.cy_ - other.height_ / 2.);
    int bottom = MIN(cy_ + height_ / 2., other.cy_ + other.height_ / 2.);
    int height = MAX(bottom - top, 0);
    return width * height;
  }

  int area() const {
    return height_ * width_;
  }

  float union_area(const Rect& other) const {
    return this->area() + other.area() - this->intersection(other);
  }

  float iou(const Rect& other) const {
    return this->intersection(other) / this->union_area(other);
  }

  bool operator==(const Rect& other) const {
    return (cx_ == other.cx_ && 
      cy_ == other.cy_ &&
      width_ == other.width_ &&
      height_ == other.height_ &&
      confidence_ == other.confidence_);
  }
};


typedef struct {
  int num_rows;
  int num_cols;
  int** cost;
  int** assignment;  
} hungarian_problem_t;

/** This method initialize the hungarian_problem structure and init 
 *  the  cost matrices (missing lines or columns are filled with 0).
 *  It returns the size of the quadratic(!) assignment matrix. **/
int hungarian_init(hungarian_problem_t* p, 
		   int** cost_matrix, 
		   int rows, 
		   int cols, 
		   int mode);
  
/** Free the memory allocated by init. **/
void hungarian_free(hungarian_problem_t* p);

/** This method computes the optimal assignment. **/
void hungarian_solve(hungarian_problem_t* p);

/** Print the computed optimal assignment. **/
void hungarian_print_assignment(hungarian_problem_t* p);

/** Print the cost matrix. **/
void hungarian_print_costmatrix(hungarian_problem_t* p);

/** Print cost matrix and assignment matrix. **/
void hungarian_print_status(hungarian_problem_t* p);

int** array_to_matrix(int* m, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif

/********************************************************************
 ********************************************************************
 **
 ** libhungarian by Cyrill Stachniss, 2004
 **
 **
 ** Solving the Minimum Assignment Problem using the 
 ** Hungarian Method.
 **
 ** ** This file may be freely copied and distributed! **
 **
 ** Parts of the used code was originally provided by the 
 ** "Stanford GraphGase", but I made changes to this code.
 ** As asked by  the copyright node of the "Stanford GraphGase", 
 ** I hereby proclaim that this file are *NOT* part of the
 ** "Stanford GraphGase" distrubition!
 **
 ** This file is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY; without even the implied 
 ** warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 ** PURPOSE.  
 **
 ********************************************************************
 ********************************************************************/


#include <stdio.h>
#include <stdlib.h>

#define INF (0x7FFFFFFF)
#define verbose (0)

#define hungarian_test_alloc(X) do {if ((void *)(X) == NULL) fprintf(stderr, "Out of memory in %s, (%s, line %d).\n", __FUNCTION__, __FILE__, __LINE__); } while (0)

int** array_to_matrix(int* m, int rows, int cols) {
  int i,j;
  int** r;
  r = (int**)calloc(rows,sizeof(int*));
  for(i=0;i<rows;i++)
  {
    r[i] = (int*)calloc(cols,sizeof(int));
    for(j=0;j<cols;j++)
      r[i][j] = m[i*cols+j];
  }
  return r;
}

void hungarian_print_matrix(int** C, int rows, int cols) {
  int i,j;
  fprintf(stderr , "\n");
  for(i=0; i<rows; i++) {
    fprintf(stderr, " [");
    for(j=0; j<cols; j++) {
      fprintf(stderr, "%5d ",C[i][j]);
    }
    fprintf(stderr, "]\n");
  }
  fprintf(stderr, "\n");
}

void hungarian_print_assignment(hungarian_problem_t* p) {
  hungarian_print_matrix(p->assignment, p->num_rows, p->num_cols) ;
}

void hungarian_print_costmatrix(hungarian_problem_t* p) {
  hungarian_print_matrix(p->cost, p->num_rows, p->num_cols) ;
}

void hungarian_print_status(hungarian_problem_t* p) {
  
  fprintf(stderr,"cost:\n");
  hungarian_print_costmatrix(p);

  fprintf(stderr,"assignment:\n");
  hungarian_print_assignment(p);
  
}

int hungarian_imax(int a, int b) {
  return (a<b)?b:a;
}

int hungarian_init(hungarian_problem_t* p, int** cost_matrix, int rows, int cols, int mode) {

  int i,j, org_cols, org_rows;
  int max_cost;
  max_cost = 0;
  
  org_cols = cols;
  org_rows = rows;

  // is the number of cols  not equal to number of rows ? 
  // if yes, expand with 0-cols / 0-cols
  rows = hungarian_imax(cols, rows);
  cols = rows;
  
  p->num_rows = rows;
  p->num_cols = cols;

  p->cost = (int**)calloc(rows,sizeof(int*));
  hungarian_test_alloc(p->cost);
  p->assignment = (int**)calloc(rows,sizeof(int*));
  hungarian_test_alloc(p->assignment);

  for(i=0; i<p->num_rows; i++) {
    p->cost[i] = (int*)calloc(cols,sizeof(int));
    hungarian_test_alloc(p->cost[i]);
    p->assignment[i] = (int*)calloc(cols,sizeof(int));
    hungarian_test_alloc(p->assignment[i]);
    for(j=0; j<p->num_cols; j++) {
      p->cost[i][j] =  (i < org_rows && j < org_cols) ? cost_matrix[i][j] : 0;
      p->assignment[i][j] = 0;

      if (max_cost < p->cost[i][j])
	max_cost = p->cost[i][j];
    }
  }


  if (mode == HUNGARIAN_MODE_MAXIMIZE_UTIL) {
    for(i=0; i<p->num_rows; i++) {
      for(j=0; j<p->num_cols; j++) {
	p->cost[i][j] =  max_cost - p->cost[i][j];
      }
    }
  }
  else if (mode == HUNGARIAN_MODE_MINIMIZE_COST) {
    // nothing to do
  }
  else 
    fprintf(stderr,"%s: unknown mode. Mode was set to HUNGARIAN_MODE_MINIMIZE_COST !\n", __FUNCTION__);
  
  return rows;
}




void hungarian_free(hungarian_problem_t* p) {
  int i;
  for(i=0; i<p->num_rows; i++) {
    free(p->cost[i]);
    free(p->assignment[i]);
  }
  free(p->cost);
  free(p->assignment);
  p->cost = NULL;
  p->assignment = NULL;
}



void hungarian_solve(hungarian_problem_t* p)
{
  int i, j, m, n, k, l, s, t, q, unmatched, cost;
  int* col_mate;
  int* row_mate;
  int* parent_row;
  int* unchosen_row;
  int* row_dec;
  int* col_inc;
  int* slack;
  int* slack_row;

  cost=0;
  m =p->num_rows;
  n =p->num_cols;

  col_mate = (int*)calloc(p->num_rows,sizeof(int));
  hungarian_test_alloc(col_mate);
  unchosen_row = (int*)calloc(p->num_rows,sizeof(int));
  hungarian_test_alloc(unchosen_row);
  row_dec  = (int*)calloc(p->num_rows,sizeof(int));
  hungarian_test_alloc(row_dec);
  slack_row  = (int*)calloc(p->num_rows,sizeof(int));
  hungarian_test_alloc(slack_row);

  row_mate = (int*)calloc(p->num_cols,sizeof(int));
  hungarian_test_alloc(row_mate);
  parent_row = (int*)calloc(p->num_cols,sizeof(int));
  hungarian_test_alloc(parent_row);
  col_inc = (int*)calloc(p->num_cols,sizeof(int));
  hungarian_test_alloc(col_inc);
  slack = (int*)calloc(p->num_cols,sizeof(int));
  hungarian_test_alloc(slack);

  for (i=0;i<p->num_rows;i++) {
    col_mate[i]=0;
    unchosen_row[i]=0;
    row_dec[i]=0;
    slack_row[i]=0;
  }
  for (j=0;j<p->num_cols;j++) {
    row_mate[j]=0;
    parent_row[j] = 0;
    col_inc[j]=0;
    slack[j]=0;
  }

  for (i=0;i<p->num_rows;++i)
    for (j=0;j<p->num_cols;++j)
      p->assignment[i][j]=HUNGARIAN_NOT_ASSIGNED;

  // Begin subtract column minima in order to start with lots of zeroes 12
  if (verbose)
    fprintf(stderr, "Using heuristic\n");
  for (l=0;l<n;l++)
    {
      s=p->cost[0][l];
      for (k=1;k<m;k++) 
	if (p->cost[k][l]<s)
	  s=p->cost[k][l];
      cost+=s;
      if (s!=0)
	for (k=0;k<m;k++)
	  p->cost[k][l]-=s;
    }
  // End subtract column minima in order to start with lots of zeroes 12

  // Begin initial state 16
  t=0;
  for (l=0;l<n;l++)
    {
      row_mate[l]= -1;
      parent_row[l]= -1;
      col_inc[l]=0;
      slack[l]=INF;
    }
  for (k=0;k<m;k++)
    {
      s=p->cost[k][0];
      for (l=1;l<n;l++)
	if (p->cost[k][l]<s)
	  s=p->cost[k][l];
      row_dec[k]=s;
      for (l=0;l<n;l++)
	if (s==p->cost[k][l] && row_mate[l]<0)
	  {
	    col_mate[k]=l;
	    row_mate[l]=k;
	    if (verbose)
	      fprintf(stderr, "matching col %d==row %d\n",l,k);
	    goto row_done;
	  }
      col_mate[k]= -1;
      if (verbose)
	fprintf(stderr, "node %d: unmatched row %d\n",t,k);
      unchosen_row[t++]=k;
    row_done:
      ;
    }
  // End initial state 16
 
  // Begin Hungarian algorithm 18
  if (t==0)
    goto done;
  unmatched=t;
  while (1)
    {
      if (verbose)
	fprintf(stderr, "Matched %d rows.\n",m-t);
      q=0;
      while (1)
	{
	  while (q<t)
	    {
	      // Begin explore node q of the forest 19
	      {
		k=unchosen_row[q];
		s=row_dec[k];
		for (l=0;l<n;l++)
		  if (slack[l])
		    {
		      int del;
		      del=p->cost[k][l]-s+col_inc[l];
		      if (del<slack[l])
			{
			  if (del==0)
			    {
			      if (row_mate[l]<0)
				goto breakthru;
			      slack[l]=0;
			      parent_row[l]=k;
			      if (verbose)
				fprintf(stderr, "node %d: row %d==col %d--row %d\n",
				       t,row_mate[l],l,k);
			      unchosen_row[t++]=row_mate[l];
			    }
			  else
			    {
			      slack[l]=del;
			      slack_row[l]=k;
			    }
			}
		    }
	      }
	      // End explore node q of the forest 19
	      q++;
	    }
 
	  // Begin introduce a new zero into the matrix 21
	  s=INF;
	  for (l=0;l<n;l++)
	    if (slack[l] && slack[l]<s)
	      s=slack[l];
	  for (q=0;q<t;q++)
	    row_dec[unchosen_row[q]]+=s;
	  for (l=0;l<n;l++)
	    if (slack[l])
	      {
		slack[l]-=s;
		if (slack[l]==0)
		  {
		    // Begin look at a new zero 22
		    k=slack_row[l];
		    if (verbose)
		      fprintf(stderr, 
			     "Decreasing uncovered elements by %d produces zero at [%d,%d]\n",
			     s,k,l);
		    if (row_mate[l]<0)
		      {
			for (j=l+1;j<n;j++)
			  if (slack[j]==0)
			    col_inc[j]+=s;
			goto breakthru;
		      }
		    else
		      {
			parent_row[l]=k;
			if (verbose)
			  fprintf(stderr, "node %d: row %d==col %d--row %d\n",t,row_mate[l],l,k);
			unchosen_row[t++]=row_mate[l];
		      }
		    // End look at a new zero 22
		  }
	      }
	    else
	      col_inc[l]+=s;
	  // End introduce a new zero into the matrix 21
	}
    breakthru:
      // Begin update the matching 20
      if (verbose)
	fprintf(stderr, "Breakthrough at node %d of %d!\n",q,t);
      while (1)
	{
	  j=col_mate[k];
	  col_mate[k]=l;
	  row_mate[l]=k;
	  if (verbose)
	    fprintf(stderr, "rematching col %d==row %d\n",l,k);
	  if (j<0)
	    break;
	  k=parent_row[j];
	  l=j;
	}
      // End update the matching 20
      if (--unmatched==0)
	goto done;
      // Begin get ready for another stage 17
      t=0;
      for (l=0;l<n;l++)
	{
	  parent_row[l]= -1;
	  slack[l]=INF;
	}
      for (k=0;k<m;k++)
	if (col_mate[k]<0)
	  {
	    if (verbose)
	      fprintf(stderr, "node %d: unmatched row %d\n",t,k);
	    unchosen_row[t++]=k;
	  }
      // End get ready for another stage 17
    }
 done:

  // Begin doublecheck the solution 23
  for (k=0;k<m;k++)
    for (l=0;l<n;l++)
      if (p->cost[k][l]<row_dec[k]-col_inc[l])
	exit(0);
  for (k=0;k<m;k++)
    {
      l=col_mate[k];
      if (l<0 || p->cost[k][l]!=row_dec[k]-col_inc[l])
	exit(0);
    }
  k=0;
  for (l=0;l<n;l++)
    if (col_inc[l])
      k++;
  if (k>m)
    exit(0);
  // End doublecheck the solution 23
  // End Hungarian algorithm 18

  for (i=0;i<m;++i)
    {
      p->assignment[i][col_mate[i]]=HUNGARIAN_ASSIGNED;
      /*TRACE("%d - %d\n", i, col_mate[i]);*/
    }
  for (k=0;k<m;++k)
    {
      for (l=0;l<n;++l)
	{
	  /*TRACE("%d ",p->cost[k][l]-row_dec[k]+col_inc[l]);*/
	  p->cost[k][l]=p->cost[k][l]-row_dec[k]+col_inc[l];
	}
      /*TRACE("\n");*/
    }
  for (i=0;i<m;i++)
    cost+=row_dec[i];
  for (i=0;i<n;i++)
    cost-=col_inc[i];
  if (verbose)
    fprintf(stderr, "Cost is %d\n",cost);


  free(slack);
  free(col_inc);
  free(parent_row);
  free(row_mate);
  free(slack_row);
  free(row_dec);
  free(unchosen_row);
  free(col_mate);
}




REGISTER_OP("Hungarian")
    .Input("true_boxes: float32")
    .Input("pred_boxes: float32")
    .Input("box_classes: int32")
    .Input("overlap_threshold: float32")
    .Output("assignments: int32")
    .Output("permuted_box_classes: int32")
    .Output("permuted_true_boxes: float32")
    .Output("pred_boxes_mask: float32");

class HungarianOp : public OpKernel {
 public:
  explicit HungarianOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& pred_input = context->input(0);
    const Tensor& true_input = context->input(1);
    const Tensor& class_input = context->input(2);
    const Tensor& overlap_input = context->input(3);
    auto pred_boxes = pred_input.tensor<float, 3>();
    auto true_boxes = true_input.tensor<float, 3>();
    auto box_classes = class_input.matrix<int>();
    auto iou_threshold = overlap_input.scalar<float>()(0);

    // Create an output tensor
    Tensor* assignments_tensor = NULL;
    std::vector<int64> pred_shape;
    for (int i = 0; i < pred_input.shape().dims(); ++i) {
      pred_shape.push_back(pred_input.shape().dim_size(i));
    }
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({pred_shape[0], pred_shape[1]}),
                                                     &assignments_tensor));
    auto assignments_output = assignments_tensor->matrix<int>();

    Tensor* permuted_class_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({pred_shape[0], pred_shape[1]}),
                                                     &permuted_class_tensor));
    auto permuted_class_output = permuted_class_tensor->matrix<int>();

    Tensor* permuted_true_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, true_input.shape(),
                                                     &permuted_true_tensor));
    auto permuted_true_output = permuted_true_tensor->tensor<float, 3>();

    Tensor* pred_mask_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, pred_input.shape(),
                                                     &pred_mask_tensor));
    auto pred_mask_output = pred_mask_tensor->tensor<float, 3>();

    std::vector<int> num_gt_;

    num_gt_.clear();
    const int batch_size = pred_input.shape().dim_sizes()[0]; 
    const int num_pred = pred_input.shape().dim_sizes()[1];
    const int channels = pred_input.shape().dim_sizes()[2];
    for (int n = 0; n < batch_size; ++n) {
      num_gt_.push_back(0);
      for (int j = 0; j < num_pred; ++j) {
        // Count the number of boxes not in the empty (0) class
        if (box_classes(n, j) > 0) {
          num_gt_[n] += 1;
        }
      }
      assert(channels == 4);
  
      vector<float> match_cost;
      vector<float> loss_mat;
      for (int i = 0; i < num_pred; ++i) {
        for (int j = 0; j < num_pred; ++j) {
          const int idx = i * num_pred + j;
          match_cost.push_back(0.);
          loss_mat.push_back(0.);
          if (j >= num_gt_[n]) { continue; }
          for (int c = 0; c < channels; ++c) {
            const float pred_value = pred_boxes(n, i, c);
            const float label_value = true_boxes(n, j, c);
            match_cost[idx] += fabs(pred_value - label_value) / 2000.;
            loss_mat[idx] += fabs(pred_value - label_value);
          }
          assert(match_cost[idx] < 0.9);
          match_cost[idx] += i;
          const int c_x = 0;
          const int c_y = 1;
          const int c_w = 2;
          const int c_h = 3;
  
          const float pred_x = pred_boxes(n, i, c_x);
          const float pred_y = pred_boxes(n, i, c_y);
          const float pred_w = MAX(pred_boxes(n, i, c_w), 1.); // Rect is at least 1 pixel wide
          const float pred_h = MAX(pred_boxes(n, i, c_h), 1.); // Rect is at least 1 pixel tall
          const Rect pred_rect = Rect(pred_x, pred_y, pred_w, pred_h, 0.);
  
          const float true_x = true_boxes(n, j, c_x);
          const float true_y = true_boxes(n, j, c_y);
          const float true_w = true_boxes(n, j, c_w);
          const float true_h = true_boxes(n, j, c_h);
          const Rect true_rect = Rect(true_x, true_y, true_w, true_h, 0.);
  
          //const float iou_threshold = 0.25;
  
          if (true_rect.iou(pred_rect) < iou_threshold) {
            match_cost[idx] += 100;
          }
  
          //float ratio = 1.0;
          //if (fabs(pred_x - true_x) / true_w > ratio ||
              //fabs(pred_y - true_y) / true_h > ratio) {
            //match_cost[idx] += 100;
          //}
        }
      }

      vector<int> assignment = get_assignment(match_cost, loss_mat, num_gt_[n], num_pred);
      for (int i = 0; i < num_pred; ++i) {
        assignments_output(n, i) = assignment[i];
        const bool assigned = assignment[i] > -1;
        permuted_class_output(n, i) = assigned ? box_classes(n, assignment[i]) : 0;
        for (int c = 0; c < channels; ++c) {
          permuted_true_output(n, i, c) = assigned ? true_boxes(n, assignment[i], c) : 0.;
          pred_mask_output(n, i, c) = assigned ? 1. : 0.;
        }
      }

    }
  }

  vector<int> get_assignment(vector<float> match_cost, vector<float> loss_mat, int num_true, int num_pred) {
    float loss = 0.;
    double max_pair_cost = 0;
    for (int i = 0; i < num_pred; ++i) {
      for (int j = 0; j < num_pred; ++j) {
        const int idx = i * num_pred + j;
        max_pair_cost = std::max(max_pair_cost, fabs(match_cost[idx]));
      }
    }
    const int kMaxNumPred = 20;
    CHECK_LE(num_pred, kMaxNumPred);
    int int_cost[kMaxNumPred * kMaxNumPred];
    for (int i = 0; i < num_pred; ++i) {
      for (int j = 0; j < num_pred; ++j) {
        const int idx = i * num_pred + j;
        int_cost[idx] = static_cast<int>(
            match_cost[idx] / max_pair_cost * float(INT_MAX) / 2.);
      }
    }
  
    std::vector<int> assignment;
  
    bool permute_matches = true;
    // bool permute_matches = this->layer_param_.hungarian_loss_param().permute_matches()
    if (permute_matches) {
      hungarian_problem_t p;
      int** m = array_to_matrix(int_cost, num_pred, num_pred);
      hungarian_init(&p, m, num_pred, num_pred, HUNGARIAN_MODE_MINIMIZE_COST);
      hungarian_solve(&p);
      for (int i = 0; i < num_pred; ++i) {
        for (int j = 0; j < num_pred; ++j) {
          if (p.assignment[i][j] == HUNGARIAN_ASSIGNED) {
            assignment.push_back(j < num_true ? j : -1);
          }
        }
      }
      CHECK_EQ(assignment.size(), num_pred);
      hungarian_free(&p);
      for (int i = 0; i < num_pred; ++i) {
        free(m[i]);
      }
      free(m);
    } else {
      for (int i = 0; i < num_pred; ++i) {
        assignment.push_back(i < num_true ? i : -1);
      }
    }
    for (int i = 0; i < num_pred; ++i) {
      const int idx = i * num_pred + assignment[i];
      loss += loss_mat[idx];
    }
  
    return assignment;
  }
};
REGISTER_KERNEL_BUILDER(Name("Hungarian").Device(DEVICE_CPU), HungarianOp);

