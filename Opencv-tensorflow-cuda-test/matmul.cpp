// matmul.cpp

#include <vector>
#include <eigen/Dense>

#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"

using namespace tensorflow;

// Build a computation graph that takes a tensor of shape [?, 2] and
// multiplies it by a hard-coded matrix.
GraphDef CreateGraphDef()
{
	Scope root = Scope::NewRootScope();

	auto X = ops::Placeholder(root.WithOpName("x"), DT_FLOAT,
		ops::Placeholder::Shape({ -1, 2 }));
	auto A = ops::Const(root, { { 3.f, 2.f },{ -1.f, 0.f } });

	auto Y = ops::MatMul(root.WithOpName("y"), A, X,
		ops::MatMul::TransposeB(true));

	GraphDef def;
	TF_CHECK_OK(root.ToGraphDef(&def));

	return def;
}

int main()
{
	GraphDef graph_def = CreateGraphDef();

	// Start up the session
	SessionOptions options;
	std::unique_ptr<Session> session(NewSession(options));
	TF_CHECK_OK(session->Create(graph_def));

	// Define some data.  This needs to be converted to an Eigen Tensor to be
	// fed into the placeholder.  Note that this will be broken up into two
	// separate vectors of length 2: [1, 2] and [3, 4], which will separately
	// be multiplied by the matrix.
	std::vector<float> data = { 1, 2, 3, 4 };
	auto mapped_X_ = Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>>
		(&data[0], 2, 2);
	auto eigen_X_ = Eigen::Tensor<float, 2, Eigen::RowMajor>(mapped_X_);

	Tensor X_(DT_FLOAT, TensorShape({ 2, 2 }));
	X_.tensor<float, 2>() = eigen_X_;

	std::vector<Tensor> outputs;
	TF_CHECK_OK(session->Run({ { "x", X_ } }, { "y" }, {}, &outputs));

	// Get the result and print it out
	Tensor Y_ = outputs[0];
	std::cout << Y_.tensor<float, 2>() << std::endl;

	session->Close();
}