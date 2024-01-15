#pragma once
#include "MatrixOps.h"


class Layer {
public:
	Layer(size_t prev_layer_node_count, size_t cur_layer_node_count)
		: node_count(cur_layer_node_count),
		  w(RandomFloatMatrix2d({ prev_layer_node_count, cur_layer_node_count }, 0., 1.)),
		  b(RandomFloatMatrix2d({1, cur_layer_node_count }, 0., 1.)),
		  a({1, cur_layer_node_count }, 0.f)
	{
	}

	size_t get_node_count() const { return node_count; }

	// public members for easy accessibility
	Matrix2d<float> w;
	Matrix2d<float> b;
	Matrix2d<float> a;

private:
	size_t node_count;
};

class NN {
public:
	void append_layer(size_t cur_layer_count)
	{
		size_t prev_layer_node_count = 0;
		if (layers.empty()) {
			// inserting input layer
			prev_layer_node_count = 1;
		}
		else {
			prev_layer_node_count = layers.back().get_node_count();
		}

		layers.emplace_back(prev_layer_node_count, cur_layer_count);
	}

	void input(const Matrix2d<float>& input) {
		layers[0].a = input;
	}

	const Matrix2d<float>& output() {
		return layers.back().a;
	}

	void forward() {
		for (size_t i = 1; i < layers.size(); ++i) {
			// Multiply previous layer's activations with current layer's weights
			MatMul2d(layers[i].a, layers[i-1].a, layers[i].w);
			// Add current layer's bias
			MatSum2d(layers[i].a, layers[i].b);
			// Current layer's activation
			MatSigmoid2d(layers[i].a);
		}
	}

	float cost(const Matrix2d<float>& X, const Matrix2d<float>& Y)
	{
		size_t N = X.rows();
		float c = 0.f;
		for (size_t i = 0; i < X.rows(); ++i) {
			Matrix2d<float> r = X.get_row(i);
			input(r);
			forward();
			auto yhat_matrix = output();
			for (size_t j = 0; j < yhat_matrix.cols(); ++j) {
				float y = Y.get({ i, j });
				float yhat = yhat_matrix.get({ 0,j });
				float d = y - yhat;
				c += d * d;
			}
		}
		return c / N;
	}

	void train(float ep, float lr, const Matrix2d<float>& X, const Matrix2d<float>& Y) {
		std::vector<Layer> layers_updated = layers;
		float original_cost = cost(X, Y);

		for (size_t layer = 1; layer < layers.size(); ++layer) {
			for (size_t i = 0; i < layers[layer].w.rows(); ++i) {
				for (size_t j = 0; j < layers[layer].w.cols(); ++j) {
					// Save the original weight[i,j] value 
					float saved = layers[layer].w.get({ i,j });
					// Update the weight[i,j] value with epsilon
					layers[layer].w.set({ i,j }, saved + ep);
					// Calculate the differentiation due to small increment
					float cost_diff = (cost(X, Y) - original_cost)/ep;
					// gradient descent - Update the weight in the copy of model
					layers_updated[layer].w.set({ i,j }, saved - lr * cost_diff);
					// Restrore the original layer weight
					layers[layer].w.set({ i,j }, saved);
				}
			}
		}

		for (size_t layer = 1; layer < layers.size(); ++layer) {
			for (size_t i = 0; i < layers[layer].b.rows(); ++i) {
				for (size_t j = 0; j < layers[layer].b.cols(); ++j) {
					// Save the original bias[i,j] value 
					float saved = layers[layer].b.get({ i,j });
					// Update the bias[i,j] value with epsilon
					layers[layer].b.set({ i,j }, saved + ep);
					// Calculate the differentiation due to small increment
					float cost_diff = (cost(X, Y) - original_cost) / ep;
					// gradient descent - Update the bias in the copy of model
					layers_updated[layer].b.set({ i,j }, saved - lr * cost_diff);
					// Restrore the original layer weight
					layers[layer].b.set({ i,j }, saved);
				}
			}
		}

		// Learning - Save updated weights and biases
		layers = layers_updated;
	}

private:
	std::vector<Layer> layers;
};