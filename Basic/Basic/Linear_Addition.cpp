#include <iostream>

namespace {
	static float train[][2] = {
		{0,10},
		{1,11},
		{2,12},
		{3,13},
		{4,14}
	};
	static size_t N = sizeof(train) / sizeof(train[0]);

	float rand_f() {
		return (float)rand() / (float)RAND_MAX;
	}

	float cost(float W)
	{
		float c = 0;
		for (auto t : train) {
			float x = t[0];
			float y = t[1];
			float yhat = W * x;
			c += (yhat - y) * (yhat - y);
		}
		c /= N;

		return c;
	}

	int WithoutBias()
	{
		srand(82);
		float W = rand_f();
		float ep = 1e-3f;
		float lr = 1e-3f;

		for (int epoch = 0; epoch < 2000; ++epoch) {
			float dW = (cost(W + ep) - cost(W)) / ep;
			W -= lr * dW;
			//std::cout << "Cost: " << cost(W) << std::endl;
			//std::cout << "dW: " << dW << std::endl;
		}

		std::cout << "N: " << N << std::endl;
		std::cout << "W: " << W << std::endl;
		std::cout << "Cost: " << cost(W) << std::endl;

		return 0;
	}

	float cost_with_bias(float W, float b)
	{
		float c = 0;
		for (auto t : train) {
			float x = t[0];
			float y = t[1];
			float yhat = W * x + b;
			c += (yhat - y) * (yhat - y);
		}
		c /= N;

		return c;
	}

	int WithBias()
	{
		srand(82);
		float W = rand_f();
		float b = rand_f();
		float ep = 1e-3f;
		float lr = 1e-3f;

		for (int epoch = 0; epoch < 30000; ++epoch) {
			float dW = (cost_with_bias(W + ep, b) - cost_with_bias(W, b)) / ep;
			float db = (cost_with_bias(W, b + ep) - cost_with_bias(W, b)) / ep;
			W -= lr * dW;
			b -= lr * db;
			//std::cout << "Cost: " << cost_with_bias(W, b) << std::endl;
			//std::cout << "dW: " << dW << std::endl;
			//std::cout << "db: " << db << std::endl;
		}

		std::cout << "N: " << N << std::endl;
		std::cout << "W: " << W << std::endl;
		std::cout << "b: " << b << std::endl;
		std::cout << "Cost: " << cost_with_bias(W, b) << std::endl;

		return 0;
	}
}

//int main()
//{
//	return WithoutBias();
//	//return WithBias();
//}
