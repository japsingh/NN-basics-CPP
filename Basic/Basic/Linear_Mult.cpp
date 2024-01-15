#include <iostream>

namespace {
	static float train[][2] = {
		{0,0},
		{1,3},
		{2,6},
		{3,9},
		{4,12}
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

	float dcost(float W)
	{
		float dW = 0.0;
		for (auto t : train) {
			float xi = t[0];
			float yi = t[1];
			float yhat = 2*(xi*W -yi)*xi;
			dW += yhat;
		}
		dW /= N;

		return dW;
	}

	// Use brute force for error optimization
	int BruteForceMethod()
	{
		srand(82);
		float W = rand_f();
		float ep = 1e-2f;

		float error = 0;
		for (int epoch = 0; epoch < 3800; ++epoch) {
			error = cost(W);
			W -= ep;
		}
		std::cout << "Error: " << error << std::endl;
		std::cout << "N: " << N << std::endl;
		std::cout << "W: " << W << std::endl;

		return 0;
	}

	// Use gradient descent for error optimization
	int GradientDescentWithDerivatives()
	{
		srand(82);
		float W = rand_f();
		float ep = 1e-3f;
		float lr = 1e-3f;

		for (int epoch = 0; epoch < 550; ++epoch) {
			float dw = dcost(W);
			W -= lr * dw;
			std::cout << "Error: " << cost(W) << std::endl;
		}

		std::cout << "N: " << N << std::endl;
		std::cout << "W: " << W << std::endl;

		return 0;
	}

	// Use gradient descent for error optimization
	int GradientDescentWithFiniteDiff()
	{
		srand(82);
		float W = rand_f();
		float ep = 1e-3f;
		float lr = 1e-3f;

		for (int epoch = 0; epoch < 700; ++epoch) {
			float error = (cost(W + ep) - cost(W)) / ep;
			W -= lr * error;
			std::cout << "Error: " << error << std::endl;
		}

		std::cout << "N: " << N << std::endl;
		std::cout << "W: " << W << std::endl;

		return 0;
	}
}

//int main()
//{
//	//return BruteForceMethod();
//	//return GradientDescentWithFiniteDiff();
//	return GradientDescentWithDerivatives();
//}