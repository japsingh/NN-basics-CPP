#include <iostream>
#include <vector>

namespace {
	static const std::vector<std::vector<float>> train_or = {
		{0, 0, 0},
		{0, 1, 1},
		{1, 0, 1},
		{1, 1, 1}
	};

	float sigmoidf(float x)
	{
		return 1 / (1 + exp(-x));
	}

	float rand_f() {
		return (float)rand() / (float)RAND_MAX;
	}

	void gcost(const std::vector<std::vector<float>>& train, float W1, float W2, float b,
		float& dw1, float& dw2, float& db) {
		dw1 = 0.;
		dw2 = 0.;
		db = 0.;
		size_t n = train.size();

		for (auto& t : train) {
			float x1 = t[0];
			float x2 = t[1];
			float y = t[2];
			float a = sigmoidf(W1 * x1 + W2 * x2 + b);
			float di = 2 * (a - y) * a * (1 - a);
			dw1 += di * x1;
			dw2 += di * x2;
			db  += di;
		}

		dw1 /= n;
		dw2 /= n;
		db /= n;
	}

	float cost(const std::vector<std::vector<float>>& train, float W1, float W2, float b)
	{
		float c = 0;
		for (auto& t : train) {
			float x1 = t[0];
			float x2 = t[1];
			float y = t[2];
			float yhat = sigmoidf(W1 * x1 + W2 * x2 + b);
			float diff = yhat - y;
			c += diff * diff;
		}
		c /= (float)train.size();

		return c;
	}

	int Train_Or(bool useDerivatives)
	{
		srand(42);
		float W1 = rand_f();
		float W2 = rand_f();
		float b = rand_f();
		float ep = 1e-3f;
		float lr = 1e-2f;

		for (int epoch = 0; epoch < 50000; ++epoch) {
			float dW1, dW2, db = 0.;

			if (useDerivatives) {
				gcost(train_or, W1, W2, b, dW1, dW2, db);
			}
			else {
				dW1 = (cost(train_or, W1 + ep, W2, b) - cost(train_or, W1, W2, b)) / ep;
				dW2 = (cost(train_or, W1, W2 + ep, b) - cost(train_or, W1, W2, b)) / ep;
				db = (cost(train_or, W1, W2, b + ep) - cost(train_or, W1, W2, b)) / ep;
			}

			W1 -= lr * dW1;
			W2 -= lr * dW2;
			b -= lr * db;
			std::cout << "Cost: " << cost(train_or, W1, W2, b) << std::endl;
			//std::cout << "dW1: " << dW1 << std::endl;
			//std::cout << "dW2: " << dW2 << std::endl;
			//std::cout << "db: " << db << std::endl;
		}

		std::cout << "N: " << train_or.size() << std::endl;
		std::cout << "W1: " << W1 << std::endl;
		std::cout << "W2: " << W2 << std::endl;
		std::cout << "b: " << b << std::endl;
		std::cout << "Cost: " << cost(train_or, W1, W2, b) << std::endl;

		for (auto& t : train_or) {
			float x1 = t[0];
			float x2 = t[1];
			float y = t[2];
			float yhat = sigmoidf(W1 * x1 + W2 * x2 + b);
			std::cout << x1 << " | " << x2 << " = " << yhat << std::endl;
		}

		return 0;
	}
}

int main()
{
	return Train_Or(true);
}