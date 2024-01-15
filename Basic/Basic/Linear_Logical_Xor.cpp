
#include <iostream>
#include <vector>

namespace {
	struct Neuron {
		float W1;
		float W2;
		float b;
	};

	struct Xor {
		// layer 1, node 1
		Neuron n11;

		// layer 1, node 2
		Neuron n12;

		// layer 2, node 1
		Neuron n21;
	};

	static const std::vector<std::vector<float>> train_xor = {
		{0, 0, 0},
		{0, 1, 1},
		{1, 0, 1},
		{1, 1, 0}
	};

	static float ep = 1e-3f;
	static float lr = 1e-2f;

	float rand_f() {
		return (float)rand() / (float)RAND_MAX;
	}

	void init(Xor& x)
	{
		srand(69);

		x.n11.W1 = rand_f();
		x.n11.W2 = rand_f();
		x.n11.b = rand_f();

		x.n12.W1 = rand_f();
		x.n12.W2 = rand_f();
		x.n12.b = rand_f();

		x.n21.W1 = rand_f();
		x.n21.W2 = rand_f();
		x.n21.b = rand_f();
	}

	float sigmoidf(const float x)
	{
		return (float)1.0f / (1.0f + exp(-x));
	}

	float calc(const Neuron n, const float x1, const float x2)
	{
		float sig = sigmoidf(n.W1 * x1 + n.W2 * x2 + n.b);
		//std::cout << "x1: " << x1 << ", W1: " << n.W1 << ", x2: " << x2 << ", W2: " << n.W2 << ", b: " << n.b << ", Sig: " << sig << std::endl;
		return sig;
	}

	float model(const Xor& x, const float x1, const float x2)
	{
		float intermediate1 = calc(x.n11, x1, x2);
		float intermediate2 = calc(x.n12, x1, x2);
		float final1 = calc(x.n21, intermediate1, intermediate2);
		//std::cout 
		//	<< "x1: " << x1 << ", x2: " << x2 
		//	<< ", intermediate1: " << intermediate1 << ", intermediate2: " << intermediate2 
		//	<< ", final1: " << final1 << std::endl;
		return final1;
	}

	float cost(const Xor& x)
	{
		float c = 0;
		for (auto& t : train_xor) {
			float x1 = t[0];
			float x2 = t[1];
			float y = t[2];
			float yhat = model(x, x1, x2);
			float diff = yhat - y;
			c += diff * diff;
			//std::cout << "x1: " << x1 << ", x2: " << x2 << ", y: " << y << ", yhat: " << yhat << std::endl;
		}
		c /= (float)train_xor.size();

		return c;
	}

	float derivative(const Xor& dx, const Xor& x) {
		return (cost(dx) - cost(x)) / ep;
	}

	Xor getdx(const Xor& x) {
		Xor dx = x;

		Xor temp = x; temp.n11.W1 += ep;
		dx.n11.W1 -= lr * derivative(temp, x);
		//std::cout << "x.n11.W1: " << x.n11.W1 << ", temp.n11.W1: " << temp.n11.W1 << ", dx.n11.W1: " << dx.n11.W1 << std::endl;
		//std::cout << "cost x: " << cost(train, x) << ", cost temp: " << cost(train, temp) << std::endl;
		//std::cout << "gradient n11.W1: " << (cost(train, temp) - c) / ep << std::endl;
		temp = x; temp.n11.W2 += ep;
		dx.n11.W2 -= lr * derivative(temp, x);
		//std::cout << "x.n11.W2: " << x.n11.W2 << ", temp.n11.W2: " << temp.n11.W2 << ", dx.n11.W2: " << dx.n11.W2 << std::endl;
		temp = x; temp.n11.b += ep;
		dx.n11.b -= lr * derivative(temp, x);
		//std::cout << "x.n11.b: " << x.n11.b << ", temp.n11.b: " << temp.n11.b << ", dx.n11.b: " << dx.n11.b << std::endl;

		temp = x; temp.n12.W1 += ep;
		dx.n12.W1 -= lr * derivative(temp, x);
		//std::cout << "x.n12.W1: " << x.n12.W1 << ", temp.n12.W1: " << temp.n12.W1 << ", dx.n12.W1: " << dx.n12.W1 << std::endl;
		temp = x; temp.n12.W2 += ep;
		dx.n12.W2 -= lr * derivative(temp, x);
		//std::cout << "x.n12.W2: " << x.n12.W2 << ", temp.n12.W2: " << temp.n12.W2 << ", dx.n12.W2: " << dx.n12.W2 << std::endl;
		temp = x; temp.n12.b += ep;
		dx.n12.b -= lr * derivative(temp, x);
		//std::cout << "x.n12.b: " << x.n12.b << ", temp.n12.b: " << temp.n12.b << ", dx.n12.b: " << dx.n12.b << std::endl;

		temp = x; temp.n21.W1 += ep;
		dx.n21.W1 -= lr * derivative(temp, x);
		//std::cout << "x.n21.W1: " << x.n21.W1 << ", temp.n21.W1: " << temp.n21.W1 << ", dx.n21.W1: " << dx.n21.W1 << std::endl;
		temp = x; temp.n21.W2 += ep;
		dx.n21.W2 -= lr * derivative(temp, x);
		//std::cout << "x.n21.W2: " << x.n21.W2 << ", temp.n21.W2: " << temp.n21.W2 << ", dx.n21.W2: " << dx.n21.W2 << std::endl;
		temp = x; temp.n21.b += ep;
		dx.n21.b -= lr * derivative(temp, x);
		//std::cout << "x.n21.b: " << x.n21.b << ", temp.n21.b: " << temp.n21.b << ", dx.n21.b: " << dx.n21.b << std::endl;

		return dx;
	}

	void print(const Xor& x) {
		std::cout << "n11 (W1: " << x.n11.W1 << ", W2: " << x.n11.W2 << ", b: " << x.n11.b << ")" << std::endl;
		std::cout << "n12 (W1: " << x.n12.W1 << ", W2: " << x.n12.W2 << ", b: " << x.n12.b << ")" << std::endl;
		std::cout << "n21 (W1: " << x.n21.W1 << ", W2: " << x.n21.W2 << ", b: " << x.n21.b << ")" << std::endl;

	}

	int Train_Xor()
	{
		Xor x;
		init(x);
		std::cout << "Initial Xor: " << std::endl;
		print(x);
		std::cout << "-------------" << std::endl;

		for (int epoch = 0; epoch < 1000 * 1000; ++epoch) {
			x = getdx(x);
			//std::cout << "Cost: " << cost(x) << std::endl;
			//std::cout << "Updated Xor: " << std::endl;
			//print(dx);
			//std::cout << "-------------" << std::endl;
		}

		std::cout << "N: " << train_xor.size() << std::endl;
		print(x);
		std::cout << "Cost: " << cost(x) << std::endl;

		for (auto& t : train_xor) {
			float x1 = t[0];
			float x2 = t[1];
			float yhat = model(x, x1, x2);
			std::cout << x1 << " | " << x2 << " = " << yhat << std::endl;
		}

		std::cout << "Layer 1 Neuron 1" << std::endl;
		std::cout << "---------" << std::endl;
		for (auto& t : train_xor) {
			float x1 = t[0];
			float x2 = t[1];
			float yhat = calc(x.n11, x1, x2);
			std::cout << x1 << " | " << x2 << " = " << yhat << std::endl;
		}

		std::cout << "Layer 1 Neuron 2" << std::endl;
		std::cout << "---------" << std::endl;
		for (auto& t : train_xor) {
			float x1 = t[0];
			float x2 = t[1];
			float yhat = calc(x.n12, x1, x2);
			std::cout << x1 << " | " << x2 << " = " << yhat << std::endl;
		}

		std::cout << "Layer 2 Neuron 1" << std::endl;
		std::cout << "---------" << std::endl;
		for (auto& t : train_xor) {
			float x1 = t[0];
			float x2 = t[1];
			float yhat = calc(x.n21, x1, x2);
			std::cout << x1 << " | " << x2 << " = " << yhat << std::endl;
		}

		return 0;
	}
}

//int main()
//{
//	return Train_Xor();
//	//return WithBias();
//}
