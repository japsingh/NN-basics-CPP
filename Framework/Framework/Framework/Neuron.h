#pragma once
#include "Matrix.h"

class Neuron {
public:
	void forward();

private:
	Matrix<float> w;
	Matrix<float> b;
	Matrix<float> a;
};