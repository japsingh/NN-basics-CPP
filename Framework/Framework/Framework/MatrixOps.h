#pragma once
#include "Matrix.h"
#include "Matrix2d.h"
#include "Random.h"
#include "Activations.h"

static Matrix<int> RandomIntMatrix(const Shape::ShapeType& shape_, int low = 0, int high = 100) {
	return Matrix<int>(shape_, [&]() {
		return (int)(rand_f() * (high - low) + low);
		});
}

static Matrix2d<int> RandomIntMatrix2d(const Shape2d::ShapeType& shape_, int low = 0, int high = 100) {
	return Matrix2d<int>(shape_, [&]() {
		return (int)(rand_f() * (high - low) + low);
		});
}

static Matrix<float> RandomFloatMatrix(const Shape::ShapeType& shape_, float low = 0.f, float high = 1.f) {
	return Matrix<float>(shape_, [&]() {
		return rand_f() * (high - low) + low;
		});
}

static Matrix2d<float> RandomFloatMatrix2d(const Shape2d::ShapeType& shape_, float low = 0.f, float high = 1.f) {
	return Matrix2d<float>(shape_, [&]() {
		return rand_f() * (high - low) + low;
		});
}

template<class T>
static void MatSum(Matrix<T>& dst, const Matrix<T>& src) {
	dst.sum(src);
}

template<class T>
static void MatSum2d(Matrix2d<T>& dst, const Matrix2d<T>& src) {
	dst.sum(src);
}

template<class T>
static void MatMul(Matrix<T>& dst, const Matrix<T>& A, const Matrix<T>& B) {
	dst.mult(A, B);
}

template<class T>
static void MatMul2d(Matrix2d<T>& dst, const Matrix2d<T>& A, const Matrix2d<T>& B) {
	dst.mult(A, B);
}

static void MatSigmoid2d(Matrix2d<float>& X) {
	for (size_t i = 0; i < X.rows(); ++i) {
		for (size_t j = 0; j < X.cols(); ++j) {
			float a = Activations::sigmoidf(X.get({ i,j }));
			X.set({ i,j }, a);
		}
	}
}

template <class T>
static void SplitXY(const Matrix2d<T>& data, Matrix2d<T>& X, Matrix2d<T>& Y) {
	if (data.cols() < 2) {
		throw std::exception();
	}

	size_t X_rows = data.rows();
	size_t X_cols = data.cols() - 1;
	X = Matrix2d<T>(X_rows, X_cols);

	size_t Y_rows = data.rows();
	size_t Y_cols = 1;
	Y = Matrix2d<T>(Y_rows, Y_cols);

	for (size_t i = 0; i < data.rows(); ++i) {
		Matrix2d<T> row = data.get_row(i);

		for (size_t j = 0; j < row.cols() - 1; ++j) {
			X.set({ i, j }, row.get({ 0,j }));
		}

		Y.set({ i,0 }, row.get({ 0,row.cols() - 1 }));
	}
}