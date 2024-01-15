#pragma once
#include <exception>
#include <algorithm>
#include <string>
#include <iostream>
#include "Shape.h"

#define MATRIX_PRINT(m) (m.print(#m))

struct Shape2d : public Shape
{
	typedef std::vector<size_t> ShapeType;

	Shape2d() {}
	Shape2d(size_t rows, size_t cols) : Shape({ rows, cols })
	{
	}

	Shape2d(const ShapeType& shape_) : Shape(shape_)
	{
		if (shape_.size() != 2) {
			throw std::exception();
		}
	}

	void swap(Shape2d& rhs) { shape.swap(rhs.shape); }
};

// row * col
// 5 * 2 = 
// a b
// c d
// e f
// g h
// i j
// a b c d e f g h i j
// 0 1 2 3 4 5 6 7 8 9
// [0][0] => 0*2 + 0 = 0 = a
// [0][1] => 0*2 + 1 = 1 = b
// [1][0] => 1*2 + 0 = 2 = c
// [1][1] => 1*2 + 1 = 3 = d
// [2][0] => 2*2 + 0 = 4 = e
// [2][1] => 2*2 + 1 = 5 = f
// [3][0] => 3*2 + 0 = 6 = g
// [3][1] => 3*2 + 1 = 7 = h
// [4][0] => 4*2 + 0 = 8 = i
// [4][1] => 4*2 + 1 = 9 = j

template <class T>
class Matrix2d
{
public:
	Matrix2d() {}

	Matrix2d(size_t rows, size_t cols, const T& val = T{}) : shape(rows, cols) {
		Init(val);
	}

	Matrix2d(const std::vector<size_t>& shape_, const T& val = T{}) : shape(shape_) {
		Init(val);
	}

	template<class Generator>
	Matrix2d(const Shape2d& shape_, Generator gen) : shape(shape_) {
		Init();

		std::generate(buffer.begin(), buffer.end(), gen);
	}

	Matrix2d(const std::vector<std::vector<float>>& input) {
		if (input.empty()) {
			return;
		}
		if (input[0].empty()) {
			return;
		}
		size_t rows = input.size();
		size_t cols = input[0].size();
		shape = Shape2d(rows, cols);

		Init();

		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				set({ i, j }, input[i][j]);
			}
		}
	}

	Matrix2d(const Matrix2d<T>& rhs) : shape(rhs.shape) {
		Init();
		CopyFrom(rhs);
	}

	Matrix2d(const Matrix2d<T>&& rhs) noexcept
		: shape(std::move(rhs.shape)), buffer(std::move(rhs.buffer))
	{
	}

	Matrix2d<T>& operator= (const Matrix2d<T>& rhs) {
		if (!shape.empty() && (shape != rhs.shape)) {
			throw std::exception();
		}
		Matrix2d<T> temp(rhs);
		swap(temp);
		return *this;
	}

	void swap(Matrix2d<T>& rhs) {
		shape.swap(rhs.shape);
		buffer.swap(rhs.buffer);
	}

	const T& get(const Shape2d& indexes) const {
		size_t index = get_index(indexes.rows(), indexes.cols());
		return buffer[index];
	}

	//const T& get(const std::vector<size_t>& shape_) const {
	//	return get(Shape2d(shape_));
	//}

	const T& get(size_t row, size_t col) const {
		return get(Shape2d({ row, col }));
	}

	Matrix2d<T> get_row(size_t row_index) const {
		if (row_index >= rows()) {
			throw std::exception();
		}

		Matrix2d<T> ret({ 1, shape.cols() });
		for (size_t c = 0; c < shape.cols(); ++c) {
			ret.set({ 0, c }, get({ row_index, c }));
		}

		return ret;
	}

	void set(const Shape2d& shape_, const T& val) {
		size_t index = get_index(shape_.rows(), shape_.cols());
		buffer[index] = val;
	}

	//void set(const std::vector<size_t>& shape_, const T& val) {
	//	set(Shape2d(shape_), val);
	//}

	const Shape2d& get_shape() const {
		return shape;
	}

	size_t rows() const {
		return shape.rows();
	}

	size_t cols() const {
		return shape.cols();
	}

	void sum(const Matrix2d<T>& rhs) {
		if (shape.empty()) {
			return;
		}

		for (size_t i = 0; i < buffer.size(); ++i) {
			buffer[i] += rhs.buffer[i];
		}
	}

	static bool is_mult_valid(const Matrix2d<T>& A, const Matrix2d<T>& B) {
		if (A.cols() != B.rows()) {
			return false;
		}

		return true;
	}

	static Shape2d get_mult_dst_shape(const Matrix2d<T>& A, const Matrix2d<T>& B) {
		if (!is_mult_valid(A, B)) {
			throw std::exception();
		}

		return std::move(Shape2d({ A.rows(), B.cols() }));
	}
	// Currently only supported for 2D matrix
	void mult(const Matrix2d<T>& A, const Matrix2d<T>& B) {

		if (shape != get_mult_dst_shape(A, B)) {
			throw std::exception();
		}

		for (size_t i = 0; i < rows(); ++i) {
			for (size_t j = 0; j < cols(); ++j) {
				T sum{};
				for (size_t k = 0; k < A.cols(); ++k) {
					sum += A.get({ i, k }) * B.get({ k, j });
				}
				set({ i, j }, sum);
			}
		}
	}

	void print(const std::string& name = "") const {
		std::cout << std::endl;
		std::cout << " Matrix2d: " << name << "(" << rows() << "," << cols() << ")" << std::endl;
		std::cout << " -------------" << std::endl;
		if (shape.empty()) {
			return;
		}

		std::cout << " { " << std::endl;
		for (size_t i = 0; i < rows(); ++i) {
			std::cout << "\t{ ";
			for (size_t j = 0; j < cols(); ++j) {
				std::cout << get(i, j);
				if (j + 1 != cols()) {
					std::cout << ", ";
				}
			}
			std::cout << " } ";
			if (i + 1 != rows()) {
				std::cout << ",";
			}
			std::cout << std::endl;
		}
		std::cout << " }";
		std::cout << std::endl;
	}

private:

	void Init(const T& val = T{}) {
		if (shape.empty()) {
			return;
		}

		size_t bufferSize = 1;
		for (auto& dim : shape.shape) {
			bufferSize *= dim;
		}
		buffer.resize(bufferSize, val);
	}

	void CopyFrom(const Matrix2d<T>& rhs) {
		if (shape.empty()) {
			return;
		}

		// Copy the buffer
		buffer = rhs.buffer;
	}

	size_t get_index(size_t row, size_t col) const {
		if (shape.empty()) {
			throw std::exception();
		}

		size_t buffer_index = row*cols() + col;
		if (buffer_index > buffer.size()) {
			throw std::exception();
		}
		return buffer_index;
	}

	Shape2d shape;
	std::vector<T> buffer;
};
