#pragma once
#include <exception>
#include <algorithm>
#include <string>
#include <iostream>
#include "Shape.h"

#define MATRIX_PRINT(m) (m.print(#m))

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

// batch * row * col
// 4 * 3 * 2 = 
// Z [ a b ]
//   [ c d ]
//   [ e f ]
// 
// Y [ g h ]
//   [ i j ]
//   [ k l ]
// 
// X [ m n ]
//   [ o p ]
//   [ q r ]
// 
// W [ s t ]
//   [ u v ]
//   [ w x ]
// 
// a b c d e f g h i j k l m n o p q r s t u v w x
// 0 1 2 3 4 5 6 7 8 9 1 1 1 1 1 1 1 1 1 1 2 2 2 2
//                     0 1 2 3 4 5 6 7 8 9 0 1 2 3
// 
// [0][0][0] => 0*3*2 + 0*2 + 0 = 0 = a
// [0][0][1] => 0*3*2 + 0*2 + 1 = 1 = b
// [0][1][0] => 0*3*2 + 1*2 + 0 = 2 = c
// [0][1][1] => 0*3*2 + 1*2 + 1 = 3 = d
// [0][2][0] => 0*3*2 + 2*2 + 0 = 4 = e
// [0][2][1] => 0*3*2 + 2*2 + 1 = 5 = f
// [1][0][0] => 1*3*2 + 0*2 + 0 = 6 = g
// [1][0][1] => 1*3*2 + 0*2 + 1 = 7 = h
// [1][1][0] => 1*3*2 + 1*2 + 0 = 8 = i
// [1][1][1] => 1*3*2 + 1*2 + 1 = 9 = j
// [1][2][0] => 1*3*2 + 2*2 + 0 = 10 = k
// [1][2][1] => 1*3*2 + 2*2 + 1 = 11 = l
// [2][0][0] => 2*3*2 + 0*2 + 0 = 12 = m
// [2][0][1] => 2*3*2 + 0*2 + 1 = 13 = m

template <class T>
class Matrix
{
public:
	Matrix() {}

	Matrix(const Shape& shape_, const T& val = T{}) : shape(shape_) {
		Init(val);
	}

	Matrix(const std::vector<size_t>& shape_, const T& val = T{}) : shape(shape_) {
		Init(val);
	}

	template<class Generator>
	Matrix(const Shape& shape_, Generator gen) : shape(shape_) {
		Init();
		if (shape.empty()) {
			return;
		}

		std::generate(buffer.begin(), buffer.end(), gen);
	}

	Matrix(const Matrix<T>& rhs) : shape(rhs.shape) {
		Init();
		CopyFrom(rhs);
	}

	Matrix(const Matrix<T>&& rhs) noexcept
		: shape(std::move(rhs.shape)), buffer(std::move(rhs.buffer))
	{
	}

	Matrix<T>& operator= (const Matrix<T>& rhs) {
		if (!shape.empty() && (shape != rhs.shape)) {
			throw std::exception();
		}
		Matrix<T> temp(rhs);
		swap(temp);
		return *this;
	}

	void swap(Matrix<T>& rhs) {
		shape.swap(rhs.shape);
		buffer.swap(rhs.buffer);
	}

	const T& get(const Shape& indexes) const {
		size_t index = get_index(indexes);
		return buffer[index];
	}

	const T& get(const std::vector<size_t>& shape_) const {
		return get(Shape(shape_));
	}

	// This operation is valid only for a 2d matrix
	Matrix<T> get_row(size_t row_index) const {
		if (shape.dims() != 2) {
			throw std::exception();
		}

		if (row_index >= rows()) {
			throw std::exception();
		}

		Matrix<T> ret({ 1, shape.cols() });
		for (size_t c = 0; c < shape.cols(); ++c) {
			ret.set({ 0, c }, get({ row_index, c }));
		}

		return ret;
	}

	void set(const Shape& shape_, const T& val) {
		size_t index = get_index(shape_);
		buffer[index] = val;
	}

	void set(const std::vector<size_t>& shape_, const T& val) {
		set(Shape(shape_), val);
	}

	const Shape& get_shape() const {
		return shape;
	}

	size_t dims() const {
		return shape.dims();
	}

	size_t rows() const {
		return shape.rows();
	}

	size_t cols() const {
		return shape.cols();
	}

	void sum(const Matrix<T>& rhs) {
		if (shape != rhs.shape) {
			throw std::exception();
		}

		if (shape.empty()) {
			return;
		}

		for (size_t i = 0; i < buffer.size(); ++i) {
			buffer[i] += rhs.buffer[i];
		}
	}

	static bool is_mult_valid(const Matrix<T>& A, const Matrix<T>& B) {
		if (A.dims() > 2 || B.dims() > 2) {
			return false;
		}

		if (A.cols() != B.rows()) {
			return false;
		}

		return true;
	}

	static Shape get_mult_dst_shape(const Matrix<T>& A, const Matrix<T>& B) {
		if (!is_mult_valid(A, B)) {
			throw std::exception();
		}

		return std::move(Shape({ A.rows(), B.cols() }));
	}
	// Currently only supported for 2D matrix
	void mult(const Matrix<T>& A, const Matrix<T>& B) {

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
		std::cout << " Matrix: " << name << std::endl;
		std::cout << " -----------" << std::endl;
		std::vector<size_t> parents;
		print_at_dimension(1, parents);
		std::cout << std::endl;
	}
private:

	void Init(const T& val = T{}) {
		if (shape.empty()) {
			return;
		}

		// shape of {3, 2, 2} requires buffer size of 3*2*2 = 12 elements
		size_t bufferSize = 1;
		for (auto& dim : shape.shape) {
			bufferSize *= dim;
		}
		buffer.resize(bufferSize, val);
	}

	void CopyFrom(const Matrix<T>& rhs) {
		if (shape != rhs.shape) {
			throw std::exception();
		}

		if (shape.empty()) {
			return;
		}

		// Copy the buffer
		buffer = rhs.buffer;
	}

	size_t get_index(const Shape& index_shape_) const {
		if (index_shape_.dims() != shape.dims()) {
			throw std::exception();
		}

		if (shape.empty()) {
			throw std::exception();
		}

		// find index
		if (index_shape_.dims() == 1) {
			if (index_shape_[0] >= shape[0]) {
				throw std::exception();
			}
			return index_shape_[0];
		}

		size_t buffer_index = 0;
		size_t factor = 1;
		for (int i = index_shape_.dims() - 1; i >= 0; --i) {
			buffer_index += index_shape_[i] * factor;
			factor *= shape[(size_t)i];
		}

		if (buffer_index > buffer.size()) {
			throw std::exception();
		}
		return buffer_index;
	}

	std::string get_tabs_string(size_t TABS) const {
		std::string tabsStr;
		for (size_t i = 0; i < TABS; ++i) {
			tabsStr += " ";
		}
		return tabsStr;
	}

	void print_at_dimension(size_t DIM, std::vector<size_t>& parents) const {
		std::string tabsStr = get_tabs_string(DIM);

		if (DIM == shape.dims()) {
			std::cout << tabsStr << "{ ";
			for (size_t i = 0; i < shape[DIM - 1]; ++i) {
				parents.push_back(i);
				std::cout << get(Shape(parents));
				if (i + 1 != shape[DIM - 1]) {
					std::cout << ", ";
				}
				parents.pop_back();
			}
			std::cout << " }";
			return;
		}

		std::cout << tabsStr << "{" << std::endl;

		for (size_t i = 0; i < shape[DIM - 1]; ++i) {
			parents.push_back(i);
			print_at_dimension(DIM + 1, parents);
			parents.pop_back();
			std::cout << std::endl;
		}

		std::cout << tabsStr << "}" << std::endl;
	}

	Shape shape;
	std::vector<T> buffer;
};
