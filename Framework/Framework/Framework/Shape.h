#pragma once
#include <vector>

struct Shape {
	typedef std::vector<size_t> ShapeType;

	Shape() {}
	Shape(const ShapeType& shape_) : shape(shape_) {
		if (shape.empty()) {
			throw std::exception();
		}
	}

	size_t rows() const {
		if (shape.size() >= 2) {
			return shape[shape.size() - 2];
		}
		return 1;
	}

	size_t cols() const {
		return shape.back();
	}

	bool empty() const {
		return dims() == 0;
	}

	size_t operator[](size_t index) const { return shape[index]; }
	size_t operator[](size_t index) { return shape[index]; }

	size_t dims() const { return shape.size(); }
	size_t back() const { return shape.back(); }
	void swap(Shape& rhs) { shape.swap(rhs.shape); }

	bool operator==(const Shape& rhs) const {
		return shape == rhs.shape;
	}
	bool operator!=(const Shape& rhs) const {
		return shape != rhs.shape;
	}

	operator ShapeType() {
		return shape;
	}

	ShapeType shape;
};
