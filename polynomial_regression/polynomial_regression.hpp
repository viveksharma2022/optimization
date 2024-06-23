#pragma once
#include<vector>
#include <array>
#include <iostream>
#include <chrono>
#define LEARN_RATE 0.00000005
#define LEARN_RATE_GN 5e-1

struct Point2Df {
	float x;
	float y;
	Point2Df(float x, float y) :
		x(x),
		y(y) {}
	Point2Df() = default;
};

template<int pOrder, int dataSize>
class PolynomialRegressor1D_GD {
protected: 
	std::array<float, dataSize> dataX;
	std::array<float, dataSize> dataY;
	std::array<float, pOrder> coefficients{};
	std::array<float, pOrder> differentiables{}; // differentiable of coefficients with respect to cost function
	std::array<float, pOrder> secondDifferentiables{}; // second differentiable of coefficients with respect to cost function
	std::array<std::vector<float>, pOrder> modelX{}; // x_raised
	std::array<float, dataSize> modelY{};
	std::array<float, dataSize> residualY{};
	float residual;
	float cost;
	uint64_t iterations;
	float tol;
public:
	PolynomialRegressor1D_GD(std::array<float, dataSize>& dataX,
		std::array<float, dataSize>& dataY) :
		dataX(dataX),
		dataY(dataY),
		residual(0.0),
		cost(0.0),
		iterations(1e4),
		tol(1E-3)
	{}
	void InitializeX(float start, float stop);
	void InitializeCoefficients(std::array<float,pOrder>& intialCoefficients);
	void ComputeModelMatrixX();
	void ComputeModelY();
	void ComputeResidual();
	void ComputeCost();
	void ComputeDifferentiables();
	void ComputeSecondDifferentiables();
	virtual void UpdateCoefficients();
	virtual void Fit();
};

template<int pOrder, int dataSize>
void PolynomialRegressor1D_GD<pOrder, dataSize>::InitializeX(float start, float stop) {
	float step = (stop - start) / dataSize;
	for (int i = 0; i < dataSize; i++) {
		dataX[i] = start + i * step;
	}
};

template<int pOrder, int dataSize>
void PolynomialRegressor1D_GD<pOrder, dataSize>::ComputeModelMatrixX() {
	for (int degree = 0; degree < pOrder; degree++) {
		switch (degree) {
		case 0:
			modelX[degree] = std::vector<float>(dataSize,1.0f);
			break;
		case 1:
			modelX[degree] = std::vector<float>(dataSize);
			for (int i = 0; i < dataSize; i++) {
				modelX[degree][i] = dataX[i];
			}
			break;
		default:
			modelX[degree] = std::vector<float>(dataSize);
			for (int i = 0; i < dataSize; i++) {
				modelX[degree][i] = modelX[degree-1][i] * dataX[i];
			}
			break;
		}
	}
};

template<int pOrder, int dataSize>
void PolynomialRegressor1D_GD<pOrder, dataSize>::Fit() {
	ComputeModelMatrixX();
	for (size_t iter = 0; iter < iterations; iter++) {
		ComputeModelY();
		ComputeResidual();
		ComputeCost();
		ComputeDifferentiables();
		UpdateCoefficients();
		/////// check loop break conditions
	}
	std::cout << "Cost function: " << cost << "\n";
	std::cout << "Coefficients:" << "\n";
	for (const auto& co : coefficients) {
		std::cout << co << "\n";
	}
}

template<int pOrder, int dataSize>
void PolynomialRegressor1D_GD<pOrder, dataSize>::InitializeCoefficients(std::array<float, pOrder>& initialCoefficients) {
	std::copy(initialCoefficients.begin(), initialCoefficients.end(), coefficients.begin());
}

template<int pOrder, int dataSize>
void PolynomialRegressor1D_GD<pOrder, dataSize>::ComputeModelY() {
	std::fill(modelY.begin(), modelY.end(), 0); // reset array to 0 as calculations are alaways updating
	for (int itx = 0; itx < dataSize; itx++) {
		for (int degree = 0; degree < pOrder; degree++) {
			modelY[itx] += coefficients[degree] * modelX[degree][itx];
		}
	}
}

template<int pOrder, int dataSize>
void PolynomialRegressor1D_GD<pOrder, dataSize>::ComputeCost() {
	cost = 0.0;
	for (const auto& res : residualY) {
		cost += res * res;
	}
}

template<int pOrder, int dataSize>
void PolynomialRegressor1D_GD<pOrder, dataSize>::ComputeDifferentiables() {
	std::fill(differentiables.begin(), differentiables.end(), 0.0); // reset all differentiables
	for (int degree = 0; degree < pOrder; degree++) {
		for (int idx = 0; idx < dataSize; idx++) {
			differentiables[degree] += (-2 * residualY[idx]) * modelX[degree][idx];
		}
	}
	int pause = 1;
}

template<int pOrder, int dataSize>
void PolynomialRegressor1D_GD<pOrder, dataSize>::ComputeSecondDifferentiables() {
	std::fill(secondDifferentiables.begin(), secondDifferentiables.end(), 0.0); // reset all differentiables
	for (int degree = 0; degree < pOrder; degree++) {
		for (int idx = 0; idx < dataSize; idx++) {
			secondDifferentiables[degree] += (2 * modelX[degree][idx] * modelX[degree][idx]);
		}
	}
};

template<int pOrder, int dataSize>
void PolynomialRegressor1D_GD<pOrder, dataSize>::ComputeResidual() {
	for (int idx = 0; idx < dataSize; idx++) {
		residualY[idx] = dataY[idx] - modelY[idx];
	}
}

template<int pOrder, int dataSize>
void PolynomialRegressor1D_GD<pOrder, dataSize>::UpdateCoefficients() {
	for (int degree = 0; degree < pOrder; degree++) {
		coefficients[degree] -= LEARN_RATE * differentiables[degree];
	}
}

// Gauss-newton method
template<int pOrder, int dataSize>
class PolynomialRegressor1D_GN: public PolynomialRegressor1D_GD<pOrder,dataSize> {
public:
	PolynomialRegressor1D_GN(std::array<float, dataSize>& dataX,
		std::array<float, dataSize>& dataY) :
		PolynomialRegressor1D_GD<pOrder,dataSize>(dataX, dataY) {}

	void UpdateCoefficients();
	void Fit();
};


template<int pOrder, int dataSize>
void PolynomialRegressor1D_GN<pOrder, dataSize>::Fit() {
		PolynomialRegressor1D_GD<pOrder, dataSize>::ComputeModelMatrixX();	
	this->ComputeSecondDifferentiables(); // second derivates are all constants, so are computed once
	for (size_t iter = 0; iter < PolynomialRegressor1D_GD<pOrder, dataSize>::iterations; iter++) {
		this->ComputeModelY();
		this->ComputeResidual();
		this->ComputeCost();
		this->ComputeDifferentiables();
		this->UpdateCoefficients();
		/////// check loop break conditions
	}
		std::cout << "Final cost function: " << this->cost << "\n";
		std::cout << "Coefficients:" << "\n";
	for (const auto& co : this->coefficients) {
		std::cout << co << "\n";
	}
}

template<int pOrder, int dataSize>
void PolynomialRegressor1D_GN<pOrder, dataSize>::UpdateCoefficients() {
	for (int degree = 0; degree < pOrder; degree++) {
		this->coefficients[degree] -=
			LEARN_RATE_GN*(this->differentiables[degree]/
				this->secondDifferentiables[degree]);
	}
}