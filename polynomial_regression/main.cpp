#include"polynomial_regression.hpp"
#include<algorithm>

#define DATA_SIZE 500
#define a0 1000.0
#define a1 8.0
#define a2 0.5
#define X_START 1.0f
#define X_END 10.0f

int main() {

	std::array<float, DATA_SIZE> dataY;
	std::array<float, DATA_SIZE> dataX;
	// Create dataX
	float xStep = (X_END - X_START) / static_cast<float>(DATA_SIZE);
	for (std::size_t i = 0; i < dataX.size();i++) {
		dataX[i] = X_START + i * xStep;
	}

	// Initialize dataY
	std::transform(dataX.begin(), dataX.end(), dataY.begin(),
		[&](const float& x) {
			return a0 + a1 * x + a2*x*x;
		});

	PolynomialRegressor1D_GN<3, DATA_SIZE> polRegressor(dataX, dataY);
	std::array<float, 3> initialCoefficients{ 70.0,1.0,0.1 };
	auto start = std::chrono::high_resolution_clock::now();
	polRegressor.InitializeCoefficients(initialCoefficients);
	polRegressor.Fit();
	auto finish = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << "ns\n";

	return 0;
}