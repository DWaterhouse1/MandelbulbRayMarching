#include <RayMarcher.hpp>

#include <iostream>

int main()
{
	rmcuda::RayMarcher rayMarcher;

	if (rmcuda::RayMarcher::hasCudaSupport())
	{
		rayMarcher.run();
	}
	else
	{
		std::cerr << "inadequate cuda support" << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}