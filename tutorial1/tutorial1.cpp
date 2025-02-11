//g++ -std=c++0x tutorial1.cpp -o tutorial1 -lOpenCL
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include "Utils.h"

#include <iostream>
#include <vector>

void print_help() {
	std::cerr << "Application usage:" << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
		
		cl::Program::Sources sources;
		AddSources(sources, "kernels/my_kernels.cl");
		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		//catch (const cl::Error& err) {
		catch (...) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			//throw err;
		}

		//Part 3 - memory allocation
		//host - input
		std::vector<float> A = { 0.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5 }; //C++11 allows this type of initialisation
		std::vector<float> B = { 0.0, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5 };
		
		size_t vector_elements = A.size();//number of elements
		size_t vector_size = A.size()*sizeof(int);//size in bytes

		//host - output
		std::vector<float> C(vector_elements);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		//Part 4 - device operations

		//4.1 Copy arrays A and B to device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0]);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_add = cl::Kernel(program, "add");
		kernel_add.setArg(0, buffer_A);
		kernel_add.setArg(1, buffer_B);
		kernel_add.setArg(2, buffer_C);
		
		cl::Kernel kernel_mult = cl::Kernel(program, "mult");
		kernel_mult.setArg(0, buffer_C);
		kernel_mult.setArg(1, buffer_B);
		kernel_mult.setArg(2, buffer_C);
		
		queue.enqueueNDRangeKernel(kernel_mult, cl::NullRange,
		cl::NDRange(vector_elements), cl::NullRange);
		
		cl::Event prof_event;
		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange,
		cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);

		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0]);

		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;
		std::cout << "C = " << C << std::endl;
		
		// Time
		std::cout << "Kernel execution time [ns]:" <<
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		
		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;
	}
	//catch (cl::Error err) {
	catch (...) {
		//std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
