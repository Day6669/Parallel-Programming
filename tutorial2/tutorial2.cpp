#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"


using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test_large.ppm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input");
		
		std::vector<int> hist(1024); // make grow
		for (int i=0; i<hist.size(); i++){
			hist[i] = 0;
		}
		

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - device operations
		cl::Event t;
		int total = 0;
		int temp = 0;

		
		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); 
		
		cl::Buffer histogram(context, CL_MEM_READ_WRITE, hist.size());
		
		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueWriteBuffer(histogram, CL_TRUE, 0, hist.size(), &hist.data()[0]);
	
		vector<unsigned char> output_buffer(image_input.size());


		//populates the histogram
		cl::Kernel kernelH = cl::Kernel(program, "createHist");
		kernelH.setArg(0, dev_image_input);
		kernelH.setArg(1, histogram);

		queue.enqueueNDRangeKernel(kernelH, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &t);
		queue.enqueueReadBuffer(histogram, CL_TRUE, 0, hist.size(), &hist.data()[0]);
		
		temp = t.getProfilingInfo<CL_PROFILING_COMMAND_END>() - t.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		total += temp;
		std::cout << "Histogram population: " << temp << endl;
		
		
		
		// blelloch scan
		cl::Kernel kernelB = cl::Kernel(program, "blelloch");
		kernelB.setArg(0, histogram);
		
		queue.enqueueNDRangeKernel(kernelB, cl::NullRange, cl::NDRange(hist.size()), cl::NullRange, NULL, &t);
		queue.enqueueReadBuffer(histogram, CL_TRUE, 0, hist.size(), &hist[0]);
		queue.enqueueWriteBuffer(histogram, CL_TRUE, 0, hist.size(), &hist.data()[0]);
		
		temp = t.getProfilingInfo<CL_PROFILING_COMMAND_END>() - t.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		total += temp;
		std::cout << "Blelloch scan: " << temp << endl;
		
		
		
		// histogram normalization 
		float scl = hist[255] / 512;
		for (int i=0; i < hist.size(); i++){
			hist[i] -= image_input.get_channel(0).size()* 3;
			hist[i] /= scl;
		}
		queue.enqueueWriteBuffer(histogram, CL_TRUE, 0, hist.size(), &hist.data()[0]);
		
		
		
		// applies histogram to image
		cl::Kernel kernelA = cl::Kernel(program, "applyHistogram");
		kernelA.setArg(0, dev_image_input);
		kernelA.setArg(1, dev_image_output);
		kernelA.setArg(2, histogram);
		
		queue.enqueueNDRangeKernel(kernelA, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &t);
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);
		
		temp = t.getProfilingInfo<CL_PROFILING_COMMAND_END>() - t.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		total += temp;
		std::cout << "Histogram application: " << temp << endl;
		
		std::cout << "Total Time (ns): " << total << endl;
		
		
		
		

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		//output_image.YCbCrtoRGB();
		
		CImgDisplay disp_output(output_image,"output");

 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
