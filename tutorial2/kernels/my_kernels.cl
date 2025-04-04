//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	//this is just a copy operation, modify to filter out the individual colour channels
	B[id] = A[id];
}

//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	B[id] = A[id];
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	uint result = 0;

	for (int i = (x-1); i <= (x+1); i++)
	for (int j = (y-1); j <= (y+1); j++) 
		result += A[i + j*width + c*image_size];

	result /= 9;

	B[id] = (uchar)result;
}

//2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	float result = 0;

	for (int i = (x-1); i <= (x+1); i++)
	for (int j = (y-1); j <= (y+1); j++) 
		result += A[i + j*width + c*image_size]*mask[i-(x-1) + j-(y-1)];

	B[id] = (uchar)result;
}


// invert image :D
kernel void invert(global const int* A, global int* B){
	int x = get_global_id(0);
	int y = get_global_id(1);
	int c = get_global_id(2);
	
	int w = get_global_size(0);
	int h = get_global_size(1);
	
	int id = x + y * w + c * (w*h);
	
	B[id] = 255 - A[id];
}


// populates histogram 
kernel void createHist (global const uchar* A, global int* H) {
	int id = get_global_id(0);
	int index = A[id];
	int ch = get_global_id(2);
	
	if (ch == 0) {
		atomic_inc(&H[index]);
	}
}


// blelloch scan
kernel void blelloch(global int* H) {
	int id = get_global_id(0);
	int n = get_global_size(0);
	int temp;
	
	if (id < 256){
		for (int s=1; s<n; s*=2){
			if (((id+1) % (s*2)) == 0) {
				H[id] += H[id-s];
			}
			barrier(CLK_GLOBAL_MEM_FENCE);
		}
		
		if (id == 0) { H[n-1] = 0; }
		barrier (CLK_GLOBAL_MEM_FENCE);
		
		for (int s=n/2; s>0; s/=2) {
			if (((id+1) % (s*2)) == 0) {
				temp = H[id];
				H[id] += H[id-s];
				H[id-s] = temp;
			}
			barrier(CLK_GLOBAL_MEM_FENCE);
		}
	}
}


kernel void applyHistogram(global const uchar* A, global uchar* B, global int* H) {
	int id = get_global_id(0);
	int ch = get_global_id(2);
	int i = A[id];
	
	if (ch == 0){
		B[id] = H[i];
	}
}
