#ifndef PNGREADER
#define PNGREADER

int rescaled_read(char* image, float* R, float* G, float* B, int width,int height);

//prepares the color layer for neural network propagation
void prepare_data(float* color_info,int array_size);

//convert 0-1 to 0-255 color systems
void revert_data(float *data,int array_size);

#endif // !PNGREADER
