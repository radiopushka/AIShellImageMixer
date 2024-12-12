#ifndef IMAGEIO_HEADER
#define IMAGEIO_HEADER
#include "image_writer/writer.h"
#include "image_reader/reader.h"

/*
 *functions:
writes image as png file, returns -1 if failed
int image_write(char* image, float* R, float* G, float* B, int width,int height);

this one does not rescale anything just crops
int rescaled_read(char* image, float* R, float* G, float* B, int width,int height);
 
//prepares the color layer for neural network propagation
void prepare_data(float* color_info,int array_size);

//convert 0-1 to 0-255 color systems
void revert_data(float *data,int array_size);
 *
 *
 *
 */

#endif // !IMAGEIO_HEADER
