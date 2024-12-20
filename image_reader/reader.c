#include <png.h>
#include <stdlib.h>
#include "reader.h"




int rescaled_read(char* image,float* out,int width,int height,int RGB){
  FILE* im=fopen(image,"rb");
  if(!im)
    return -1;
  png_structp png=png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if(!png)
    return -1;
  png_infop info = png_create_info_struct(png);
  if(!info)
    return -1;
  if(setjmp(png_jmpbuf(png)))
    return -1;
  png_init_io(png, im);
  png_read_info(png, info);
  
  int color_type = png_get_color_type(png, info);
  if(color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png);
  if(color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(png);

  png_read_update_info(png, info);
  png_bytep* rows=malloc(sizeof(png_bytep)*height);
  for(int i=0;i<height;i++){
    rows[i]=(png_byte*)malloc(png_get_rowbytes(png,info));
  }
  
  png_read_image(png, rows);
  fclose(im);
  png_destroy_read_struct(&png, &info, NULL);
  //unsigned long long area=width*height*4;
  unsigned long long yend=height;
  unsigned long long xend=width;
  unsigned long long i1;
  unsigned long long i2;
  unsigned long long i3=0;
  unsigned long long i4;
  int mult=3;
  if(color_type==PNG_COLOR_TYPE_RGB_ALPHA ){
    mult=4;
  }
  for(i1=0;i1<yend;i1++){
    i4=0;
    if(i3>=height-1){
      i3=height-1;
    }
    for(i2=i1*xend;i2<(i1*xend+xend);){
      if(RGB==1){
        *out=rows[i3][i4];
        out++;
        *out=rows[i3][i4+1];
        out++;
        *out=rows[i3][i4+2];
        out++;

      }else{
        float sum=0;
        sum= sum + rows[i3][i4];
        sum= sum + rows[i3][i4+1];
        sum= sum + rows[i3][i4+2];
        *out = sum/3;
        out++;
      }
        i2++;
        i4=i4+mult;
      
    }

      i3=i3+1;
  }
  for(int i=0;i<height;i++){
    free(rows[i]);
  }
  free(rows);
  png_destroy_read_struct(&png, &info,NULL);
  return 1;
}

void flatten(float* dim_data,int size){
  for(int i = 0; i < size; i++){

    
    *dim_data = (*dim_data)/255.0;

    dim_data++;
  }
}

void prepare_data(float* dim_data, int size){
  flatten(dim_data,size);
  float* start = dim_data;
  float max = 0;
  for(int i = 0; i < size; i++){
    if(*dim_data > max)
      max = *dim_data;
    dim_data++;
  }
  dim_data = start;
  float mult = 1/max;
  for(int i = 0; i < size; i++){
    *dim_data = (*dim_data) * mult;
    dim_data++;
  }
}

void revert_data(float* dim_data,int size){
  for(int i = 0; i < size; i++){
    *dim_data = (*dim_data) * 255;
    dim_data++;
  }
}


