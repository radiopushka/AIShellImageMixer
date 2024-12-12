#include <png.h>
#include <stdlib.h>
#include "writer.h"
#include "amplifier.c"






int image_write(char* image,float* R,float* G, float* B,int width,int height){
  FILE* im=fopen(image,"wb");

  amplify_image(R,width*height);
  amplify_image(G,width*height);
  amplify_image(B,width*height);


  if(!im)
    return -1;
  png_structp png=png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if(!png){
    printf("png_structp\n");
    return -1;
  }
  png_infop info = png_create_info_struct(png);
  if(!info){
    printf("png_infop\n");
    return -1;
  }
  if(setjmp(png_jmpbuf(png))){
    printf("jmp\n");
    return -1;
  }
  png_init_io(png, im);
  png_set_palette_to_rgb(png);

  png_bytep* rows=malloc(sizeof(png_bytep)*height);

  for(int i=0;i<height;i++){
    rows[i]=(png_byte*)malloc(sizeof(png_bytep)*width*3);
  }

  for(int i =0 ; i < height;i++){
    for(int i2 = 0; i2 < width*3;){
      rows[i][i2++] = *B;
      rows[i][i2++] = *G;
      rows[i][i2++] = *R;

      //printf(" %g %g %g",*B,*G,*R);
      B++; R++; G++;
    }
  }

  png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB,
      PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);


  png_write_info(png, info);
  printf("\nwriting image\n");
  png_write_image(png, rows);
  png_write_end(png, info);
  png_destroy_write_struct(&png, &info);

  for(int i = 0; i < height; i++){
    free(rows[i]);
  }
  free(rows);
  fclose(im);
  return 1;
}
