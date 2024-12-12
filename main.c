#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "imageio.h"
#include "CAI/nnnet.h"
#include "config/CF.h"

char* PROGRAM_NAME;


void print_config_format(){
  printf("config file format:\n");
  printf("         PNG file : PNG file or digit                    : floating point\n");
  printf("         (input)  : (train image or rcognition location) : (learning rate)\n");
  printf("NO EXTRA SPACES\n");
}

void print_help(){
  char* name = PROGRAM_NAME;
  printf("arguments: ./%s <mode>\nmode: learn:\n./%s learn <config file> <output file> <image width> <image height> <itterations>\nimage height and width need to be consistent accross all images specified in the file\n\nmode: run\n./%s run <model_name> <input image> <image width> <image height> (output file PNG, optional)\nif the output file is not specified raw image data will be sent to stdout\ninput image dimensions must match those used to train the neural network\n",
         name,name,name);

  printf("\nit is important that you pre process all the incoming images to have the same width and height\n");
  printf("after the training is done you will still have to process the input image to be of the same dimension as the training set\n");


}

int get_int_p(char* string){
  int size =strlen(string);
  char* tp = string;
  for(int i=0;i<size;i++){
    if(isdigit(*tp)==0)
      return -1;
    tp++;
  }
  return atoi(string);
}

float nn_compare(struct net_stack* stk, float* input, float* expected, int size){
  float check[size];
  nn_fwd(stk,input,check);

  float average_error=0;
  float error;
  for(int i = 0; i < size; i++){
    error = expected[i] - check[i];
    if(i==0)
      average_error = error;
    else 
      average_error = (average_error + error)/2;

  }

  return fabs(average_error);
}



void learn(char* infile, char* outfile,int width, int height,int itterations){
  printf("starting..\n");

  FILE* write_test = fopen(outfile,"wb");
  if(!write_test){
    printf("output file not writeable\n");
    printf("abort\n");
    return;
  }
  fclose(write_test);
  printf("alocating memory..\n");

  struct net_stack* nnR = setup_nn(width*height,ACTIVATION_RELU,3);
  struct net_stack* nnG = setup_nn(width*height,ACTIVATION_RELU,3);
  struct net_stack* nnB = setup_nn(width*height,ACTIVATION_RELU,3);

  float* R=malloc(sizeof(float)*(width*height));
  float* G=malloc(sizeof(float)*(width*height));
  float* B=malloc(sizeof(float)*(width*height));

  float* Re=malloc(sizeof(float)*(width*height));
  float* Ge=malloc(sizeof(float)*(width*height));
  float* Be=malloc(sizeof(float)*(width*height));


  printf("starting training..\n");
//put the following in a loop
for(int i = 0;i < itterations;i++){
  if(init_config(infile)==-1){
    printf("failed to open config file\n");
    free(R);free(G);free(B);
    free(Re);free(Ge);free(Be);
    nn_free(nnR);nn_free(nnG);nn_free(nnB);
    return;
  }
  struct config_line cline;
  while(read_config_line(&cline)!=-1){
    if(rescaled_read(cline.input_image,R,G,B,width,height)==-1){
      printf("failed to open image %s\n",cline.input_image);
      printf("abort\n");
      free_cfg_data(&cline);
      close_config();
      free(R);free(G);free(B);
      free(Re);free(Ge);free(Be);
      nn_free(nnR);nn_free(nnG);nn_free(nnB);
      return;
    }
    prepare_data(R,width*height);
    prepare_data(G,width*height);
    prepare_data(B,width*height);

    if(cline.type==TYPE_NN_IMAGE){
      if(rescaled_read(cline.output_image,Re,Ge,Be,width,height)==-1){
        printf("failed to open image %s\n",cline.output_image);
        printf("abort\n");
        free_cfg_data(&cline);
        close_config();
        free(R);free(G);free(B);
        free(Re);free(Ge);free(Be);
        nn_free(nnR);nn_free(nnG);nn_free(nnB);
        return;

      }
    prepare_data(Re,width*height);
    prepare_data(Ge,width*height);
    prepare_data(Be,width*height);

    }else{
      bzero(Re,sizeof(float)*(width*height));
      bzero(Ge,sizeof(float)*(width*height));
      bzero(Be,sizeof(float)*(width*height));
      Re[cline.position_recog]=1;
      Ge[cline.position_recog]=1;
      Be[cline.position_recog]=1;
    }

    nn_back_prop(nnR,R,Re,cline.weight);
    nn_back_prop(nnG,G,Ge,cline.weight);
    nn_back_prop(nnB,B,Be,cline.weight);

    


    float ravg=nn_compare(nnR, R , Re, width*height);
    float gavg=nn_compare(nnG, G , Ge, width*height);
    float bavg=nn_compare(nnB, B , Be, width*height);

    float error_average = (ravg + gavg + bavg)/3;
    printf("error for %s is %g\n",cline.input_image,error_average*100);
    free_cfg_data(&cline);
  }
  close_config();
  }

  free(Re);free(Ge);free(Be);
  free(R);free(G);free(B);

  char fstring[strlen(outfile)+2];
  sprintf(fstring,"%sR",outfile);
  nn_to_file(nnR,fstring);
  sprintf(fstring,"%sG",outfile);
  nn_to_file(nnG,fstring);
  sprintf(fstring,"%sB",outfile);
  nn_to_file(nnB,fstring);

  printf("learning complete data written to %s[R,G,B] \n",outfile);



  nn_free(nnR);nn_free(nnG);nn_free(nnB);
}


void run_test(char* net_name_path,char* input_image,char* output,int width,int height){
  struct net_stack* Rn;
  struct net_stack* Gn;
  struct net_stack* Bn;

  char* outfile=net_name_path;
  char fstring[strlen(outfile)+2];
  sprintf(fstring,"%sR",outfile);
  Rn=nn_from_file(fstring);
  sprintf(fstring,"%sG",outfile);
  Gn=nn_from_file(fstring);
  sprintf(fstring,"%sB",outfile);
  Bn=nn_from_file(fstring);

  if( Rn == NULL || Gn == NULL || Bn == NULL){
    printf("failed to load one or more of the following neural networks:\n");
    printf("%sR %sG %sB\n",outfile,outfile,outfile);
    nn_free(Rn);nn_free(Gn);nn_free(Bn);
    return;
  }

  float* R=malloc(sizeof(float)*(width*height));
  float* G=malloc(sizeof(float)*(width*height));
  float* B=malloc(sizeof(float)*(width*height));

  float* Re=malloc(sizeof(float)*(width*height));
  float* Ge=malloc(sizeof(float)*(width*height));
  float* Be=malloc(sizeof(float)*(width*height));

   if(rescaled_read(input_image,R,G,B,width,height)==-1){
        printf("failed to open image %s\n",input_image);
        printf("abort\n");
        free(R);free(G);free(B);
        free(Re);free(Ge);free(Be);
        nn_free(Rn);nn_free(Gn);nn_free(Bn);
        return;

      }

    prepare_data(R,width*height);
    prepare_data(G,width*height);
    prepare_data(B,width*height);

    nn_fwd(Rn,R,Re);
    nn_fwd(Gn,G,Ge);
    nn_fwd(Bn,B,Be);

    revert_data(Re,width*height);
    revert_data(Ge,width*height);
    revert_data(Be,width*height);

    if(output == NULL){
      float avg_max=0;
      int max_index=-1;
      float prev=0;
      for(int i = 0; i < width*height;i++){
        float avg=Re[i]+Be[i]+Ge[i];
        if(prev!=avg)
          printf("%g ",avg/3);
        prev=avg;
        if(avg > avg_max){
          avg_max=avg;
          max_index=i;
        }
      }
    
      printf("\nindex match: %d\n",max_index);
      printf("match value: %g\n",avg_max/3);

      free(R);free(G);free(B);
      free(Re);free(Ge);free(Be);
      nn_free(Rn);nn_free(Gn);nn_free(Bn);

      return;

    }

   if(image_write(output,Re,Ge,Be,width,height) ==-1){
      printf("warning, failed to write output image\n");
    }

   free(R);free(G);free(B);
   free(Re);free(Ge);free(Be);
   nn_free(Rn);nn_free(Gn);nn_free(Bn);


}


int main(int argn, char* argv[]){

  PROGRAM_NAME = argv[0];
  //arguments: ./ImageNet <mode> 
  //mode: learn:
  //./ImageNet learn <config file> <output file> <image width> <image height> <itterations>
  //image height and width need to be consistent accross all images specified in the file
  /*
   * mode: run
   * ./ImageNet run <model_name> <input image> <image width> <image height> (output file PNG, optional)
   * if the output file is not specified raw image data will be sent to stdout
   * input image dimensions must match those used to train the neural network
   */

  if(argn < 2){
    print_help(argv[0]);
    return 0;
  }

  if(strcmp(argv[1],"learn")!=0 && strcmp(argv[1],"run")!=0){
    print_help();
    printf("\nargument expects learn or run\n");
    return 0;
  }

  if(strcmp(argv[1],"learn")==0){
    if(argn != 7){
      print_help();
      print_config_format();
      printf("\nlearn expects 5 arguments two files and three integers\n");
      return 0;
    }
    int width = get_int_p(argv[4]);
    int height = get_int_p(argv[5]);
    int itter = get_int_p(argv[6]);
    if(width == -1 || height == -1 || itter == -1){
      print_help();
      printf("width and height and itterations must be an integer\n");
      return 0;
    }
    learn(argv[2],argv[3],width,height,itter);
    return 0;
  }
  char* outputv = NULL;
  if(argn != 6 && argn != 7){
      print_help();
      printf("\nrun expects 4 arguments two files and two integers \n");
      printf("\nit can also take a 5th argument for the output image file \n");
      return 0;
  }
  if(argn == 7){
    outputv = argv[6];
  }
  int width = get_int_p(argv[4]);
  int height = get_int_p(argv[5]);

  if(width == -1 || height == -1 ){
      print_help();
      printf("width and height must be an integer\n");
      return 0;
    }


  run_test(argv[2],argv[3],outputv,width,height);

  return 0;
}
