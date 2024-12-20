#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include <ctype.h>
#include<unistd.h>
#include <math.h>
#include "imageio.h"
#include "CAI/nnnet.h"
#include "config/CF.h"

char* PROGRAM_NAME;
int multiplier=1;


void print_config_format(){
  printf("config file format:\n");
  printf("         PNG file : PNG file or digit                    : floating point\n");
  printf("         (input)  : (train image or rcognition location) : (learning rate)\n");
  printf("NO EXTRA SPACES\n");
}

void print_help(){
  char* name = PROGRAM_NAME;
  printf("arguments: %s <mode>\nmode: learn:\n%s learn <config file> <output file> <image width> <image height> <itterations or min percent error>\nimage height and width need to be consistent accross all images specified in the file\n\nmode: run\n%s run <model_name> <input image> <image width> <image height> (output file PNG, optional)\nif the output file is not specified raw image data will be sent to stdout\ninput image dimensions must match those used to train the neural network\n",
         name,name,name);
  printf("%s learn <config file> <output file> <image width> <image height> <- to run untill all logical comparisons pass\n",name);
  printf("console run mode:\n");
  printf("%s runc <model name> <image width> <image height>\n",name);
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

float nn_compare(struct net_stack* stk, float* input, float* expected, int size,int* failed,int expected_l){
  float check[size];
  //nn_fwd(stk,input,check);
  get_last_values(stk,check);

  float average_error=0;
  float error;
  float max=0;
  int index=-1;
  for(int i = 0; i < size; i++){
    error = expected[i] - check[i];

    if(check[i]>max){
      index=i;
      max=check[i];
    }

    if(i==0)
      average_error = error;
    else 
      average_error = (average_error + error)/2;

  }
  if(index != expected_l){
    if(expected_l!=-1)
      *failed=1;
  }

  return fabs(average_error);
}


void learn(char* infile, char* outfile,int width, int height,int itterations,float stop_at){
  printf("starting..\n");


  FILE* write_test = fopen(outfile,"wb");
  if(!write_test){
    printf("output file not writeable\n");
    printf("abort\n");
    return;
  }
  fclose(write_test);
  printf("alocating memory..\n");

  struct net_stack* nn = setup_nn(width*height*multiplier,ACTIVATION_NIL,1);

  float* In=malloc(sizeof(float)*(width*height*multiplier));

  float* Out=malloc(sizeof(float)*(width*height*multiplier));



  printf("starting training..\n");
//put the following in a loop
  //
float max=0;



for(int i = 0;i < itterations || itterations == -1 ;i++){
  if(init_config(infile)==-1){
    printf("failed to open config file\n");
    free(In);
    free(Out);
    nn_free(nn);
    return;
  }
  struct config_line cline;
  int failed_logic=0;
  max=0;
  while(read_config_line(&cline)!=-1){
    if(rescaled_read(cline.input_image,In,width,height,multiplier)==-1){
      printf("failed to open image %s\n",cline.input_image);
      printf("abort\n");
      free_cfg_data(&cline);
      close_config();
      free(In);
      free(Out);
      nn_free(nn);

      return;
    }
    prepare_data(In,width*height);

    if(cline.type==TYPE_NN_IMAGE){
      if(rescaled_read(cline.output_image,Out,width,height,multiplier)==-1){
        printf("failed to open image %s\n",cline.output_image);
        printf("abort\n");
        free_cfg_data(&cline);
        close_config();
        free(In);
        free(Out);
        nn_free(nn);

        return;

      }
    prepare_data(Out,width*height*multiplier);

    }else{
      bzero(Out,sizeof(float)*(width*height*multiplier));
      printf("pos:%d\n",cline.position_recog);
      Out[cline.position_recog]=1;
    }




    nn_back_prop(nn,In,Out,cline.weight);


    int pass_in=cline.position_recog;
    if(cline.type==TYPE_NN_IMAGE){
        pass_in=-1;
      }

    float error_average=nn_compare(nn, In , Out, width*height*multiplier, &failed_logic ,pass_in);

    printf("error for %s is %g\n",cline.input_image,error_average*100);
    free_cfg_data(&cline);

    if(error_average*100 > max){
        max = error_average*100;
      }
           
  }
  if(itterations == -1 && stop_at < 0 && failed_logic==0 ){
    i=0;
    itterations = 0;
    printf("logic success\n");
  }
  if(max < stop_at && itterations ==-1){
      i=0;
      itterations = 0;
      }

  printf("itteration: %d\n",i);
  close_config();
  }

  free(In);
  free(Out);

  nn_to_file(nn,outfile);

  printf("learning complete data written to %s \n",outfile);
  

  nn_free(nn);
}

//for the image recognition console

struct net_mem{
  struct net_stack* nn;

  float* In;

  float* Out;

};

int load_test_memory(struct net_mem* mem_container,char* net_name_path,int width,int height){
  struct net_stack* nn;

  char* outfile=net_name_path;
  nn=nn_from_file(outfile);

  if( nn == NULL){
    printf("failed to load the neural network");
    nn_free(nn);
    return -1;
  }

  mem_container->In=malloc(sizeof(float)*(width*height*multiplier));

  mem_container->Out=malloc(sizeof(float)*(width*height*multiplier));

  mem_container->nn=nn;

  return 1;
}

void free_test_memory(struct net_mem* mem_container){
        free(mem_container->In);
        free(mem_container->Out);
        nn_free(mem_container->nn);

}

void mem_process_data(struct net_mem* mem,int width,int height){
    prepare_data(mem->In,width*height*multiplier);

    nn_fwd(mem->nn,mem->In,mem->Out);

    revert_data(mem->Out,width*height*multiplier);

}

void print_mem_data(struct net_mem* mem,int width,int height){
      float avg_max=0;
      int max_index=-1;
      float prev=0;
      for(int i = 0; i < width*height*multiplier;i++){
        float avg=mem->Out[i];
        if(prev!=avg)
          printf("%d:%g ",i,avg/3);
        prev=avg;
        if(avg > avg_max){
          avg_max=avg;
          max_index=i;
        }
      }
    
      printf("\nindex match: %d\n",max_index);
      printf("match value: %g\n",avg_max/3);

}



void run_test(char* net_name_path,char* input_image,char* output,int width,int height){

  struct net_mem memc;
  if(load_test_memory(&memc,net_name_path,width,height)==-1)
    return;

   if(rescaled_read(input_image,memc.In,width,height,multiplier)==-1){
        printf("failed to open image %s\n",input_image);
        printf("abort\n");
        free_test_memory(&memc);
        return;

      }

    mem_process_data(&memc,width,height); 
  

    if(output == NULL){
      
      print_mem_data(&memc,width,height);
        
      free_test_memory(&memc);
      

      return;

    }

   if(image_write(output,memc.Out,width,height,multiplier) ==-1){
      printf("warning, failed to write output image\n");
    }


      free_test_memory(&memc);

}

void get_command(char** st_array){
  char c;
  int pos=0;
  int pos_array=0;
  char prev = 0;
  char deli = ' ';
  while((c=getchar())!='\n'){
    if(pos>2)
      break;
    
    if(c == deli){
      if(prev != deli){
        
        pos++;
        pos_array=0;
      }
      if(deli == '\''){
        pos--;
        deli = ' ';
      }

    }else{
      if(c == '\''){
        deli = '\'';
      }else{

        if(pos_array<1024){
          st_array[pos][pos_array]=c;
          pos_array++;
        }
      }
    }
    prev = c;
  }
}

void run_console(char* net_name_path,int width,int height){

  struct net_mem memc;
  if(load_test_memory(&memc,net_name_path,width,height)==-1)
    return;

  printf("commands: \n");
  printf("exit\n");
  printf("identify 'image path'\n");
  printf("inout 'input image path' 'output image path'\n");
  printf("\n");
  
  char** command_s;
  command_s=malloc(sizeof(char*)*3);
  command_s[0]=malloc(sizeof(char)*1024);
  command_s[1]=malloc(sizeof(char)*1024);
  command_s[2]=malloc(sizeof(char)*1024);

  while(1){

   printf("> ");
    
   bzero(command_s[0],sizeof(char)*1024);
   bzero(command_s[1],sizeof(char)*1024);
   bzero(command_s[2],sizeof(char)*1024);
   get_command(command_s);

   printf("%s %s %s\n",command_s[0],command_s[1],command_s[2]);

   if((strcmp(command_s[0],"identify")==0 || strcmp(command_s[0],"inout")==0) && command_s[1][0]!=0){


      if(rescaled_read(command_s[1],memc.In,width,height,multiplier)==-1){
        printf("failed to open image %s\n",command_s[1]);

      }else{
        mem_process_data(&memc,width,height); 
        if(strcmp(command_s[0],"identify")==0){
          print_mem_data(&memc,width,height);
        }else{
          if(command_s[2][0]!=0){
             if(image_write(command_s[2],memc.Out,width,height,multiplier) ==-1){
              printf("warning, failed to write output image\n");
            }
          }else{
            printf("expects 2 arguments\n");
          }

        }
      }

   }else{

      if(strcmp(command_s[0],"exit")==0)
        break;

      printf("commands: \n");

      printf("exit\n");
      printf("identify 'image path'\n");
      printf("inout 'input image path' 'output image path'\n");
      printf("\n");

      printf("command not known\n");
  }
  }
  
      free(command_s[0]);
      free(command_s[1]);
      free(command_s[2]);
      free(command_s);

      free_test_memory(&memc);

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

  if(strcmp(argv[1],"learn")!=0 && strcmp(argv[1],"run")!=0 && strcmp(argv[1],"runc")!=0){
    print_help();
    printf("\nargument expects learn or run or runc\n");
    return 0;
  }

  if(strcmp(argv[1],"learn")==0){
    if(argn != 7 && argn != 6){
      print_help();
      print_config_format();
      printf("\nlearn expects 5 arguments two files and three integers\n");
      printf("\nlearn also expects 4 arguments two files and two integers\n");
      return 0;
    }
    int width = get_int_p(argv[4]);
    int height = get_int_p(argv[5]);
  
    if(width == -1 || height == -1 ){
      print_help();
      printf("width and height and itterations must be an integer\n");
      return 0;
    }

    if(argn == 6){
      printf("running until all logical comparisons are correct\n");
      learn(argv[2],argv[3],width,height,-1,-1);
      return 0;

    }
    int itter = get_int_p(argv[6]);
    if(itter != -1){
      printf("runnning for %d trials\n",itter);
      printf("add a decimal point to run untill percent error is lower than...\n");
      learn(argv[2],argv[3],width,height,itter,0);
      return 0;
    }

    if(is_float_number(argv[6])!=-1){
      printf("running until percent error is under %g\n",atof(argv[6]));
      learn(argv[2],argv[3],width,height,-1,atof(argv[6]));
      return 0;
    }

  } 

if(strcmp(argv[1],"runc")==0){
      if( argn != 5){
        print_help();
        printf("\nrun expects 3 arguments one file and two integers \n");
        return 0;
      }
      int width = get_int_p(argv[3]);
      int height = get_int_p(argv[4]);


      if(width == -1 || height == -1 ){
        print_help();
        printf("width and height must be an integer\n");
        return 0;
      }

      run_console(argv[2],width,height);
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
