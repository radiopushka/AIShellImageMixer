#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include <pthread.h>
#include <ctype.h>
#include<unistd.h>
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
  printf("arguments: %s <mode>\nmode: learn:\n%s learn <config file> <output file> <image width> <image height> <itterations or min percent error>\nimage height and width need to be consistent accross all images specified in the file\n\nmode: run\n%s run <model_name> <input image> <image width> <image height> (output file PNG, optional)\nif the output file is not specified raw image data will be sent to stdout\ninput image dimensions must match those used to train the neural network\n",
         name,name,name);
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

//multi threaded learning

int count=0;

float cur_weight=0;

struct net_stack* nnT1;
struct net_stack* nnT2;

float* T1in;
float* T2in;

float* T1out;
float* T2out;

int exit_s=0;

pthread_mutex_t tm1;
pthread_mutex_t tm1_h;

pthread_mutex_t tm2;
pthread_mutex_t tm2_h;

void *thread1(void* unused){
  while(exit_s!=1){
  pthread_mutex_lock(&tm1);

  if(exit_s != 1) 
    nn_back_prop(nnT1,T1in,T1out,cur_weight);

  count--;
  pthread_mutex_unlock(&tm1);
  pthread_mutex_lock(&tm1_h);
  pthread_mutex_unlock(&tm1_h);
  }
  return NULL;
}

void *thread2(void* unused){
  while(exit_s!=1){
  pthread_mutex_lock(&tm2);
  
  if(exit_s != 1)
    nn_back_prop(nnT2,T2in,T2out,cur_weight);

  count--;
  pthread_mutex_unlock(&tm2);
  pthread_mutex_lock(&tm2_h);
  pthread_mutex_unlock(&tm2_h);
  }
  return NULL;
}


void learn(char* infile, char* outfile,int width, int height,int itterations,float stop_at){
  printf("starting..\n");

  pthread_t t1;
  pthread_t t2;

  FILE* write_test = fopen(outfile,"wb");
  if(!write_test){
    printf("output file not writeable\n");
    printf("abort\n");
    return;
  }
  fclose(write_test);
  printf("alocating memory..\n");

  struct net_stack* nnR = setup_nn(width*height,ACTIVATION_RELU,1);
  struct net_stack* nnG = setup_nn(width*height,ACTIVATION_RELU,1);
  struct net_stack* nnB = setup_nn(width*height,ACTIVATION_RELU,1);

  float* R=malloc(sizeof(float)*(width*height));
  float* G=malloc(sizeof(float)*(width*height));
  float* B=malloc(sizeof(float)*(width*height));

  float* Re=malloc(sizeof(float)*(width*height));
  float* Ge=malloc(sizeof(float)*(width*height));
  float* Be=malloc(sizeof(float)*(width*height));

  T1in = G;
  T2in = B;

  T1out = Ge;
  T2out = Be;

  nnT1 = nnG;
  nnT2 = nnB;


  printf("starting training..\n");
//put the following in a loop
  //
float max=0;

pthread_mutex_init(&tm1,NULL);
pthread_mutex_init(&tm1_h,NULL);
pthread_mutex_init(&tm2,NULL);
pthread_mutex_init(&tm2_h,NULL);
  
pthread_mutex_lock(&tm1);
pthread_mutex_lock(&tm2);

exit_s=0;

pthread_create(&t1,NULL,thread1,NULL);
pthread_create(&t2,NULL,thread2,NULL);

for(int i = 0;i < itterations || itterations == -1 ;i++){
  if(init_config(infile)==-1){
    printf("failed to open config file\n");
    free(R);free(G);free(B);
    free(Re);free(Ge);free(Be);
    nn_free(nnR);nn_free(nnG);nn_free(nnB);
    pthread_mutex_destroy(&tm1);pthread_mutex_destroy(&tm2);
    pthread_mutex_destroy(&tm1_h);pthread_mutex_destroy(&tm2_h);
    return;
  }
  struct config_line cline;
  max=0;
  while(read_config_line(&cline)!=-1){
    if(rescaled_read(cline.input_image,R,G,B,width,height)==-1){
      printf("failed to open image %s\n",cline.input_image);
      printf("abort\n");
      free_cfg_data(&cline);
      close_config();
      free(R);free(G);free(B);
      free(Re);free(Ge);free(Be);
      nn_free(nnR);nn_free(nnG);nn_free(nnB);
      pthread_mutex_destroy(&tm1);pthread_mutex_destroy(&tm2);
      pthread_mutex_destroy(&tm1_h);pthread_mutex_destroy(&tm2_h);
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
        pthread_mutex_destroy(&tm1);pthread_mutex_destroy(&tm2);
        pthread_mutex_destroy(&tm1_h);pthread_mutex_destroy(&tm2_h);
        return;

      }
    prepare_data(Re,width*height);
    prepare_data(Ge,width*height);
    prepare_data(Be,width*height);

    }else{
      bzero(Re,sizeof(float)*(width*height));
      bzero(Ge,sizeof(float)*(width*height));
      bzero(Be,sizeof(float)*(width*height));
      printf("pos:%d\n",cline.position_recog);
      Re[cline.position_recog]=1;
      Ge[cline.position_recog]=1;
      Be[cline.position_recog]=1;
    }


    cur_weight = cline.weight;

    count = 2;

    pthread_mutex_lock(&tm1_h);
    pthread_mutex_lock(&tm2_h);
    pthread_mutex_unlock(&tm1);
    pthread_mutex_unlock(&tm2);
    nn_back_prop(nnR,R,Re,cline.weight);
    //nn_back_prop(nnG,G,Ge,cline.weight);
    //nn_back_prop(nnB,B,Be,cline.weight);


    while(count!=0);
    pthread_mutex_lock(&tm1);
    pthread_mutex_lock(&tm2);
    pthread_mutex_unlock(&tm1_h);
    pthread_mutex_unlock(&tm2_h);



    float ravg=nn_compare(nnR, R , Re, width*height);
    float gavg=nn_compare(nnG, G , Ge, width*height);
    float bavg=nn_compare(nnB, B , Be, width*height);

    float error_average = (ravg + gavg + bavg)/3;
    printf("error for %s is %g\n",cline.input_image,error_average*100);
    free_cfg_data(&cline);

    if(error_average*100 > max){
        max = error_average*100;
      }
           
  }
  if(max < stop_at && itterations ==-1){
      i=0;
      itterations = 0;
      }

  printf("itteration: %d\n",i);
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
  
  exit_s=1;
  count = 2;
  pthread_mutex_lock(&tm1_h);
  pthread_mutex_lock(&tm2_h);
  pthread_mutex_unlock(&tm1);
  pthread_mutex_unlock(&tm2);

  while(count!=0);


  pthread_mutex_lock(&tm1);
  pthread_mutex_lock(&tm2);
  pthread_mutex_unlock(&tm1_h);
  pthread_mutex_unlock(&tm2_h);


  pthread_join(t1,NULL);
  pthread_join(t2,NULL);

  pthread_mutex_destroy(&tm1);pthread_mutex_destroy(&tm2);
  pthread_mutex_destroy(&tm1_h);pthread_mutex_destroy(&tm2_h);


  nn_free(nnR);nn_free(nnG);nn_free(nnB);
}

//for the image recognition console

struct net_mem{
  struct net_stack* Rn;
  struct net_stack* Gn;
  struct net_stack* Bn;

  float* R;
  float* G;
  float* B;

  float* Re;
  float* Ge;
  float* Be;

};

int load_test_memory(struct net_mem* mem_container,char* net_name_path,int width,int height){
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
    return -1;
  }

  mem_container->R=malloc(sizeof(float)*(width*height));
  mem_container->G=malloc(sizeof(float)*(width*height));
  mem_container->B=malloc(sizeof(float)*(width*height));

  mem_container->Re=malloc(sizeof(float)*(width*height));
  mem_container->Ge=malloc(sizeof(float)*(width*height));
  mem_container->Be=malloc(sizeof(float)*(width*height));

  mem_container->Rn=Rn;
  mem_container->Gn=Gn;
  mem_container->Bn=Bn;

  return 1;
}

void free_test_memory(struct net_mem* mem_container){
        free(mem_container->R);free(mem_container->G);free(mem_container->B);
        free(mem_container->Re);free(mem_container->Ge);free(mem_container->Be);
        nn_free(mem_container->Rn);nn_free(mem_container->Gn);nn_free(mem_container->Bn);

}

void mem_process_data(struct net_mem* mem,int width,int height){
    prepare_data(mem->R,width*height);
    prepare_data(mem->G,width*height);
    prepare_data(mem->B,width*height);

    nn_fwd(mem->Rn,mem->R,mem->Re);
    nn_fwd(mem->Gn,mem->G,mem->Ge);
    nn_fwd(mem->Bn,mem->B,mem->Be);

    revert_data(mem->Re,width*height);
    revert_data(mem->Ge,width*height);
    revert_data(mem->Be,width*height);

}

void print_mem_data(struct net_mem* mem,int width,int height){
      float avg_max=0;
      int max_index=-1;
      float prev=0;
      for(int i = 0; i < width*height;i++){
        float avg=mem->Re[i]+mem->Be[i]+mem->Ge[i];
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

   if(rescaled_read(input_image,memc.R,memc.G,memc.B,width,height)==-1){
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

   if(image_write(output,memc.Re,memc.Ge,memc.Be,width,height) ==-1){
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


      if(rescaled_read(command_s[1],memc.R,memc.G,memc.B,width,height)==-1){
        printf("failed to open image %s\n",command_s[1]);

      }else{
        mem_process_data(&memc,width,height); 
        if(strcmp(command_s[0],"identify")==0){
          print_mem_data(&memc,width,height);
        }else{
          if(command_s[2][0]!=0){
             if(image_write(command_s[2],memc.Re,memc.Ge,memc.Be,width,height) ==-1){
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
    if(argn != 7){
      print_help();
      print_config_format();
      printf("\nlearn expects 5 arguments two files and three integers\n");
      return 0;
    }
    int width = get_int_p(argv[4]);
    int height = get_int_p(argv[5]);

    if(is_float_number(argv[6])!=-1){

      learn(argv[2],argv[3],width,height,-1,atof(argv[6]));
      return 0;
    }

    int itter = get_int_p(argv[6]);
    if(width == -1 || height == -1 || itter == -1){
      print_help();
      printf("width and height and itterations must be an integer\n");
      return 0;
    }
    learn(argv[2],argv[3],width,height,itter,0);
    return 0;
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
