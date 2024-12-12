#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "CF.h"

void string_builder(char** string, char c){
  if(*string == NULL){
    *string = malloc(sizeof(char)*2);
    (*string)[0]=c;
    (*string)[1]=0;
    return;
  }
  int ssize=strlen(*string);
  char* nstring = malloc(sizeof(char)*(ssize+2));
  memcpy(nstring, *string,ssize*sizeof(char));
  nstring[ssize] = c;
  nstring[ssize+1] = 0;
  free(*string);
  *string = nstring;

}
int read_line_in(FILE* in,char** path, char** ident, char** train){
  int rv=0;
  
  int walk =0;

  *path = NULL;
  *ident = NULL;
  *train = NULL;

  while(rv != EOF && rv != '\n'){
    rv=fgetc(in);
    
    if( rv != ':' && rv != '\n' && rv != EOF){
      switch(walk){
        case 0 : 
          string_builder(path,rv);
          break;
        case 1:
          string_builder(ident,rv);
          break;
        case 2:
          string_builder(train,rv);
          break;
      }
    }else{
      walk++;
    }
  }

  if(rv == EOF)
    return -1;

  return 1;
}

int is_real_number(char* string){
  int size = strlen(string);
  for(int i=0; i < size; i++){
    if(isdigit(*string) == 0){
      return -1;
    }
    string++;
  }
  return 1;
}

int is_float_number(char* string){
  int size = strlen(string);
  int dot_count = 0;
  for(int i=0; i < size; i++){
    if(isdigit(*string) == 0){
      if(*string == '.'){
        if(dot_count > 0){
          return -1;
        }
        dot_count++;
      }else{
        return -1;
      }
    }
    string++;
  }


  if(dot_count > 1)
    return -1;

  return 1;
}

FILE* global_cfg;

int init_config(char* file){

  FILE* f = fopen(file,"rb");

  if(!f){
    printf("file not found\n");

    printf("configuration file syntax:\n");

    printf("         PNG file : PNG file or digit  : floating point\n");
    printf("         input    : output or rcog loc : learning rate\n");
    printf("NO SPACES\n");
        
    return -1;
  }

  global_cfg = f;
  return 1;

}

int read_config_line(struct config_line* cl){
    char* ichi;
    char* ni;
    char* san;
    if(read_line_in(global_cfg,&ichi,&ni, &san) == -1){
      printf("read file or EOF, done\n");
      return -1;
    }
    if(ichi != NULL && ni != NULL && san != NULL){
      if(is_real_number(ni) == 1){
        cl -> type = TYPE_NN_NODE;
        cl -> position_recog = atoi(ni);
        cl -> output_image = NULL;
        free(ni);
        ni = NULL;
      }else{
        cl -> type = TYPE_NN_IMAGE;
        cl -> output_image = ni;
      }
      if(is_float_number(san) == -1){
        printf("%s %s %s\n",ichi,ni,san);
        printf("syntax error, third value must be a floating point\n");
        printf("expects: PNG file : PNG file or digit  : floating point\n");
        printf("         input    : output or rcog loc : learning rate\n");
        printf("NO EXTRA SPACES\n");
        free(ichi);
        free(ni);
        free(san);
        return -1;
      }
      //printf("%s %s %s\n",ichi,ni,san);
    }else{
      printf("configuration file syntax error\n");
      free(ichi);
      free(ni);
      free(san);
      return -1;

    }
    cl -> weight = atof(san);
    cl -> input_image = ichi;
  
    free(san);
  

  return 1;
}

void free_cfg_data(struct config_line* cl){
  free(cl -> input_image);
  free(cl -> output_image);
}

void close_config(){
  fclose(global_cfg);
}

/*int main(int argn,char* argv[]){

  if(init_config(argv[1]) == -1)
    return 0;

  
  struct config_line cline;
  while(read_config_line(&cline) != -1){

    printf("input: %s %s\n",cline.input_image,cline.output_image);
    printf("type: %d\n",cline.type);
    free_cfg_data(&cline);
  }

  close_config();

   return 0;
}*/
