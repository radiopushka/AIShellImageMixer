#include "network.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define EULER 2.7182818284
#define PI 3.1415926535898

//basic functions
//
//
struct nn* make_network(int size,int activation_type){
  struct nn* network = malloc(sizeof(struct nn));
  network -> output_size = size;
  network -> activation_type = activation_type;
  int i;
  // allocate memory

  float* output_buff = malloc(sizeof(float)*size);
  for(i=0;i<size;i++){
    output_buff[i]=0;
  }

  float* weight_buff = malloc((sizeof(float)*size)*size);
  float cweight = 1.0/size;
  for(i=0;i<size*size;i++){
    weight_buff[i]=cweight;
  }
  network -> outputs = output_buff;
  network -> weights = weight_buff;
  network -> errors = malloc(sizeof(float)*size);
  return network;
}

void free_network(struct nn** network){
  if(*network == NULL)
    return;

  free((*network) -> outputs);
  free((*network) -> weights);
  free((*network) -> errors);
  free(*network);
  *network = NULL;
}

void print_net(struct nn* network){

  float* optr = network -> outputs;
  float* optr2 = optr;
  float* oend = network -> outputs + network -> output_size;
  float* weights  = network -> weights;


  while(optr2 < oend){
    printf(" %g(",*optr);
    while(optr < oend){
      printf(" %g", *weights);
      weights++;
      optr++;
    }
    printf(")");
    optr2++;
    optr = network -> outputs;
  }
  printf("\n\n");
}

//forware propagation
void propagate(float* inputs,struct nn* network){
  float* iptr;
  float* wpointer = network -> weights;
  
  float* iptr_end = inputs + network -> output_size;

  float* optr = network -> outputs;
  float* ostart = optr;
  float* oend = network -> outputs + network -> output_size;

  while(optr < oend){
    *optr = 0;
    optr++;
  }

  float load;

  for(iptr=inputs;iptr < iptr_end;iptr++){
    load = *iptr;
    optr = ostart;
    for(optr=ostart;optr < oend;optr++){
      *optr = *optr + (load * (*wpointer));
      wpointer++;
    }
  }
}
//activation function
void activate(struct nn* network){
  int type = network -> activation_type;

  if(type == ACTIVATION_NIL)
    return;

  float size = network -> output_size;
  float* optr = network -> outputs;
  float* oend = network -> outputs + network -> output_size;

  while(optr < oend){
    switch(type){
      case ACTIVATION_TAN:
        *optr = 0.5+(atan(*optr))/PI;
        break;
      case ACTIVATION_LIN:
          *optr = (*optr)/size;
          if(*optr>1)
            *optr=1;
          if(*optr<0)
            *optr=0;
        break;
       case ACTIVATION_SIG:
        *optr = 1/(1+pow(EULER,-(*optr)));
        break;
    }
    optr++;
  }

}
//this is for when you need to match it with the output data
void back_propagation_tail(struct nn* network, float* outputs){

  float* optr = network -> outputs;
  float* oend = network -> outputs + network -> output_size;
  float* oshibki = network -> errors;
  
  float v_z;

  while(optr < oend){
  
    v_z = *optr;
    //*oshibki = (v_z) * (1 - v_z) * (*outputs - v_z);
    *oshibki = (*outputs - v_z);
      
    outputs++;
    optr++;
    oshibki++;
  }


} 
// the first argument is the neural network that is before in forward propagation
void back_propagation_middle(struct nn* before, struct nn* after, float learn_rate){
  float* vhod = before -> outputs;
  float* vkonets = before -> outputs + before -> output_size;

  float* weights = after -> weights;

  float* errorsw = after -> errors;
  float* eswst = errorsw;

  float* errorsi = before -> errors;
  float* estart = errorsi;

  float* error_end = errorsi + before -> output_size;

  float etmp;
  float change;


  while(errorsi < error_end){
    *errorsi = 0;
    errorsi++;
  }

  while(vhod < vkonets){
    errorsi = estart;

    while(errorsi < error_end){
      change = *vhod;
      //etmp = (*vhod) * (1 - *vhod) * ((*weights) * (*errorsw));
      etmp = (*weights) * (*errorsw);
      errorsw++;
      *errorsi = *errorsi + etmp;

      change = learn_rate * etmp * change;

      errorsi++;
      *weights = change + (*weights);

      weights++;
    }

    errorsw = eswst;
    vhod++;
  }



}

void back_propagation_head(float* in,struct nn* after,float learn_rate){
  float* vhod = in;
  float* vkonets = in + after -> output_size;

  float* weights = after -> weights;

  float* errorsw = after -> errors;
  float* eswst = errorsw;


  float* error_end = errorsw + after -> output_size;

  float etmp;
  float change;


  while(vhod < vkonets){

    while(errorsw < error_end){
      change = *vhod;

      etmp = (*weights) * (*errorsw);

      errorsw++;

      change = learn_rate * etmp * change;

      *weights = change + (*weights);

      weights++;
    }
    errorsw = eswst;
    vhod++;
  }

}
void n_to_file_stream(struct nn* network,FILE* f){

  int nsize = network -> output_size;
  int activation = network -> activation_type;

  fwrite(&nsize,sizeof(int),1,f);
  fwrite(&activation,sizeof(int),1,f);
  fwrite(network -> outputs,sizeof(float),nsize,f);
  fwrite(network -> weights,sizeof(float),nsize*nsize,f);
}

struct nn* n_from_file_stream(FILE* f){
  int size;
  int activation;
  int result=0;

  result=fread(&size,sizeof(int),1,f);
  if(result != 1)
    printf("warning possible file corruption\n");

  result=fread(&activation,sizeof(int),1,f);
  if(result != 1)
    printf("warning possible file corruption\n");

  struct nn* network = malloc(sizeof(struct nn));

  network -> output_size = size;
  network -> activation_type = activation;

  network -> errors = malloc(sizeof(float)*size);
  network -> outputs = malloc(sizeof(float)*size);
  network -> weights = malloc(sizeof(float)*(size*size));

  result=fread(network -> outputs, sizeof(float), size,f);
  if(result != size)
    printf("warning possible file corruption\n");

  result=fread(network -> weights, sizeof(float), size*size,f);
  if(result != size*size)
    printf("warning possible file corruption\n");

  return network;
}
