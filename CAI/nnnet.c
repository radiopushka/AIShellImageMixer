#include "nnnet.h"
#include <stdlib.h>

struct net_stack* setup_nn(int io_port_size,int activation,int depth){

  struct net_stack* top;

  if(depth < 1)
    return NULL;

  struct net_stack* exit = malloc(sizeof(struct net_stack));

  exit -> next=NULL;
  
  exit -> contained = make_network(io_port_size,activation);

  top = exit;

  for(int i = 0; i < depth; i++){

    struct net_stack* element = malloc(sizeof(struct net_stack));
    element -> next = top;
    top = element;
    element -> contained = make_network(io_port_size,activation);
  }


  return top;
}

void nn_fwd(struct net_stack* nst,float* input, float* output){

  propagate(input,nst -> contained);
  struct net_stack* prev = nst;
  nst = nst -> next;
  while(nst != NULL){
    activate(prev -> contained);
    propagate(prev -> contained -> outputs , nst -> contained);
    prev = nst;
    nst = nst -> next;

  }
  int osize = prev -> contained -> output_size;
  float* output_array = prev -> contained -> outputs;
  float* output_end = prev -> contained -> outputs + osize;

  while(output_array < output_end){
    *output = *output_array;
    output++;
    output_array++;
  }
  
}

void nn_back_prop(struct net_stack* nst, float* input, float* expected, float learn_rate){
  propagate(input,nst -> contained);
  struct net_stack* prev = nst;
  struct net_stack* top = nst;
  nst = nst -> next;
  int size=0;
  while(nst != NULL){
    activate(prev -> contained);
    propagate(prev -> contained -> outputs , nst -> contained);
    prev = nst;
    nst = nst -> next;
    size++;
  }
  
  back_propagation_tail(prev -> contained, expected);

  nst = top;
  struct net_stack* stackoff[size];
  size = 0;
  while(nst != NULL){
    stackoff[size] = nst;
    nst = nst -> next;
    size++;
  }

  for(int i = size - 1; i > 0; i--){
    back_propagation_middle(stackoff[i-1] -> contained,stackoff[i] -> contained,learn_rate);
  }
  back_propagation_head(input, stackoff[0] -> contained,learn_rate);


}
int nn_size_d(struct net_stack* nst){
  int size = 0;
  while(nst != NULL){
    nst = nst -> next;
    size++;
  }
  return size;
}

void nn_free(struct net_stack* nst){
  if(nst == NULL)
    return;

  struct net_stack* nstk;
  while(nst != NULL){
    struct nn* th = nst -> contained;
    free_network(&th);
    nstk = nst -> next;
    free(nst);
    nst = nstk;
  }

}

void nn_dump(struct net_stack* nst){

 while(nst != NULL){
    struct nn* th = nst -> contained;
    print_net(th);
    nst = nst -> next;
  }

}

int nn_to_file(struct net_stack* nst,char* file){
 FILE* f = fopen(file, "wb"); 
  if(!f)
    return -1;

  int size = nn_size_d(nst);
  fwrite(&size, sizeof(int),1,f);


  struct net_stack* stackoff[size];
  size = 0;
  while(nst != NULL){
    stackoff[size] = nst;
    nst = nst -> next;
    size++;
  }

  for( int i = size - 1; i >= 0; i--){
    n_to_file_stream(stackoff[i] -> contained,f);
  }
  fclose(f);
  return 1;
}

struct net_stack* nn_from_file(char* file){

 FILE* f = fopen(file, "rb"); 

  if(!f)
    return NULL;

  int size;
  if(fread(&size,sizeof(int),1,f)!=1)
    printf("warning possible file corruption\n");

  struct net_stack* top = NULL;

  for(int i = 0; i < size; i++){

    struct net_stack* element = malloc(sizeof(struct net_stack));
    element -> next = top;
    top = element;
    element -> contained = n_from_file_stream(f);
  }

  fclose(f);
  return top;
}
