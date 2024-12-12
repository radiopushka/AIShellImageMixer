#include "nnnet.h"
#include <stdio.h>

void print_array(float* arr, int size){

  for(int i = 0; i < size ; i++){
    printf(" %g", arr[i]);
  }
  printf("\n");

}

int main(){


  

  struct net_stack* nst = setup_nn(4,ACTIVATION_RELU,3);

  float some_data[] = {0.5,1,0.5,1};
  float expected[] = {0,0,0,1};

  float some_data2[] = {1,0.5,1,0.5};
  float expected2[] = {1,0,1,0};

  float some_data3[] = {0.4,0.6,0.5,1};

  float out_d[4];

  struct net_stack* tmp_n = nn_from_file("network");

  nn_fwd(tmp_n,some_data,out_d);
  print_array(out_d,4);
  nn_free(tmp_n);

  //nn_dump(nst);
  printf("starting\n");
  int i;

  for(i=0; i < 400; i++){
    nn_back_prop(nst,some_data,expected,0.1);
    nn_back_prop(nst,some_data2,expected2,0.1);
 
  }
  //nn_dump(nst);
  printf("done learning\n");

  nn_fwd(nst,some_data,out_d);

  printf("\nset 1, expected 0 0 0 1\n");
  print_array(out_d,4);
  
  nn_fwd(nst,some_data2,out_d);

  printf("\nset 2, expected 1 0 1 0\n");
  print_array(out_d,4);
 
  nn_fwd(nst,some_data3,out_d);

  printf("\nset 3, expected 0 0 0 1\n");
  print_array(out_d,4);
 
  nn_to_file(nst,"network");

  nn_free(nst);

  return 0;
}
