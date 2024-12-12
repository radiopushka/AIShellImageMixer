#ifndef NNNETWORK
#define NNNETWORK

#include<stdio.h>

#define ACTIVATION_TAN 0
#define ACTIVATION_SIN 1
#define ACTIVATION_COS 2
#define ACTIVATION_SIG 3
#define ACTIVATION_RELU 4


struct nn{
  int output_size;
  float* outputs;
  float* weights;
  int activation_type;
  //needed for back propagation
  float* errors;
};
//use with caution on embedded devices:
// malloc is present, do not make the net too large
//activation_type:
//0 - atan
//1 - sin
//2 - cos
//3 - sigmoid
//4 - ReLU
//structure for holding the neurons
void propogate(float* inputs,struct nn* network);
struct nn* make_network(int size, int activation_type);
void free_network(struct nn** net);
void print_net(struct nn* net); // prints the network to stdout

//base functions:
void propagate(float* inputs,struct nn* network);//forward prop
void activate(struct nn* network); // activation function
                                   //need to call this and propagate before calling the functions following


void back_propagation_tail(struct nn* network, float* outputs);// back prop to front, at the end when comparing
void back_propagation_middle(struct nn* before, struct nn* after, float learn_rate);// to be called in between two nets
void back_propagation_head(float* in,struct nn* after, float learn_rate); // to be called between input and first net


//write the network to a file
void n_to_file_stream(struct nn* network,FILE* f);

struct nn* n_from_file_stream(FILE* f);
#endif // !NNNETWORK
