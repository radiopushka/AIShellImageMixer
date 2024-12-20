
#ifndef NN_NET
#define NN_NET
#include "network.h"

struct net_stack{
  struct nn* contained;
  struct net_stack* next;
};

//io port size is consistent, all inputs and outputs must be of this size
//acivation is the activation function type
//depth is for deep learning how many sub layers
//creates the neural network
struct net_stack* setup_nn(int io_port_size,int activation,int depth);

//performs back propagation, 
//input is the input data array, output is the start pointer to which the output is written
void nn_fwd(struct net_stack* nst,float* input, float* output);
//peforms back propagation on the network. both input and expected are inputs where the network is trained for input make me expected
//learn rate is how heavy to drive the net
void nn_back_prop(struct net_stack* nst, float* input, float* expected, float learn_rate);

//debug print
void nn_dump(struct net_stack* nst);
//get the network size, depth
int nn_size_d(struct net_stack* nst);
//free the memory with the network
void nn_free(struct net_stack* nst);

//get the values from the last forward propagation
void get_last_values(struct net_stack* nst, float* output);


//file IO
int nn_to_file(struct net_stack* nst,char* file);

struct net_stack* nn_from_file(char* file);
#endif // !NN_NET
