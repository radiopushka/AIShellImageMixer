#ifndef CONFIGS
#define CONFIGS

#define TYPE_NN_NODE 0
#define TYPE_NN_IMAGE 1

struct config_line{
  int type;
  char* input_image;
  char* output_image;
  int position_recog;
  float weight;
};

int init_config(char* file);
//open the configuration file, returns -1 if failed and prints error to stdout

int read_config_line(struct config_line* cline);
//read a line from the configuration file

void free_cfg_data(struct config_line* cline);
//free the data from the memory, call after each read_config

void close_config();
//close the configuration file

#endif // !CONFIGS
