void amplify_image(float* input,int size){

  float average=0;
  for(int i = 0;i<size;i++){
    if(i==0)
      average=input[i];
    else 
      average=(average+input[i])/2;
  }

  //now convert the image into signal on the average

 float max_deviation=0;
 for(int i = 0;i<size;i++){
    input[i]=input[i]-average;
    if(input[i]>max_deviation)
      max_deviation=input[i];
  }

  // good deviation is 127
  
  float multiplier=127/max_deviation;

//apply and rescale
 for(int i = 0;i<size;i++){
    input[i]=(input[i]*multiplier)+127;

  }


}
