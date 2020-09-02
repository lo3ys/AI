import java.lang.Math;

class Perceptron{
 
  float[] input;
  
  //output with activation function
  float output;
  
  //output without activation function (to make your own)
  float out;
  float[] weight;
  float lr= 0.0001;
  
  
  Perceptron(int inputs){
    input = new float[inputs];
    
    //inputs+1 for the biais's weight
    weight = new float[inputs+1];
    
    
    for(int i = 0; i < inputs; i++){
      //0.00000152 is just a random value close to 0 that has little chance of being used
      weight[i] = 0.00000152f;
    }
  }
  
  float sum(){
    //biais 
    float s = weight[weight.length-1];
    
    //inputs
    for(int i = 0; i < input.length; i++){
      s += weight[i]*input[i];
    }
    return(s);
  }
  
  void randomize(){
    for(int i = 0; i < weight.length;i++){
      //assign a random value between -1 and 1
      weight[i] = (float)Math.random()*2-1;
      
      //security to avoid soft lock
      if(weight[i] == 0){
          weight[i] = 0.00000152f;
      }
    }
  }
  
  void calculate_output(float[] I){
      if(I.length != input.length){
         Error("Inputs numbers doesn't match with the perceptron structure","output calculation");
      }else{
        //mapping inputs
         try{
           for(int i = 0; i < input.length; i++){
             input[i] = I[i];
           }
         }catch(Exception e){
           Error("Unknow error in input assignment ( "+e+" )","output calculation");
         }
         
         //calculate output
         try{
            out = sum();
            
            //activation function : signum
            if(out > 0){
               output = 1; 
            }else if(out==0){
               output = 0; 
            }else{
               output = -1; 
            }
         }catch(Exception e){
           Error("Unknow error in output calculation","output calculation");
         }
      }
  }
  
  void train(float[] I,float T){
       
      //calcul
      calculate_output(I);
      
      //learn
      float error = T-output;
      for(int i = 0; i < input.length; i++){
          weight[i] += error*input[i]*lr;
      }
      //biais
      weight[weight.length-1] += error*lr;
      
      //security to avoid soft lock
      for(int i = 0; i < weight.length; i++){
        if(weight[i] == 0){
          weight[i] = 0.00000152;
        }
      }
  }
  
  //for people who have made their own activation function
  void train_only(float error){
      for(int i = 0; i < input.length; i++){
          weight[i] += error*input[i]*lr;
      }
      //biais
      weight[weight.length-1] += error*lr;
      
      //security to avoid soft lock
      for(int i = 0; i < weight.length; i++){
        if(weight[i] == 0){
          weight[i] = 0.00000152;
        }
      }
  }
  
  void Error(String msg,String pos){
    System.out.println("=====___PERCEPTRON___=====");
    System.out.println("An error occurs at "+pos+" :");
    System.out.println("");
    System.out.println(msg);
    System.out.println("");
    System.out.println("==========================");
  }
  
}