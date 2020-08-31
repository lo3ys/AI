import java.lang.*;
import java.awt.*;

public class neural_network{
  
  matrix[] biais;
  matrix[] memory;
  matrix[] value;
  float act_fun_smooth = -2;
  float learning_rate = 0.01;
  int resultat;
  
  neural_network(int input, int output, int[] hidden){
    
    // compacting layers
    int[] layer = new int[hidden.length+2];
    for(int i = 1; i < hidden.length+1; i++){
      
      layer[i] = hidden[i-1];
      
    }
    layer[layer.length-1] = output;
    layer[0] = input;
    
    //init value
    int loop_value = 0;
    int loop_memory = -1;
    memory = new matrix[layer.length-1];
    biais = new matrix[layer.length-1];
    value = new matrix[layer.length];
    
    //create matrix
    for(int v: layer){
      if(loop_memory >= 0){
        
        
        memory[loop_memory] = new matrix(layer[loop_memory],v);
        biais[loop_memory] = new matrix(1,v);
        
      }
      value[loop_value] = new matrix(1,v);
      
      loop_value ++;
      loop_memory ++;
    }
    
  }
  
  void set_FeedForward(float smooth, float lr){
      
    
    act_fun_smooth = smooth;
    learning_rate = lr;
    
  }
  
  
  void train(float[] input,int t){
   
      nn.calculate(input);
      float[] debug = new float[10];
      debug[t] = 1;
      matrix target = new matrix(1,10);
      target.set_value_line(debug);
      nn.back_propagation(target);
    
  }
  
  void reset(){
     for(int i = 0 ; i < memory.length; i++){
       
        
        memory[i].randomize((float)(-1),(float)(1));
        biais[i].randomize((float)(-1),(float)(1));
        
     }
  }
  

  
  void calculate(float[] input){
    
    
      
      value[0].set_value_line(input);
    
    for(int i = 1; i < value.length; i++){
      
        
        matrix sum = multiply(value[i-1],memory[i-1]);
        sum = plus(sum, biais[i-1]);
        sum.activation_function(act_fun_smooth);
        value[i] = sum;
        
      
    }
    
    
    value[value.length-1].show();
   
  }
  
  
  
  void calculate_grid(Grid grid){
    
    float input[] = new float[grid.w*grid.h];
    for(int i = 0; i < grid.data.length; i++){
      for(int j = 0; j < grid.data[i].length; j++){
        input[i*grid.data[i].length+j] = grid.data[i][j];
      }
    }
   value[0].set_value_line(input);
    
    for(int i = 1; i < value.length; i++){
      
        
        matrix sum = multiply(value[i-1],memory[i-1]);
        sum = plus(sum, biais[i-1]);
        sum.activation_function(act_fun_smooth);
        value[i] = sum;
        
      
    }
    
    value[value.length-1].show();
    resultat = predict();
  }
  
  void back_propagation(matrix target){
    
    matrix error = substract(target , value[value.length-1]);
    //error = fusion(error,error);
    
    for(int i = 0; i < memory.length ; i++){
      
      
      
      matrix Wt = new matrix(0,0);
      Wt.calc_to(memory[memory.length-1-i]);
      
     
      Wt.transpose();
      
      
     
      //error.transpose();
      
      matrix gradient = new matrix(0,0);
      gradient.calc_to(value[value.length-1-i]);
      gradient.derivative_function(0);
      
      gradient = fusion(gradient,error);
      gradient = multi(gradient,learning_rate);
      
      
      matrix Lt = new matrix(0,0);
      Lt.calc_to(value[value.length-2-i]);
      Lt.transpose();
      
      
      
      
      matrix delta = new matrix(0,0);
      delta = multiply(Lt , gradient);
      
      
      //delta.transpose();
      //System.out.println(error.x+" "+error.y);
      //gradient.transpose();
      //delta.transpose();
      memory[memory.length-1-i] = plus(memory[memory.length-1-i] , delta);
      
      
      biais[biais.length-1-i] = plus(biais[biais.length-1-i],gradient);
      
      
      
      
      matrix actErr = new matrix(0,0);
      actErr = multiply(error,Wt);
      
      error = actErr;
      
      
      
      
      
    }
    
    
  }
  
  
 
  
  int predict(){
    
  
    return(value[value.length-1].truc(value[value.length-1].max_index()));
  }
  
  
  
  
  
  
  
  
  
  
  void save(){
     
     JSONArray  mat = new JSONArray();
     JSONArray  mat1 = new JSONArray();
     for(int i = 0 ; i < memory.length; i++){
        for(int j = 0 ; j < memory[i].matrice.length; j++){
          JSONArray  mat2 = new JSONArray();
           for(int k = 0 ; k < memory[i].matrice[j].length; k++){
              
              
              mat2.setFloat(k,memory[i].matrice[j][k]);
              //mat2.setFloat(k,i);

          
           }
            
            mat1.setJSONArray(j,mat2);
            mat2 = new JSONArray();
          
        }
        
        mat.setJSONArray(i,mat1);
        mat1 = new JSONArray();
     }
   // System.out.println(mat);
    saveJSONArray(mat, "data/network_memory/main_memory.json");
    
     mat = new JSONArray();
     mat1 = new JSONArray();
     for(int i = 0 ; i < biais.length; i++){
        for(int j = 0 ; j < biais[i].matrice.length; j++){
          JSONArray  mat2 = new JSONArray();
           for(int k = 0 ; k < biais[i].matrice[j].length; k++){
              
              
              mat2.setFloat(k,biais[i].matrice[j][k]);
              //mat2.setFloat(k,i);

          
           }
            
            mat1.setJSONArray(j,mat2);
            mat2 = new JSONArray();
          
        }
        
        mat.setJSONArray(i,mat1);
        mat1 = new JSONArray();
     }
    System.out.println("end");
    saveJSONArray(mat, "data/network_memory/biais_memory.json");
    
  }
  
  void load(){
    
    JSONArray mat = loadJSONArray(dataPath("network_memory/main_memory.json"));
      for(int i = 0 ; i < memory.length; i++){
        for(int j = 0 ; j < memory[i].matrice.length; j++){
           for(int k = 0 ; k < memory[i].matrice[j].length; k++){
              
              memory[i].set_value(j,k,mat.getJSONArray(i).getJSONArray(j).getFloat(k));
           //   System.out.println(mat.getJSONArray(i).getJSONArray(j).getFloat(k));
           //   System.out.println(memory[i].matrice[j][k]);
          
           }
          
        }
        
     }
     
      mat = loadJSONArray(dataPath("network_memory/biais_memory.json"));
      for(int i = 0 ; i < biais.length; i++){
        System.out.println(i);
        for(int j = 0 ; j < biais[i].matrice.length; j++){
           for(int k = 0 ; k < biais[i].matrice[j].length; k++){
              
              biais[i].set_value(j,k,mat.getJSONArray(i).getJSONArray(j).getFloat(k));
           //   System.out.println(mat.getJSONArray(i).getJSONArray(j).getFloat(k));
           //   System.out.println(biais[i].matrice[j][k]);
          
           }
          
        }
        
     }
    
  }
  
  
  
  
  
  
  

}