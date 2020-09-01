import java.lang.*;
import java.awt.*;

public class matrix{
  
  float[][] matrice;
  int x,y;
  
  matrix(int x_, int y_){
    
    x = x_;
    y= y_;
    matrice = new float[x_][y_];
    
  }  
  
  int[] max_index(){
    float max_value = 0;
    int x_index = 0;
    int y_index = 0;
    for(int i = 0; i < x; i++){
          for(int j = 0; j < y; j++){
              if(matrice[i][j] > max_value){
                //System.out.println(max_value);
                 max_value = matrice[i][j];
                 x_index = i;
                 y_index = j;
              }
         }
     }
    int[] coord = {x_index,y_index};
    
    //this.show();
    return(coord);
    
  }
  
  void calc_to(matrix m){
    
    x = m.x;
    y= m.y;
    matrice = new float[m.x][m.y];
     for(int i = 0; i < x; i++){
          for(int j = 0; j < y; j++){
     
             matrice[i][j] = m.matrice[i][j];
     
         }
     }
    
  }
  
  int truc(int[] value){
    
    return((int)(value[0]*y+value[1]));
    
  }
  
  void transpose(){
    
    float[][] new_matrice = new float[y][x];
    
     for(int i = 0; i < x; i++){
          for(int j = 0; j < y; j++){
     
             new_matrice[j][i] = matrice[i][j];
     
         }
     }
     matrice = new_matrice;
     x = matrice.length;
     y = matrice[0].length;
    
    
  }
  
  void randomize(float min, float max){
     for(int i = 0; i < x; i++){
          for(int j = 0; j < y; j++){
     
             matrice[i][j] = (float)(Math.random()*(max-min)+min);
     
         }
     }
     
    
  }
  
  void activation_function(float smooth){
  for(int i = 0; i < x; i++){
          for(int j = 0; j < y; j++){
         
             matrice[i][j] = (float)((1/(1+Math.exp(smooth*matrice[i][j]))));
           
         }
     }
  }
  
  float act_fun(float value,float smooth){
    
    return((float)((1/(1+Math.exp(smooth*value)))));
    
  }
  
  void show(){
    System.out.println("");
    System.out.print("{");
    for(int i = 0; i < x; i++){
          for(int j = 0; j < y; j++){
             
             System.out.print(matrice[i][j]);
              System.out.print(";");
             
         }
         
        System.out.println("");
    }
    System.out.println("}");
     
  }
  
  
  
  void derivative_function(float smooth){
    for(int i = 0; i < x; i++){
          for(int j = 0; j < y; j++){ 
            //float x = matrice[i][j];
            
              // if( act_fun(x,smooth) *  (1+act_fun(x,smooth)) > -0.000001 &&  act_fun(x,smooth) *  (1+act_fun(x,smooth)) < 0.000001){
                 // matrice[i][j] = -4*matrice[i][j];
             //  }else{
                 //matrice[i][j] = act_fun(x,smooth) *  (1+act_fun(x,smooth));
              // }
              Double test_ultime = ((Math.exp(smooth*matrice[i][j])) / ( Math.pow(Math.exp(smooth*matrice[i][j])+1,2)));
            if(test_ultime.isNaN()){
              System.out.println("error");
            }else{
                matrice[i][j] = matrice[i][j] * ( 1 - matrice[i][j]);
               // System.out.println((float)((Math.exp(1*j)) / ( Math.pow(Math.exp(1*j)+1,2))));
            }
              
         }
     }
  }
  
  void set_value(int x_, int y_, float value){
   
    matrice[x_][y_] = value;
    
  }
  
  matrix Add(float value){
      
    matrix result = this;
       for(int i = 0; i < x; i++){
          for(int j = 0; j < y; j++){
              result.matrice[i][j] += value;
             
         }
     }
     return(result);
      
  }
  
  
  void set_value_line(float[] array){
    
    if(x != 1){
     
      error("la matrice est plus grande que 1");
      
    }else{
      
      int loop = 0;
      for(float value: array){
        
        matrice[0][loop] = value;
        loop++;
        
      }
      
    }
    
  }
  
  
  float[] get_line_value(int line){
    
    if(y == 1){
    float[] result = new float[x]; 
    for(int i = 0; i < x; i++){
      
      result[i] = matrice[i][line];
      
    }
    return(result);
  }else{
  if(x == 1){
    float[] result = new float[y]; 
    for(int i = 0; i < y; i++){
      
      result[i] = matrice[line][i];
      
    }
    return(result);
  }else{
    float[] debug = {0};
    return(debug);
  }
  }
  }
 
  
 
  
}

void error(String error_message){
    
    System.out.println("");
    System.out.print("=======ERROR :");
    System.out.println(error_message);
    System.out.println("");
  
}

public matrix multiply(matrix factor1, matrix factor2){
  if(factor1.y != factor2.x){
   error(" factor1 y ("+factor1.y+") and factor2 x ("+factor2.x+") doesn't match");
   
   error(factor1.x+"   "+factor1.y+" | " + factor2.x+"   "+factor2.y);
   return(new matrix(1,1));
    
  }else{
    
   
   matrix result = new matrix(factor1.x, factor2.y);
   for(int i = 0; i < result.y; i++){
     for(int j = 0; j < result.x; j++){
       
       float sum = 0;
       for(int k = 0; k < factor2.x ; k++){
             sum += factor1.matrice[j][k]*factor2.matrice[k][i];
         
         
       }
       
       result.set_value(j,i,sum);
       
     }
   }
   return(result);
  }
}

public matrix plus(matrix term1, matrix term2){
  
  if(term1.x != term2.x){
   
    error(" term1 x ("+term1.x+") and term2 x ("+term2.x+") doesn't match");
    error(term1.x+"   "+term1.y+" | " + term2.x+"   "+term2.y);
    return(new matrix(1,1));
    
  }else{
    if(term1.y != term2.y){
   
    error(" term1 y and term2 y doesn't match");
    
    error(term1.x+"   "+term1.y+" | " + term2.x+"   "+term2.y);
    return(new matrix(1,1));
    
    }else{
     matrix result = new matrix(term1.x,term1.y);
       for(int i = 0; i < result.x; i++){
          for(int j = 0; j < result.y; j++){
            if(!((Double)((double)(term1.matrice[i][j]+term2.matrice[i][j]))).isNaN()){
              if(term1.matrice[i][j]+term2.matrice[i][j] > 0.25){
                  
              // System.out.println("error"); 
              }
               result.set_value(i,j,term1.matrice[i][j]+term2.matrice[i][j]);
            }else{
            //   System.out.println("error"); 
            }
          }
       }
      
      return(result);
    }
  }
}

public matrix substract(matrix term1, matrix term2){
  
  if(term1.x != term2.x){
   
    error(" term1 x and term2 x doesn't match");
    return(new matrix(1,1));
    
  }else{
    if(term1.y != term2.y){
   
    error(" term1 y and term2 y doesn't match");
    return(new matrix(1,1));
    
    }else{
     matrix result = new matrix(term1.x,term1.y);
       for(int i = 0; i < result.x; i++){
          for(int j = 0; j < result.y; j++){
               result.set_value(i,j,term1.matrice[i][j]-term2.matrice[i][j]);
            
          }
       }
      
      return(result);
    }
  }
}

public matrix multi(matrix factor, float multiplier){
  
 
     matrix result = factor;
       for(int i = 0; i < result.x; i++){
          for(int j = 0; j < result.y; j++){
               result.set_value(i,j,result.matrice[i][j]*multiplier);
            
          }
       }
      
      return(result);
}

public matrix fusion(matrix factor, matrix factor_2){
  
  if(factor.x != factor_2.x){
   
   error(" term1 x ("+factor.x+") and term2 x ("+factor_2.x+") doesn't match");
    error("  "+ factor.x+" |  " +    factor.y);
    error("  "+ factor_2.x+" |  " +    factor_2.y);
    return(new matrix(1,1));
    
  }else{
    if(factor.y != factor_2.y){
   
    error(" term1 y and term2 y doesn't match");
    return(new matrix(1,1));
    
    }else{
     matrix result = factor;
       for(int i = 0; i < result.x; i++){
          for(int j = 0; j < result.y; j++){
             
               result.set_value(i,j,result.matrice[i][j]*factor_2.matrice[i][j]);
             
          }
       }
      
      return(result);
    }
  }
}