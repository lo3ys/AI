/*
iNodes = input number (int)

this.lr = learning rate

activation = activation function/ sigmod||tahn

trainMode = continuously training= default || burst training= burst

this.cycles = number of cycle for one burst (default 1000)

functions:

copy, return the Perceptron

serialize, return the Perceptron in JSON string

deserialize, take à JSON strig of a Perceptron, return the Perceptron in the JSON

guess, feed forward inputs and give output

train, take inputs and expected output for these, if trainMode is egual to "default" train all weights one time
if trainMode is egual to "burst" train all weights n time(s) (n = this.cycles)

sigmoid and tahn, are just activation functions
*/

class Perceptron {
  in_neurons;
  weights;
  trainMode;
  biais;
  lr;
  /**
   * @type {(x: number) => number}
   */
  activation_function;
  cycles;

  /**
   * 
   * @param {number | Perceptron} in_neurons number of input nodes
   * @param {string} train_mode 
   * @param {(x: number) => number} activation_function 
   */
  constructor(in_neurons, lr = 0.001, train_mode = "default", activation_function = Perceptron.sigmoid) {

    if(in_neurons instanceof Perceptron){

      const copy_perceptron = in_neurons;
      this.weights = copy_perceptron.weights;
      this.in_neurons = copy_perceptron.in_neurons;
      this.train_mode = copy_perceptron.train_mode;
      this.biais = copy_perceptron.biais;
      this.lr = copy_perceptron.lr;
      this.activation_function = copy_perceptron.activation_function;
      this.cycles = copy_perceptron.cycles;

    }else{

      this.weights = [];
      this.in_neurons = in_neurons;
      this.train_mode = train_mode;
      this.biais = -1;
      this.lr = 0.1;
      this.activation_function = activation_function
      this.cycles = 1000;

      for (let i = 0; i < this.in_neurons; i++) {
        this.weights[i] = Math.random();
      }

    }
  }

  /**
   * 
   * @param {number[]} inputs 
   */
  guess(inputs) {
    if (!Array.isArray(inputs) || inputs.length != this.weights.length)
      throw new Error("Inputs array must of the same length as the number of input nodes");
        
    let sum = 0;
    for (let i in inputs) {
      sum += inputs[i] * this.weights[i];
      sum -= this.biais;
    }

    return this.activation_function(sum);
  }

  train(inputs, output) {

    return new Promise((resolve, reject) => {
      if(!Array.isArray(inputs) || inputs.length != this.in_neurons)
        reject("Inputs array must of the same length as the number of input nodes");
      if (typeof output != "number")
        reject("Output needs to be a number");

      switch (this.train_mode) {
        case "default":
          for (let w in this.weights) {
            this.weights[w] = this.weights[w] + (this.lr * (output - this.guess(inputs)) * inputs[w]);
          }
          break;

        case "burst":
          for(let i = 0; i< this.cycles; i++){
            for (let w in this.weights) {
  
              this.weights[w] = this.weights[w] + (this.lr * (output - this.guess(inputs)) * inputs[w]);
            }
          }
          break;

        default:
          reject("Training mode needs to be default or burst")
          break;
      }

      resolve("ok");
    });
  }


  static deserialize(data){
    if(typeof data == "string"){
      data = JSON.parse(data);
    }
    try{
      let perc = new Perceptron(data.in_neurons, data.lr, data.activation_function, data.train_mode);
      perc.weights = data.weights;
      perc.in_neurons = data.in_neurons;
      perc.train_mode = data.train_mode;
      perc.biais = data.biais;
      perc.lr = data.lr;
      perc.activation_function = data.activation_function;
      perc.cycles = 1000;
      return perc;
    }catch(e){
      throw new Error(e);
    }
  }

  copy(){
    return new Perceptron(this);
  }

  static sigmoid = function(x) {
    return 1 / (1 + Math.pow(Math.E, -x));
  }
  
  static tahn = function(x) {
    return 2 * Perceptron.sigmoid(2 * x) - 1;
  }
}