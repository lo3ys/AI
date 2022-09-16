/*NETSCAPE_SWEGA®
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
function Perceptron(iNodes, trainMode="default", activation = "sigmoid") {
  if(iNodes instanceof Perceptron){
    let a = iNodes;
    this.weights = a.weights;
    this.iNodes = a.iNodes;
    this.trainMode = a.trainMode;
    this.biais = a.biais;
    this.lr = a.lr;
    this.activation = a.activation;
    this.cycles = 1000;
  }else{
    this.weights = [];
    this.iNodes = iNodes;
    this.trainMode = trainMode;
    this.biais = -1;
    this.lr = 0.1;
    this.activation = activation;
    this.cycles = 1000;
    for (let i = 0; i < this.iNodes; i++) {
      this.weights[i] = Math.random();
    }
  }
}

Perceptron.prototype.guess = function(inputs) {
  if (Array.isArray(inputs)) {
    if (inputs.length == this.weights.length) {
      let sum = 0;
      for (let i in inputs) {
        sum += inputs[i] * this.weights[i];
        sum -= this.biais;
      }
      if (this.activation == "sigmoid") {
        return this.sigmoid(sum);
      }else if(this.activation == "tahn"){
        return this.tahn(sum);
      }else{
        console.log("no valid function, must be 'sigmoid' or 'tahn'.");
        return;
      }

    } else {
      console.log(`invalid inputs length, must be ${this.weights.length}`);
      return;
    }
  }
}

Perceptron.prototype.train = function(inputs, output) {
  if (Array.isArray(inputs) && Array.isArray(output) == false) {
    if (inputs.length == this.weights.length) {
      if(this.trainMode == "default"){
        for (let w in this.weights) {

          this.weights[w] = this.weights[w] + (this.lr * (output - this.query(inputs)) * inputs[w]);
        }
      }else if(this.trainMode == "burst"){
        for(let i = 0; i< this.cycles; i++){
          for (let w in this.weights) {

            this.weights[w] = this.weights[w] + (this.lr * (output - this.query(inputs)) * inputs[w]);
          }
        }
      }else{
        console.log("trainMode must be 'default' or 'burst'.");
        return;
      }
    }
  }
  return "trained";
}

Perceptron.prototype.serialize = function(){
  return JSON.stringify(this);
}

Perceptron.deserialize = function(data){
  if(typeof data == "string"){
    data = JSON.parse(data);
  }
  try{
    let perc = new Perceptron(data.iNodes, data.lr, data.activation, data.trainMode);
    perc.weights = data.weights;
    perc.iNodes = data.iNodes;
    perc.trainMode = data.trainMode;
    perc.biais = data.biais;
    perc.lr = data.lr;
    perc.activation = data.activation;
    perc.cycles = 1000;
    return perc;
  }catch(e){
    console.error(e);
  }
}

Perceptron.prototype.copy = function(){
  return new Perceptron(this);
}

Perceptron.prototype.sigmoid = function(t) {
  return 1 / (1 + Math.pow(Math.E, -t));
}

Perceptron.prototype.tahn = function(t) {
  return 2 * this.sigmoid(2 * t) - 1;
}
