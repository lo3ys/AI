class Perceptron {

  in_nodes;
  weights;
  bias;
  lr;

  /**
   * 
   * @param {number} in_nodes 
   * @param {number} lr learning rate
   */
  constructor(in_nodes, lr=0.001) {
    this.in_nodes = in_nodes;
    this.weights = new Array(in_nodes).fill(Math.random());
    this.bias = 1;
    this.lr = lr;
  }

  /**
   * 
   * @param {number[]} inputs 
   * @returns 
   */
  guess(inputs) {
    if(inputs.length != this.in_nodes && !Array.isArray(inputs))
      throw new Error("Inputs array not matching");

    let sum = 0;
    for(let i in inputs){
      sum += (inputs[i]*this.weights[i])+this.bias;
    }
    return this.sigmoid(sum);
  }

  /**
   * 
   * @param {number[]} inputs 
   * @param {number} output 
   */
  train(inputs, output) {
    if(!Array.isArray(inputs) || inputs.length != this.in_nodes)
      throw new Error("Inputs array not matching");
    if (typeof output != "number")
      throw new Error("Output needs to be a number");

    for(let w in this.weights){
      this.weights[w] = this.weights[w] + (this.lr*(output - this.guess(inputs))*inputs[w]);
    }
  }

  sigmoid(x) {
    return 1 / (1 + Math.pow(Math.E, -x));
  }
}

/*test

p = new Perceptron(2);
console.log(p.guess([0,1]));
for(let i = 0; i< 1000; i++){
  p.train([0,1],0);
}
console.log(p.guess([0,1]));
*/
