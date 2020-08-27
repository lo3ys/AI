function Perceptron(iNodes){
  this.iNodes = iNodes;
  this.weights = new Array(iNodes).fill(Math.random());
  this.bias = 1;
  this.lr = 0.1;
}
Perceptron.prototype.guess = function(inputs){
  if(inputs.length != this.iNodes && !Array.isArray(inputs)) return;
  sum = 0;
  for(let i in inputs){
    sum += (inputs[i]*this.weights[i])+this.bias;
  }
  return this.sigmoid(sum);
}
Perceptron.prototype.train = function(inputs,output){
  if(!Array.isArray(inputs) || inputs.length != this.iNodes || Array.isArray(output)) return;
  for(let w in this.weights){
    this.weights[w] = this.weights[w] + (this.lr*(output - this.guess(inputs))*inputs[w]);
  }
}
Perceptron.prototype.sigmoid = function(t) {
  return 1 / (1 + Math.pow(Math.E, -t));
}



/*test

p = new Perceptron(2);
console.log(p.guess([0,1]));
for(let i = 0; i< 1000; i++){
  p.train([0,1],0);
}
console.log(p.guess([0,1]));
*/
