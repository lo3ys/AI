//MultiLayered NeuralNetwork Class with a Matrix class for calculation.
import { Activation_Function, Sigmoid } from "@ai-mer/activationfunctions"


// class Neuron {

// }

// class Layer {
// 	values: Matrix;
// 	length: number;
// 	activation_function: Activation_Function;

// 	constructor(length: number, activation_function=Sigmoid) {
// 		this.length = length;
// 		this.activation_function = activation_function;
// 	}
// }

class NeuralNetwork {
	
	constructor(in_neurons, hid_nodes, out_nodes, lr = 0.001) {
		if (typeof in_neurons != 'number')
			throw new TypeError("in_neurons must be integer")
		else if (typeof out_nodes != 'number')
			throw new TypeError("out_nodes must be integer")
		if (!Array.isArray(hid_nodes))
			hid_nodes = [hid_nodes];

		this.input_nodes = in_neurons;
		this.hidden_nodes = hid_nodes;
		this.output_nodes = out_nodes;
		this.weights = [];
		this.biases = [];
		this.learning_rate = 0.001;
		this.mutateMin = 0.02;
		this.mutateMax = 0.12;
		this.hidden_func = ReLU;
		this.hidden_dfunc = ReLU_deriv;
		this.output_func = linear;
		this.output_dfunc = linear_deriv;
		let totalLayerNum = 1 + this.hidden_nodes.length
		let layers = [];

		for (let l of this.hidden_nodes) layers.push(l);
		layers.push(this.output_nodes);

		for (let i = 0; i < totalLayerNum; i++) {
			if (i == 0) {
				this.weights.push(new Matrix(this.hidden_nodes[i], this.input_nodes));
			} else if (i == (totalLayerNum - 1)) {
				this.weights.push(new Matrix(this.output_nodes, this.hidden_nodes[i - 1]));
			} else {
				this.weights.push(new Matrix(this.hidden_nodes[i], this.hidden_nodes[i - 1]));
			}

			this.weights[i].randomize();
		}

		for (let i = 0; i < totalLayerNum; i++) {
			this.biases.push(new Matrix(layers[i], 1));
			this.biases[i].randomize();
		}
	}

	/**
	 *
	 * @param {!number[]} input_array input array to compute
	 * @returns {?number[]} computed values
	 */
	guess(input_array) {
		if (input_array.length != this.input_nodes) {
			console.log("input length must match with the length of the input's neurons");
			return null;
		}

		let inputs = Matrix.fromArray(input_array);
		let layers = [];
		let buffer;
		let i;

		for (i = 0; i < this.weights.length - 1; i++) {
			buffer = 0;
			if (i == 0) {
				buffer = Matrix.multiply(this.weights[i], inputs);
			} else {
				buffer = Matrix.multiply(this.weights[i], layers[i - 1]);
			}
			buffer.add(this.biases[i]);
			buffer.map(this.hidden_func);
			layers.push(buffer);
		}
		buffer = Matrix.multiply(this.weights[i], layers[i - 1]);
		buffer.add(this.biases[i]);
		buffer.map(this.output_func);
		return (buffer.toArray());
	}

	/**
	 *
	 * @param {!number} learning_rate
	 */
	setLearningRate(learning_rate = 0.01) {
		this.learning_rate = learning_rate;
	}

	setHiddenActivationFunctions(func, dfunc) {
		this.hidden_func = func;
		this.hidden_dfunc = dfunc;
	}

	setOutputActivationFunctions(func, dfunc) {
		this.output_func = func;
		this.output_dfunc = dfunc;
	}

	/**
	 *
	 * @param {!number[]} input_array input array for training
	 * @param {!number[]} target_array expected outputs
	 * @returns
	 */
	train(input_array, target_array) {
		if (input_array.length != this.input_nodes)
			throw new Error("input length must match with the length of the input's neurons");

		if (target_array.length != this.output_nodes) 
			throw new Error("target length must match with the length of the output's neurons");

		let inputs = Matrix.fromArray(input_array);
		let layers = [];
		for (let i = 0; i < this.weights.length; i++) {
			let buffer;
			if (i == 0) {
				buffer = Matrix.multiply(this.weights[i], inputs);
			} else {
				buffer = Matrix.multiply(this.weights[i], layers[i - 1]);
			}
			buffer.add(this.biases[i]);
			buffer.map(this.activation_function.func);
			layers.push(buffer);
		}

		let out = layers[layers.length - 1];
		let targets = Matrix.fromArray(target_array);
		let output_errors = Matrix.subtract(targets, out);
		let hGradients = [];
		let hPrevError = output_errors;
		let actErr = output_errors;
		let hDeltas = [];
		let wT;
		let antI = 0;
		for (let i = this.weights.length - 1; i > 0; --i) {
			if (i != this.weights.length - 1) {
				wT = Matrix.transpose(this.weights[i + 1]);
				actErr = Matrix.multiply(wT, hPrevError);
				hPrevError = actErr;
			}
			let bufferG = Matrix.map(layers[i], this.activation_function.dfunc);
			bufferG.multiply(actErr);
			bufferG.multiply(this.learning_rate);
			hGradients.push(bufferG);
			let lT = Matrix.transpose(layers[i - 1]);
			hDeltas.push(Matrix.multiply(hGradients[antI], lT));
			this.weights[i].add(hDeltas[antI]);
			this.biases[i].add(hGradients[antI]);
			antI += 1;
		}
	}

	/**
	 *
	 * @returns {!string} JSON string of the object
	 */
	serialize() {
		return JSON.stringify(this);
	}

	/**
	 *
	 * @param {!(string|object)} data Json object or a string to parse into a new neural network
	 * @returns {NeuralNetwork} a new neural network
	 */
	static deserialize(data) {
		if (typeof data == 'string') {
			data = JSON.parse(data);
		}
		let nn = new NeuralNetwork(data.input_nodes, data.hidden_nodes, data.output_nodes);
		nn.weights_ih = Matrix.deserialize(data.weights_ih);
		nn.weights_ho = Matrix.deserialize(data.weights_ho);
		nn.bias_h = Matrix.deserialize(data.bias_h);
		nn.bias_o = Matrix.deserialize(data.bias_o);
		nn.learning_rate = data.learning_rate;
		return nn;
	}

	copy() {
		return new NeuralNetwork(this);
	}

	// Accept an arbitrary function for mutation
	mutate(func = NeuralNetwork.randomizer) {
		for (let w of this.weights) {
			w.map(func);
		}
		for (let b of this.biases) {
			b.map(func);
		}

	}

	/**
	 * add or substract a random value between mutateMin and mutateMax to x
	 * @param {!number} x number to randomize
	 * @returns {number} randomized number
	 */
	randomizer(x) {
		return x + (Math.random() * (this.mutateMax - this.mutateMin) + this.mutateMax)
	}
}


