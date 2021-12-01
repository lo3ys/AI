//MultiLayered NeuralNetwork Class with a Matrix class for calculation.

class ActivationFunction {
	constructor(func, dfunc) {
		this.func = func;
		this.dfunc = dfunc;
	}
}

let sigmoid = new ActivationFunction(
	x => 1 / (1 + Math.exp(-x)),
	y => y * (1 - y)
);

let tanh = new ActivationFunction(
	x => Math.tanh(x),
	y => 1 - (y * y)
);

class NeuralNetwork {
	/*
	* if the first argument is a NeuralNetwork the constructor clones it
	* USAGE: Cloned_NN = new NeuralNetwork(NN_ToClone);
	*
	* call guess(input_array) to feed forward input(s) to get result(s).
	*
	* call train(input_array, target_array) to make a train the NeuralNetwork with a backpropagation.
	* target_array is the expected output array for these inputs.
	*/

	/**
	 *
	 * @param {!number} in_nodes input nodes
	 * @param {!(number[]|number)} hid_nodes hidden nodes array, if one layer no needs to send an array
	 * @param {!number} out_nodes output nodes
	 * @returns null if an error occured
	 */
	constructor(in_nodes, hid_nodes, out_nodes) {
	if (typeof in_nodes != 'number') {
		console.log('in_nodes must be integer');
		return null;
	}

	if (!Array.isArray(hid_nodes)) {
		hid_nodes = [hid_nodes];
	}

	if (typeof out_nodes != 'number') {
		console.log('out_nodes must be integer');
		return null;
	}

	this.input_nodes = in_nodes;
	this.hidden_nodes = hid_nodes;
	this.output_nodes = out_nodes;
	let totalLayerNum = 1 + this.hidden_nodes.length
	let layers = [];

	for (let l of this.hidden_nodes) layers.push(l);
	layers.push(this.output_nodes);
	this.weights = [];

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

	this.biases = [];
	for (let i = 0; i < totalLayerNum; i++) {
		this.biases.push(new Matrix(layers[i], 1));
		this.biases[i].randomize();
	}

	this.setLearningRate();
	this.setActivationFunction();
	this.mutateMin = 0.02;
	this.mutateMax = 0.12;
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
		//calculating all layers values
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

		let out = layers[layers.length - 1].toArray();
		return out
	}

	/**
	 *
	 * @param {!number} learning_rate
	 */
	setLearningRate(learning_rate = 0.01) {
		this.learning_rate = learning_rate;
	}

	setActivationFunction(func = sigmoid) {
		this.activation_function = func;
	}

	/**
	 *
	 * @param {!number[]} input_array input array for training
	 * @param {!number[]} target_array expected outputs
	 * @returns
	 */
	train(input_array, target_array) {
		if (input_array.length != this.input_nodes) {
			console.log("input length must match with the length of the input's neurons");
			return null;
		}
		if (target_array.length != this.output_nodes) {
			console.log("target length must match with the length of the output's neurons");
			return null;
		}
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
			this.w.map(func);
		}
		for (let b of this.biases) {
			this.b.map(func);
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

// ----------MATRIX CLASS----------

class Matrix {
	constructor(rows, cols) {
		this.rows = rows;
		this.cols = cols;
		this.data = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
	}

	copy() {
		let m = new Matrix(this.rows, this.cols);
		for (let i = 0; i < this.rows; i++) {
			for (let j = 0; j < this.cols; j++) {
				m.data[i][j] = this.data[i][j];
			}
		}
		return m;
	}

	static fromArray(arr) {
		return new Matrix(arr.length, 1).map((e, i) => arr[i]);
	}

	static subtract(a, b) {
	if (a.rows !== b.rows || a.cols !== b.cols) {
		console.log('Columns and Rows of A must match Columns and Rows of B.');
		return;
	}

	// Return a new Matrix a-b
	return new Matrix(a.rows, a.cols)
		.map((_, i, j) => a.data[i][j] - b.data[i][j]);
	}

	toArray() {
		let arr = [];
		for (let i = 0; i < this.rows; i++) {
			for (let j = 0; j < this.cols; j++) {
				arr.push(this.data[i][j]);
			}
		}
		return arr;
	}

	randomize() {
	 return this.map(e => Math.random() * 2 - 1);
	}

	add(n) {
		if (n instanceof Matrix) {
			if (this.rows !== n.rows || this.cols !== n.cols) {
				console.log('Columns and Rows of A must match Columns and Rows of B.');
				return;
			}
			return this.map((e, i, j) => e + n.data[i][j]);
		} else {
			return this.map(e => e + n);
		}
	}

	static transpose(matrix) {
		return new Matrix(matrix.cols, matrix.rows)
			.map((_, i, j) => matrix.data[j][i]);
	}

	static multiply(a, b) {
		// Matrix product
		if (a.cols !== b.rows) {
			console.log('Columns of A must match rows of B.');
			return;
		}

		return new Matrix(a.rows, b.cols)
			.map((e, i, j) => {
				// Dot product of values in col
				let sum = 0;
				for (let k = 0; k < a.cols; k++) {
				sum += a.data[i][k] * b.data[k][j];
				}
				return sum;
			});
	}

	multiply(n) {
		if (n instanceof Matrix) {
			if (this.rows !== n.rows || this.cols !== n.cols) {
				console.log('Columns and Rows of A must match Columns and Rows of B.');
				return;
			}
			return this.map((e, i, j) => e * n.data[i][j]);
		} else {
			return this.map(e => e * n);
		}
	}

	map(func) {
		// Apply a function to every element of matrix
		for (let i = 0; i < this.rows; i++) {
			for (let j = 0; j < this.cols; j++) {
				let val = this.data[i][j];
				this.data[i][j] = func(val, i, j);
			}
		}
		return this;
	}

	static map(matrix, func) {
		// Apply a function to every element of matrix
		return new Matrix(matrix.rows, matrix.cols)
			.map((e, i, j) => func(matrix.data[i][j], i, j));
	}

	print() {
		console.table(this.data);
		return this;
	}

	serialize() {
		return JSON.stringify(this);
	}

	static deserialize(data) {
		if (typeof data == 'string') {
			data = JSON.parse(data);
		}
		let matrix = new Matrix(data.rows, data.cols);
		matrix.data = data.data;
		return matrix;
	}
}
