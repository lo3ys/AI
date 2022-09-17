//MultiLayered NeuralNetwork Class with a Matrix class for calculation.

class ActivationFunction {
	constructor(func, dfunc) {
		this.func = func;
		this.dfunc = dfunc;
	}
}

class NeuralNetwork {
	/**
	 *
	 * @param {!number} in_nodes input nodes
	 * @param {!(number[]|number)} hid_nodes hidden nodes array, if one layer no needs to send an array
	 * @param {!number} out_nodes output nodes
	 * @param {!number} [lr] learning rate (default 0.001)
	 * @returns null if an error occured
	 */
	constructor(in_nodes, hid_nodes, out_nodes, lr = 0.001) {
		if (typeof in_nodes != 'number') {
			console.log('in_nodes must be integer');
			return null;
		}
		else if (typeof out_nodes != 'number') {
			console.log('out_nodes must be integer');
			return null;
		}
		if (!Array.isArray(hid_nodes))
			hid_nodes = [hid_nodes];

		this.input_nodes = in_nodes;
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
// ----------ACTIVATION FUNCTIONS----------

let sigmoid = function(x) {
	return (1 / (1 + Math.exp(-x)));
}

let sigmoid_deriv = function(x) {
	return (sigmoid(x) * (1 - sigmoid(x)))
}

let tanh = function (x) {
	return (Math.tanh(x));
}

let tahn_deriv = function (x) {
	return (1 - (tanh(x) * tanh(x)));
}

let ReLU = function(x) {
	console.log("ReLU with:", x);
	return Math.max(0, x);
};

let ReLU_deriv = function(x) {
	if (x <= 0) return (0);
	else if (x > 0) return (1);
}

let linear = function(x) {
	console.log("lin with:", x);
	return (x);
}

let linear_deriv = function(x) {
	return (1);
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
