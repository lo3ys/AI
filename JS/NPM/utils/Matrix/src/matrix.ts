class Matrix {
  rows: number;
  cols: number;
  data: number[][];

  constructor(rows: number, cols: number) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array(this.rows)
      .fill(0)
      .map(() => Array(this.cols).fill(0));
  }

  copy() {
    const m = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        m.data[i][j] = this.data[i][j];
      }
    }
    return m;
  }

  static fromArray(arr: number[]) {
    return new Matrix(arr.length, 1).map((e: number, i: number) => arr[i]);
  }

  static subtract(a: Matrix, b: Matrix) {
    if (a.rows !== b.rows || a.cols !== b.cols)
      throw new Error('Columns and Rows of A must match Columns and Rows of B.');

    // Return a new Matrix a-b
    return new Matrix(a.rows, a.cols).map((_: number, i: number, j: number) => a.data[i][j] - b.data[i][j]);
  }

  toArray() {
    const arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  randomize() {
    return this.map(() => Math.random() * 2 - 1);
  }

  add(n: Matrix): Matrix;
  add(n: number): Matrix;
  add(n: number | Matrix): Matrix {
    if (n instanceof Matrix) {
      if (this.rows !== n.rows || this.cols !== n.cols)
        throw new Error('Columns and Rows of A must match Columns and Rows of B.');

      return this.map((e: number, i: number, j: number) => e + n.data[i][j]);
    } else {
      return this.map((e: number) => e + n);
    }
  }

  static transpose(matrix: Matrix) {
    return new Matrix(matrix.cols, matrix.rows).map((_: number, i: number, j: number) => matrix.data[j][i]);
  }

  static multiply(a: Matrix, b: Matrix) {
    // Matrix product
    if (a.cols !== b.rows) throw new Error('Columns of A must match rows of B.');

    return new Matrix(a.rows, b.cols).map((e: number, i: number, j: number) => {
      // Dot product of values in col
      let sum = 0;
      for (let k = 0; k < a.cols; k++) {
        sum += a.data[i][k] * b.data[k][j];
      }
      return sum;
    });
  }

  multiply(n: Matrix): Matrix;
  multiply(n: number): Matrix;
  multiply(n: number | Matrix): Matrix {
    if (n instanceof Matrix) {
      if (this.rows !== n.rows || this.cols !== n.cols)
        throw new Error('Columns and Rows of A must match Columns and Rows of B.');

      return this.map((e: number, i: number, j: number) => e * n.data[i][j]);
    } else {
      return this.map((e: number) => e * n);
    }
  }

  map(func: (e: number, i: number, j: number) => number) {
    // Apply a function to every element of matrix
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        const val = this.data[i][j];
        this.data[i][j] = func(val, i, j);
      }
    }
    return this;
  }

  static map(matrix: Matrix, func: (e: number, i: number, j: number) => number) {
    // Apply a function to every element of matrix
    return new Matrix(matrix.rows, matrix.cols).map((e: number, i: number, j: number) => func(matrix.data[i][j], i, j));
  }

  print() {
    console.table(this.data);
    return this;
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data: string): Matrix;
  static deserialize(data: Matrix): Matrix;
  static deserialize(data: string | Matrix): Matrix {
    if (typeof data == 'string') {
      data = JSON.parse(data) as Matrix;
    }

    const matrix = new Matrix(data.rows, data.cols);
    matrix.data = data.data;
    return matrix;
  }
}

export default Matrix;
