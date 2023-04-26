interface Activation_Function {
  /**
   * the activation function
   */
  func: (x: number) => number;
  /**
   * the activation function derivative
   */
  func_deriv: (x: number) => number;
}

const Sigmoid: Activation_Function = {
  func: function (x: number) {
    return 1 / (1 + Math.exp(-x));
  },

  func_deriv: function (x: number) {
    return this.func(x) * (1 - this.func(x));
  },
};

const Than: Activation_Function = {
  func: function (x: number) {
    return Math.tanh(x);
  },

  func_deriv: function (x: number) {
    return 1 - this.func(x) * this.func(x);
  },
};

const ReLU: Activation_Function = {
  func: function (x: number) {
    return Math.max(0, x);
  },

  func_deriv: function (x: number) {
    if (x <= 0)
      return 0;
    else
      return 1;
  },
};

const Linear: Activation_Function = {
  func: function (x: number) {
    return x;
  },

  func_deriv: function () {
    return 1;
  },
};

export { Sigmoid, Than, ReLU, Linear };
