const Sigmoid = {
  func: function (x: number) {
    return 1 / (1 + Math.exp(-x));
  },

  func_deriv: function (x: number) {
    return this.func(x) * (1 - this.func(x));
  },
};

const Than = {
  func: function (x: number) {
    return Math.tanh(x);
  },

  func_deriv: function (x: number) {
    return 1 - this.func(x) * this.func(x);
  },
};

const ReLU = {
  func: function (x: number) {
    return Math.max(0, x);
  },

  func_deriv: function (x: number) {
    if (x <= 0) return 0;
    else if (x > 0) return 1;
  },
};

const Linear = {
  func: function (x: number) {
    return x;
  },

  func_deriv: function () {
    return 1;
  },
};

export { Sigmoid, Than, ReLU, Linear };
