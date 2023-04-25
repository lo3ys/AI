import * as activationFunctions from '../activationFunctions';

describe('Basic tests on activation functions', () => {
  test('Sigmoid', () => {
    expect(activationFunctions.Sigmoid.func(5)).toEqual(0.9933071490757153);
  });
});
