import Matrix from '../matrix';

describe('Constructor tests', () => {
  test('Sigmoid', () => {
    const rows = 5;
    const cols = 2;
    const mat = new Matrix(rows, cols);
    expect(mat.rows).toEqual(rows);
    expect(mat.cols).toEqual(cols);
  });
});
