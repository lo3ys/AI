import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;
import java.util.function.Function;
/**
 * matrix class for neural network
 *
 * @author Physic Dev
 * @version 1.2
 */
//
public class Matrix {
	
	//size of the matrix
	private int X,Y;
	/**
	 * the core of the class, store all the values of the matrix
	 */
	public float[][] values;
	//for the random generation
	private Random Rand = new Random();
	
	/**
	 * get the width of the matrix
	 * 
	 * @return the width of the matrix
	 * @since 1.0
	 */
	public int getX() {
		return(X);
	}
	
	/**
	 * get the height of the matrix
	 * 
	 * @return the height of the matrix
	 * @since 1.0
	 */
	public int getY() {
		return(Y);
	}
	
	//constructor of empty matrix
	/**
	 * create an empty matrix of size x,y
	 * 
	 * @param X the width of the matrix
	 * @param Y the height of the matrix
	 * @since 1.0
	 * @throws IllegalArgumentException if the values are null or negative
	 */
	public Matrix(int X,int Y) throws IllegalArgumentException{
		if(X<=0 || Y<=0) {
			throw new IllegalArgumentException("matrix must have positive size");
		}
		this.X=X;
		this.Y=Y;
		values=new float[X][Y];
	}
	
	/**
	 * create an empty square matrix of size s
	 * 
	 * @see #Matrix(int, int)
	 * 
	 * @param S the size of the matrix
	 * @since 1.0
	 * @throws IllegalArgumentException if the value is null or negative
	 */
	public Matrix(int S){
		this(S,S);	
	}
	
	/**
	 * create a matrix from a float array
	 * 
	 * @see #assign(float[][])
	 * 
	 * @param dat the float array
	 * @since 1.0
	 */
	public Matrix(float[][] dat){
		this(dat.length,dat[0].length);
		assign(dat);
	}
	
	/**
	 * create a matrix from another Matrix
	 * @see #assign(Matrix)
	 * 
	 * @param dat the float array
	 * @since 1.0
	 */
	public Matrix(Matrix m){
		this(m.X,m.Y);
		assign(m);
	}
	
	/**
	 * create a matrix column (size 1*x) from a 1D float array
	 * 
	 * @see #Matrix(float[][])
	 * 
	 * @param dat the float array
	 * @since 1.0
	 */
	public Matrix(float[] dat){
		this(1,dat.length);
		for(int j=0;j<Y;j++) {
			values[0][j]=dat[j];
		}
	}
	
	/**
	 * return a text representation of the matrix. </br>
	 * The representation look like this :<p> <b>| 0.0  0.1  0.2 | 
	 * 									   </br> | 1.0  1.1  1.2 | 
	 * 									   </br> | 2.0  2.1  2.2 | </b></br></p>
	 * 
	 * @return a String representation of the matrix
	 * @since 1.0
	 */
	public String toString() {
		String R="";
		for(float[] a:values) {
			R+="|";
			for(float f:a) {
				R+=" "+f+" ";
			}
			R+="|\n";
		}
		return(R);
	}
	
	/**
	 * return a copy of itself
	 * 
	 * @see #Matrix(Matrix)
	 * 
	 * @return a matrix with the same size and the same values
	 * @since 1.0
	 */
	public Matrix clone(){
		Matrix m = new Matrix(X,Y);
		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				m.values[i][j]=values[i][j];
			}
		}
		return(m);
	}
	
	/**
	 * compare two matrix.
	 * 
	 * @param m the matrix to compare
	 * @return true if the matrix have the same values, else false
	 * @since 1.0
	 */
	public boolean equal(Matrix m) {
		if(X!=m.X || Y!=m.Y) {
			return(false);
		}
		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				if(m.values[i][j]!=values[i][j]) {
					return(false);
				}
			}
		}
		return(true);
	}
	
	/**
	 * assign values to the matrix
	 * 
	 * @see #Matrix(float[][])
	 * @see #assign(Matrix)
	 * 
	 * @return true if the matrix have the same values, else false
	 * @param dat the float array to copy
	 * @since 1.0
	 * @throws IllegalArgumentException if the float array size doesn't match the matrix size
	 */
	public void assign(float[][] dat) throws IllegalArgumentException{
		if(dat.length != X || dat[0].length != Y) {
			throw new IllegalArgumentException("float array length didn't match matrix size");
		}
		for(int i=0;i<X;i++) {
			if(dat[i].length != dat[0].length) {
				throw new IllegalArgumentException("variable sub-array length");
			}
			for(int j=0;j<Y;j++) {
				values[i][j]=dat[i][j];
			}
		}
	}

	/**
	 * assign values to the matrix
	 * 
	 * @see #Matrix(Matrix)
	 * @see #assign(float[][])
	 * 
	 * @return true if the matrix have the same values, else false
	 * @param m the matrix to copy
	 * @since 1.0
	 * @throws IllegalArgumentException if the matrix size doesn't match the current matrix size
	 */
	public void assign(Matrix m) {
		if(m.X != X || m.Y != Y) {
			throw new IllegalArgumentException("matrix's length didn't match matrix size");
		}
		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				values[i][j]=m.values[i][j];
			}
		}
	}
	
	/**
	 * transpose the matrix
	 * 
	 * @return a transposed matrix
	 * @since 1.0
	 * 
	 * @see <a href="https://en.wikipedia.org/wiki/Transpose"> what is transpose ?</a>
	 */
	public Matrix transpose() {
		Matrix m = new Matrix(Y,X);
		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				m.values[j][i]=values[i][j];
			}
		}
		return(m);
	}
	
	//addition
	/**
	 * sum the two matrix
	 * 
	 * @see #substract(Matrix)
	 * @see #product(Matrix)
	 * @see #factor(float)
	 * @see #fusion(Matrix)
	 * 
	 * @param m the matrix to sum
	 * @since 1.0
	 * @throws ArithmeticException if the matrix doesn't have the same size
	 */
	public void add(Matrix m) throws ArithmeticException{
		if(m.X!=X || m.Y!=Y) {
			throw new ArithmeticException("matrix's size doesn't match");
		}
		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				values[i][j] += m.values[i][j];
			}
		}
	}
	
	//soustraction
	/**
	 * substract the two matrix
	 * @see #add(Matrix)
	 * @see #product(Matrix)
	 * @see #factor(float)
	 * @see #fusion(Matrix)
	 * 
	 * @param m the matrix to substract
	 * @since 1.0
	 * @throws ArithmeticException if the matrix doesn't have the same size
	 */
	public void substract(Matrix m) throws ArithmeticException{
		if(m.X!=X || m.Y!=Y) {
			throw new ArithmeticException("matrix's size doesn't match");
		}
		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				values[i][j] -= m.values[i][j];
			}
		}
	}
	
	//matrix product
	/**
	 * return the product of the two matrix
	 * @see #add(Matrix)
	 * @see #substract(Matrix)
	 * @see #factor(float)
	 * @see #fusion(Matrix)
	 * @see #power(int)
	 * @see <a href="https://en.wikipedia.org/wiki/Matrix_multiplication"> what is matrix product ?</a>
	 * 
	 * @param m the matrix to multiply
	 * @since 1.0
	 * @throws ArithmeticException if the matrix can't be multiplied (wrong size)
	 */
	public Matrix product(Matrix m) throws ArithmeticException{
		if(m.Y!=X) {
			throw new ArithmeticException("matrix's size doesn't match");
		}
		Matrix R = new Matrix(m.X,Y);
		for(int i=0;i<m.X;i++){
			for(int j=0;j<Y;j++){
				float S=0;
				for(int k=0;k<X;k++) {
					S+=m.values[i][k]*values[k][j];
				}
				R.values[i][j] = S;
			}
		}
		return(R);
	}
	
	//factor product
	/**
	 * multiply all the term of the matrix by a value
	 * 
	 * @see #add(Matrix)
	 * @see #substract(Matrix)
	 * @see #product(Matrix)
	 * @see #fusion(Matrix)
	 * 
	 * @param a the value of the factor
	 * @since 1.0
	 */
	public void factor(float a) {
		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				values[i][j] *= a;
			}
		}
	}
	
	/**
	 * not realy a matrix operation but usefull in neural network calculation.</br>
	 * like the matrix sum but multiply the elements.
	 * @see #add(Matrix)
	 * @see #substract(Matrix)
	 * @see #product(Matrix)
	 * @see #factor(float)
	 * 
	 * @param m the matrix to fuse
	 * @since 1.0
	 * @throws ArithmeticException if the matrix doesn't have the same size
	 */
	public void fusion(Matrix m)throws ArithmeticException{
		if(m.X!=X || m.Y!=Y) {
			throw new ArithmeticException("matrix's size doesn't match");
		}
		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				values[i][j] *= m.values[i][j];
			}
		}
	}
	
	/**
	 * raise the matrix to the power n
	 * @see #product(Matrix)
	 * @see <a href="https://en.wikipedia.org/wiki/Matrix_multiplication#Square_matrices"> how work matrix power ?</a>
	 * 
	 * @param n the power
	 * @since 1.0
	 * @throws ArithmeticException if the matrix is not a square matrix
	 */
	public Matrix power(int n)throws ArithmeticException{
		if(X!=Y) {
			throw new ArithmeticException("only square matrix can be raise to power");
		}
		Matrix R=new Matrix(this);
		for(int i=0;i<n;i++) {
			R=R.product(this);
		}
		return(R);
	}
	
	//randomize matrix with 0-1 value
	/**
	 * randomize the matrix terms with value between 0 and 1 
	 * 
	 * @see #randomize(float, float)
	 * 
	 * @since 1.0
	 */
	public void randomize() {
		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				values[i][j] = Rand.nextFloat();
			}
		}
	}
	
	/**
	 * randomize the matrix terms with value between the wanted value 
	 * 
	 * @see #randomize()
	 * 
	 * @param start the minimum range value
	 * @param end the maximum range value
	 * 
	 * @since 1.0
	 */
	public void randomize(float start,float end) {
		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				values[i][j] = Rand.nextFloat()*(end-start)+start;
			}
		}
	}
	
	/**
	 * return the highest term of the matrix
	 * 
	 * @return the highest term of the matrix
	 * 
	 * @since 1.0
	 */
	public int[] max(){
		int[] Max= {-1,-1};
		float MaxValue=Float.MIN_VALUE;

		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				if(values[i][j]>MaxValue) {
					Max=new int[]{i,j};
					MaxValue=values[i][j];
				}
			}
		}
		return(Max);
	}
	
	/**
	 * return the lowest term of the matrix
	 * 
	 * @return the lowest term of the matrix
	 * 
	 * @since 1.2
	 */
	public int[] min(){
		int[] Min= {-1,-1};
		float MinValue=Float.MAX_VALUE;

		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				if(values[i][j]<MinValue) {
					Min=new int[]{i,j};
					MinValue=values[i][j];
				}
			}
		}
		return(Min);
	}
	
	/**
	 * extends a column matrix to a length long matrix (not very usefull)
	 * 
	 * @param length the length of the new matrix
	 * @return the new matrix
	 * 
	 * @since 1.0
	 */
	public Matrix extend(int length)throws ArithmeticException {
		if(X!=1) {
			throw new ArithmeticException("not a column matrix");
		}
		Matrix R=new Matrix(length,Y);
		for(int i=0;i<R.X; i++) {
			for(int j=0;j<Y; j++) {
				R.values[i][j]=values[0][j];
			}
		}
		return(R);
	}
	
	/**
	 * map the function F on every element of the matrix
	 * 
	 * @param F the function to apply
	 * 
	 * @since 1.0
	 */
	public void map(Function<Float, Float> F) {
		for(int i=0;i<X; i++) {
			for(int j=0;j<Y; j++) {
				values[i][j]=(float)F.apply(values[i][j]);
			}
		}
	}
	
	/**
	 * write the matrix in file (or in anything else)
	 * 
	 * @param out the output stream
	 * 
	 * @since 1.1
	 */
	public void writeInFile(DataOutputStream out) throws IOException {
		for(int i=0;i<X; i++) {
			for(int j=0;j<Y; j++) {
				out.writeFloat(values[i][j]);
			}
		}
	}
	
	/**
	 * load a matrix from file (or from anything else)
	 * 
	 * @param in the input stream
	 * 
	 * @since 1.1
	 */
	public void loadFromFile(DataInputStream in) throws IOException {
		for(int i=0;i<X; i++) {
			for(int j=0;j<Y; j++) {
				values[i][j]=in.readFloat();
			}
		}
	}
}
