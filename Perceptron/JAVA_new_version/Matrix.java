import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;
import java.util.function.Function;
/*

Matrix Class : Made by PhysicDev (physic gamer)

 */
//matrix class for neural network
public class Matrix {
	
	//size of the matrix
	private int X,Y;
	//value in the matrix
	public float[][] values;
	//for the random generation
	private Random Rand = new Random();
	
	public int getX() {
		return(X);
	}
	public int getY() {
		return(Y);
	}
	
	//constructor of empty matrix
	public Matrix(int X,int Y) throws IllegalArgumentException{
		if(X<=0 || Y<=0) {
			throw new IllegalArgumentException("matrix must have positive size");
		}
		this.X=X;
		this.Y=Y;
		values=new float[X][Y];
	}
	
	//constructor of square matrix
	public Matrix(int S){
		this(S,S);	
	}
	
	//constructor with float array
	public Matrix(float[][] dat) throws IllegalArgumentException{
		this(dat.length,dat[0].length);
		assign(dat);
	}
	
	//constructor with float array
	public Matrix(Matrix m) throws IllegalArgumentException{
		this(m.X,m.Y);
		assign(m);
	}
	
	//constructor for column matrix
	public Matrix(float[] dat){
		this(1,dat.length);
		for(int j=0;j<Y;j++) {
			values[0][j]=dat[j];
		}
	}
	
	//representation of the matrix
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
	
	public Matrix clone(){
		Matrix m = new Matrix(X,Y);
		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				m.values[i][j]=values[i][j];
			}
		}
		return(m);
	}
	
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
	
	//assign values to the matrix
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

	//assign values to the matrix from another matrix (to duplicate matrix)
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
	
	//transpose the matrix
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
	public void factor(float a) {
		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				values[i][j] *= a;
			}
		}
	}
	
	//not realy a matrix operation but usefull in neural network calculation
	//like the matrix sum but multiply the elements.
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
	
	//raise a square matrix to is power n
	public Matrix power(float n)throws ArithmeticException{
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
	public void randomize() {
		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				values[i][j] = Rand.nextFloat();
			}
		}
	}
	
	//randomize matrix with choosen value
	public void randomize(float start,float end) {
		for(int i=0;i<X;i++) {
			for(int j=0;j<Y;j++) {
				values[i][j] = Rand.nextFloat()*(end-start)+start;
			}
		}
	}
	
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
	
	//extends a column matrix to a length long matrix
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
	
	//map the function F on every element of the matrix
	public void map(Function<Float, Float> F) {
		for(int i=0;i<X; i++) {
			for(int j=0;j<Y; j++) {
				values[i][j]=(float)F.apply(values[i][j]);
			}
		}
	}
	
	//write the matrix in file
	public void writeInFile(DataOutputStream out) throws IOException {
		for(int i=0;i<X; i++) {
			for(int j=0;j<Y; j++) {
				out.writeFloat(values[i][j]);
			}
		}
	}

	//load a matrix from file
	public void loadFromFile(DataInputStream in) throws IOException {
		for(int i=0;i<X; i++) {
			for(int j=0;j<Y; j++) {
				values[i][j]=in.readFloat();
			}
		}
	}
}
