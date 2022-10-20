package neuralnet;

import java.util.Random;
import java.io.*;
import java.util.function.Function;

import org.json.*;

import utilities.Matrix;
import utilities.functions;
/**
 * Perceptron class, simulate a single neuron,
 * 
 * @author physic dev
 * 
 * @version 1.0
 *
 * @see <a href="https://en.wikipedia.org/wiki/Perceptron">what is a Perceptron ?</a>
 */
public class Perceptron {
	
	private int inputLength;
	/**
	 * return the Perceptron's number of input
	 * 
	 * @since 1.0
	 * @return the inputLength
	 */
	public int getLength(){ 
		return(inputLength);
	}

	//input and weight
	private Matrix input;
	private Matrix weight;
	
	//output and biais (there are only 1 so no need of a matrix)
	/**
	 * output value of the perceptron
	 * @since 1.0
	 */
	public float output;
	private float biais;
	
	//number of time the perceptron was trained
	private int experience;
	
	/**
	 * experience variable increased every time the perceptron is trained.
	 * </br>
	 * this variable is reset to 0 when the perceptron is randomized or redeclared
	 * 
	 * @since 1.0
	 * @return the number of time the perceptron trained
	 */
	public int getXp(){
		return(experience);
	}

	//learningRate
	/**
	 * the amount of modification by train,
	 * </br> higher value result in learning faster but with less precision
	 * </br>
	 * </br> the standard value of this variable is usualy around 0.01 or 0.001
	 * 
	 * @since 1.0
	 * 
	 */
	public float learningRate=0.001f;
	
	//random variable
	private Random rand=new Random();
	
	//activation function
	private functions activation;
	
	//to use custom activation function
	private Function<Float, Float> CustomActivation=null;
	private Function<Float, Float> CustomDerivative=null;
	private boolean Custom=false;
	
	/**
	 * return the activation function used by the perceptron
	 * 
	 * @since 1.0
	 * @return the activation function
	 * @see #setActFunction(functions)
	 * @see #getDerFunction()
	 * @see <a href="https://en.wikipedia.org/wiki/Activation_function"> what is an activation function ?</a>
	 */
	public Function<Float, Float> getActFunction(){
		if(Custom) {
			return(CustomActivation);
		}
		return(activation.Function);
	}

	/**
	 * return the derivative function used by the perceptron
	 * 
	 * @since 1.0
	 * @return the derivative function
	 * @see #setActFunction(functions)
	 * @see #getActFunction()
	 */
	public Function<Float, Float> getDerFunction(){
		if(Custom) {
			return(CustomDerivative);
		}
		return(activation.Derivative);
	}

	//function to manage custom function
	/**
	 * select the activation the perceptron use.
	 * 
	 * @param use <b>true</b> if the Perceptron must use the custom activation function, else <b>false</b> 
	 *
	 * @throws NullPointerException
	 */
	public void Custom(boolean use) throws NullPointerException{
		if(CustomActivation==null) {
			throw new NullPointerException("There are no Custom Activation Function in this perceptron");
		}
		Custom=use;
	}

	public boolean CustomUsed(){
		return(Custom);
	}

	/**
	 * change the activation function of the neural network
	 * 
	 * @param fun the new activation function
	 * @since 1.0
	 * @see #setCustomActivation(Function, Function)
	 * @see #setCustomActivation(Function, Function, boolean)
	 * @see #getDerFunction()
	 * @see #getActFunction()
	 */
	public void setActFunction(functions fun){
		activation=fun;
	}
	
	/**
	 * assign custom activation function to the perceptron. usefull if your activation function isn't in the utilities library.
	 * 
	 * @param Activation the activation function
	 * @param Derivative the derivative function .Warning ! the derivative variable are the result of the function not its input
	 * @param use <b>true</b> if the Perceptron must use the custom activation function, else <b>false</b> 
	 * @since 1.0
	 * @see #setActFunction(functions)
	 * @see #setCustomActivation(Function, Function)
	 * @see #getDerFunction()
	 * @see #getActFunction()
	 */
	public void setCustomActivation(Function<Float, Float> Activation,Function<Float, Float> Derivative, boolean use){
		CustomActivation=Activation;
		CustomDerivative=Derivative;
		Custom(use);
	}
	
	/**
	 * assign custom activation function to the perceptron. usefull if your activation function isn't in the utilities library.
	 * 
	 * @param Activation the activation function
	 * @param Derivative the derivative function .Warning ! the derivative variable are the result of the function not its input
	 * @since 1.0
	 * @see #setActFunction(functions)
	 * @see #setCustomActivation(Function, Function,use)
	 * @see #getDerFunction()
	 * @see #getActFunction()
	 */
	public void setCustomActivation(Function<Float, Float> Activation,Function<Float, Float> Derivative){
		setCustomActivation(Activation,Derivative,true);
	}
	
	/** Constructor **/
	
	/**
	 * create a new perceptron with L inputs
	 * @param L the number of perceptron's input
	 * @param ActFun the activation function (if you want to assign custom activation function, you must use another method after the constructor)
	 *
	 * @throws NegativeArraySizeException if L is negative
	 * @throws NullPointerException if L is null
	 * 
	 * @since 1.0
	 * 
	 * @see #Perceptron(int)
	 * @see #Perceptron()
	 *
	 */
	public Perceptron(int L,functions ActFun) throws NegativeArraySizeException,NullPointerException{
		if(L<=0) {
			throw new NegativeArraySizeException("tried to create perceptron with negative number of input");
		}else if(L==0) {
			throw new NegativeArraySizeException("tried to create perceptron with no input");
		}
		inputLength=L;
		input=new Matrix(1,L);
		weight=new Matrix(L,1);
		activation=ActFun;
		experience=0;
	}
	
	/**
	 * create a new perceptron with L inputs
	 * @param L the number of perceptron's input
	 * 
	 * @throws NegativeArraySizeException if L is negative
	 * @throws NullPointerException if L is null
	 * 
	 * @since 1.0
	 *
	 *
	 * @see #Perceptron(int, functions)
	 * @see #Perceptron()
	 */
	public Perceptron(int L){
		this(L,functions.Step);
	}
	
	/**
	 * create a new basic perceptron with 2 inputs
	 * 
	 * 
	 * @since 1.0
	 *
	 * @see #Perceptron(int, functions)
	 * @see #Perceptron(int)
	 */
	public Perceptron(){
		this(2);
	}
	
	/**
	 * randomize weight and biais with custom range, this methods reset the experience variable because the perceptron is "brainwashed"
	 * 
	 * @param start the minimum value of range
	 * @param end the maximum value of range
	 *
	 * @see #randomize()
	 * @since 1.0
	 */
	public void randomize(float start,float end){
		biais=rand.nextFloat()*(end-start)+start;
		weight.randomize(start, end);
		//reset experience because the perceptron is "brainwashed"
		experience=0;
	}

	/**
	 * randomize weight and biais with value between -1 and 1, this methods reset the experience variable because the perceptron is "brainwashed"
	 * 
	 * @see #randomize(float, float)
	 * @since 1.0
	 */
	public void randomize(){
		randomize(-1,1);
	}
	
	//generate random number between -(1/sqrt(n) and (1/sqrt(n)
	/**
	 * randomize weight and biais with the xavier initialisation,this methods reset the experience variable because the perceptron is "brainwashed"
	 * 
	 * @param n the parameter of the xavier distribution
	 * @since 1.0
	 * @see #XavierInit()
	 * @see #randomize()
	 */
	public void XavierInit(int n) {
		randomize(-1/(float)Math.sqrt(n),1/(float)Math.sqrt(n));
	}
	
	//for optimal n value
	/**
	 * randomize weight and biais with the xavier initialisation,this methods reset the experience variable because the perceptron is "brainwashed"
	 * 
	 * @since 1.0
	 * @see #XavierInit(int)
	 * @see #randomize()
	 */
	public void XavierInit() {
		 XavierInit(inputLength);
	}
	
	/**
	 * 
	 * assign input valeu to the perceptron
	 * 
	 * @param In the Input data
	 * @throws IllegalArgumentException if the input data aren't Matrix or float array
	 * @throws IndexOutOfBoundException if the input data has not the same size as the perceptron
	 * @since 1.0
	 */
	public void setInput(Object In) throws IllegalArgumentException{
		if(In.getClass() == Matrix.class) {
			setInput((Matrix)In);
			return;
		}else if(In.getClass() == float[].class){
			setInput((float[])In);
			return;
		}
		throw new IllegalArgumentException("Illegal Input type, input must be a matrix of a float array");
	}
	
	private void setInput(Matrix In) throws IndexOutOfBoundsException{
		if(In.getX()!=1 || In.getY()!=inputLength) {
			throw new IndexOutOfBoundsException("matrix size didn't match perceptron's input size");
		}
		input=In;
	}
	
	private void setInput(float[] In) throws IndexOutOfBoundsException{
		if(In.length!=inputLength) {
			throw new IndexOutOfBoundsException("array size didn't match perceptron's input size");
		}
		input=new Matrix(In);
	}
	
	/**
	 * 
	 * compute the Perceptron output with the given input
	 * 
	 * @param In the input data
	 * @since 1.0
	 * @see #compute()
	 * @see #setInput(Object)
	 */
	public void compute(Object In){
		setInput(In);
		compute();
	}
	
	/**
	 * compute the Perceptron output with the current input
	 * 
	 * @param In the input data
	 * @since 1.0
	 * @see #compute(Object)
	 * @see #setInput(Object)
	 */
	public void compute(){
		Matrix Out=weight.product(input);
		output=(Out.values[0][0]+biais);
		if(Custom) {
			output=CustomActivation.apply(output);
		}else{
			output=activation.apply(output);
		}
	}
	/**
	 * 
	 * train the Perceptron with its actual output.</br>
	 * The Perceptron train by changing its weights to get an output closer to the target the next time.
	 *
	 * @param target the wanted output
	 * @since 1.0
	 * @see #train(Object, float)
	 * @see #train(Object[], float[])
	 * @see #train(Object[], float[], int)
	 */
	public void train(float target){
		float cost;
		if(Custom) {
			cost=CustomDerivative.apply(output)*(target-output);
		}else{
			if(activation==functions.Step || activation ==functions.AbsStep) {
				cost=(target-output);
			}else {
				cost=activation.derivative(output)*(target-output);
			}
		}
		Matrix gradient = input.clone().transpose();
		gradient.factor(cost*learningRate);
		weight.add(gradient);
		//biais training
		biais+=cost*learningRate;
		experience +=1;
	}
	/**
	 * compute the Perceptron output then train it.</br>
	 * The Perceptron train by changing its weights to get an output closer to the target the next time.
	 * 
	 * @param In the Input data
	 * @param target the wanted output
	 * @since 1.0
	 * @see #train(float)
	 * @see #train(Object[], float[])
	 * @see #train(Object[], float[], int)
	 */
	public void train(Object In,float target) {
		compute(In);
		train(target);
	}
	
	//train for every element of the array
	/**
	 * compute the Perceptron output then train it for every item of the array.</br>
	 * The Perceptron train by changing its weights to get an output closer to the target the next time.
	 *
	 * @param dataBase the Inputs database
	 * @param target the wanted output array (one for each input from the database)
	 * @since 1.0
	 * @see #train(float)
	 * @see #train(Object, float)
	 * @see #train(Object[], float[], int)
	 * 
	 * @throws IndexOutOfBoundsException if the target array doesn't match the database array
	 */
	public void train(Object[] dataBase,float[] target) throws IndexOutOfBoundsException{
		if(dataBase.length != target.length) {
			throw new IndexOutOfBoundsException("Inputs array length doesn't match target array length");
		}
		for(int i=0;i<dataBase.length;i++) {
			train(dataBase[i],target[i]);
		}
	}

	//train X time with random element in the dataBase
	/**
	 * compute the Perceptron output then train it X times with random input from the database.</br>
	 * The Perceptron train by changing its weights to get an output closer to the target the next time.
	 *
	 * @param dataBase the Inputs database
	 * @param target the wanted output array (one for each input from the database)
	 * @param X the number of train
	 * @since 1.0
	 * @see #train(float)
	 * @see #train(Object, float)
	 * @see #train(Object[], float[])
	 * 
	 * @throws IndexOutOfBoundsException if the target array doesn't match the database array
	 */
	public void train(Object[] dataBase,float[] target,int train_number) throws IndexOutOfBoundsException{
		if(dataBase.length != target.length) {
			throw new IndexOutOfBoundsException("Inputs array length doesn't match target array length");
		}
		for(int i=0;i<train_number;i++) {
			int index=rand.nextInt(dataBase.length);
			train(dataBase[index],target[index]);
		}
	}
	
	/**
	 * Return a String containing informations about the Perceptron.</br>
	 * <ul>informations contain in this representation are :
		 * <li>number of inputs</li>
		 * <li>experience</li>
		 * <li>learningRate</li>
		 * <li>the used activation function</li>
	 * </ul>
	 * 
	 * @return the String representation of the Perceptron
	 * @since 1.0
	 */
	public String toString() {
		return("<Class : Perceptron ; inputs : "+inputLength+" ; experience : "+experience+" ; learning Rate : "+learningRate+" ; Activation Function : "+(Custom?"Custom":activation)+" >");
	}
	
	/**
	 * Return a String containing all weight value (including the bias)
	 * 
	 * @return a string containing the weights values
	 * @since 1.0
	 */
	public String PrintWeight() {
		String R="";
		for(int i=0; i<weight.getX();i++) {
			R+="w"+i+" : "+weight.values[i][0]+"\n";
		}
		R+="biais : "+biais;
		return(R);
	}
	
	/**
	 * save data in a special file.</br></br>
	 * <p>the file contain : <ul>
	 * 		<li>a header containing :
	 * 			<ul>
	 * 			<li>a begining of file : "PerceptronDataFile :" to indicate the nature of the file</li>
	 * 			<li>the number of input</li>
	 * 			<li>the perceptron experience</li>
	 * 			</ul>
	 * 		</li>
	 * 		<li>the weights (and biais) values</li>
	 * 		<li>a end of file "==>" to indicate the file was writed correctly</li>
	 * </ul></p>
	 * 
	 * @apiNote the activation function used by the perceptron isn't saved in the file
	 * 
	 * @param path of the file
	 * @since 1.0
	 */
	public void save(String path) {
		DataOutputStream out = null;
		try {
			out=new DataOutputStream(new FileOutputStream(new File(path).getAbsoluteFile()));
			out.writeUTF("PerceptronDataFile :");
			out.writeInt(inputLength);
			out.writeInt(experience);
			for(float[] w:weight.values) {out.writeFloat(w[0]);}
			out.writeFloat(biais);
			out.writeUTF("==>");
			out.flush();
			out.close();
		}
		catch(Exception e) {
			System.err.println("something went wrong ... ");
			e.printStackTrace(System.err);
		}
	}
	
	/**
	 * load data from file an addapt the percetron size to the data
	 * 
	 * @param path of the file
	 * @since 1.0
	 */
	public void load(String path){
		DataInputStream in = null;
		try {
			in=new DataInputStream(new FileInputStream(new File(path).getAbsoluteFile()));
			String typeCheck=in.readUTF();
			if(0!=typeCheck.compareTo("PerceptronDataFile :")) {
				throw new Exception("wrong or corrupt file header");
			}
			int Input = in.readInt();
			int XP = in.readInt();
			float[] dat=new float[Input];
			for(int i=0;i<Input;i++){
				dat[i]=in.readFloat();
			}
			float b=in.readFloat();
			String endCheck=in.readUTF();
			if(0!=endCheck.compareTo("==>")) {
				throw new Exception("wrong or corrupt end file");
			}
			inputLength=Input;
			experience=XP;
			biais=b;
			weight=new Matrix(dat).transpose();
			in.close();
		}
		catch(Exception e) {
			System.err.println("something went wrong ... ");
			e.printStackTrace(System.err);
		}
	}
	
	/**
	 * load data from file , fail if the perceptron size doesn't match the data in the file
	 * 
	 * @param path of the file
	 * @since 1.0
	 */
	public void strictLoad(String path) {
		DataInputStream in = null;
		try {
			in=new DataInputStream(new FileInputStream(new File(path).getAbsoluteFile()));
			String typeCheck=in.readUTF();
			if(typeCheck != "PerceptronDataFile :") {
				throw new Exception("wrong or corrupt file header : ");
			}
			int Input = in.readInt();
			int XP = in.readInt();
			float[] dat=new float[Input];
			for(int i=0;i<Input;i++){
				dat[i]=in.readFloat();
			}
			float b=in.readFloat();
			String endCheck=in.readUTF();
			if(typeCheck != "==>") {
				throw new Exception("wrong or corrupt end file");
			}
			if(inputLength != Input) {
				throw new IndexOutOfBoundsException("data size doesn't match perceptron size");
			}
			experience=XP;
			biais=b;
			weight=new Matrix(dat).transpose();
			in.close();
		}
		catch(Exception e) {
			System.err.println("something went wrong ... ");
			e.printStackTrace(System.err);
		}
	}
	
	/** have to solve weird eclipse json error before implementing this
	public void saveJson(File f) {
		JSONArray dat = new JSONArray();
	}**/
}
