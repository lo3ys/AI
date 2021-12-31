import java.util.Random;
import java.io.*;
import java.util.Arrays;
import java.util.IllegalFormatException;
import java.util.function.Function;

import org.json.*;

/** if the Matrix and functions are in a separated library
import utilities.Matrix;
import utilities.functions;**/

public class Perceptron {
	
	private int inputLength;
	
	public int getLength(){
		return(inputLength);
	}

	//input and weight
	private Matrix input;
	private Matrix weight;
	
	//output and biais (there are only 1 so no need of a matrix)
	public float output;
	public float biais;
	
	//number of time the perceptron was trained
	private int experience;
	
	public int getXp(){
		return(experience);
	}

	//learningRate
	public float learningRate=0.001f;
	
	//random variable
	private Random rand=new Random();
	
	//activation function
	private functions activation;
	
	//to use custom activation function
	private Function<Float, Float> CustomActivation=null;
	private Function<Float, Float> CustomDerivative=null;
	private boolean Custom=false;
	
	public void setActFunction(functions fun){
		activation=fun;
	}
	
	public Function<Float, Float> getActFunction(){
		if(Custom) {
			return(CustomActivation);
		}
		return(activation.Function);
	}

	public Function<Float, Float> getDerFunction(){
		if(Custom) {
			return(CustomDerivative);
		}
		return(activation.Derivative);
	}

	//function to manage custom function
	public void Custom(boolean use) throws NullPointerException{
		if(CustomActivation==null) {
			throw new NullPointerException("There are no Custom Activation Function in this perceptron");
		}
		Custom=use;
	}

	public boolean CustomUsed(){
		return(Custom);
	}

	public void setCustomActivation(Function<Float, Float> Activation,Function<Float, Float> Derivative, boolean use){
		CustomActivation=Activation;
		CustomDerivative=Derivative;
		Custom(use);
	}
	
	public void setCustomActivation(Function<Float, Float> Activation,Function<Float, Float> Derivative){
		setCustomActivation(Activation,Derivative,true);
	}
	
	/** Constructorq **/
	
	public Perceptron(int L,functions ActFun){
		inputLength=L;
		input=new Matrix(1,L);
		weight=new Matrix(L,1);
		activation=ActFun;
		experience=0;
	}
	
	public Perceptron(int L){
		this(L,functions.Step);
	}
	
	public Perceptron(){
		this(2);
	}
	
	//randomize weight and biais
	public void randomize(float start,float end){
		biais=rand.nextFloat()*(end-start)+start;
		weight.randomize(start, end);
		//reset experience because the perceptron is "brainwashed"
		experience=0;
	}
	//default randomize
	public void randomize(){
		randomize(-1,1);
	}
	
	//generate random number between -(1/sqrt(n) and (1/sqrt(n)
	public void XavierInit(int n) {
		randomize(-1/(float)Math.sqrt(n),1/(float)Math.sqrt(n));
	}
	
	//for optimal n value
	public void XavierInit() {
		 XavierInit(inputLength);
	}
	
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
	
	public void setInput(Matrix In) throws IndexOutOfBoundsException{
		if(In.getX()!=1 || In.getY()!=inputLength) {
			throw new IndexOutOfBoundsException("matrix size didn't match perceptron's input size");
		}
		input=In;
	}
	
	public void setInput(float[] In) throws IndexOutOfBoundsException{
		if(In.length!=inputLength) {
			throw new IndexOutOfBoundsException("array size didn't match perceptron's input size");
		}
		input=new Matrix(In);
	}
	
	public void compute(Object In){
		setInput(In);
		compute();
	}
	
	public void compute(){
		Matrix Out=weight.product(input);
		output=(Out.values[0][0]+biais);
		if(Custom) {
			output=CustomActivation.apply(output);
		}else{
			output=activation.apply(output);
		}
	}
	
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
	
	public void train(Object In,float target) {
		compute(In);
		train(target);
	}
	
	//train for every element of the array
	public void train(Object[] In,float[] target) throws IndexOutOfBoundsException{
		if(In.length != target.length) {
			throw new IndexOutOfBoundsException("Inputs array length doesn't match target array length");
		}
		for(int i=0;i<In.length;i++) {
			train(In[i],target[i]);
		}
	}

	//train X time with random element in the dataBase
	public void train(Object[] dataBase,float[] target,int train_number) throws IndexOutOfBoundsException{
		if(dataBase.length != target.length) {
			throw new IndexOutOfBoundsException("Inputs array length doesn't match target array length");
		}
		for(int i=0;i<train_number;i++) {
			int index=rand.nextInt(dataBase.length);
			train(dataBase[index],target[index]);
		}
	}
	
	public String toString() {
		return("<Class : Perceptron ; inputs : "+inputLength+" ; experience : "+experience+" ; learning Rate : "+learningRate+" ; Activation Function : "+(Custom?"Custom":activation)+" >");
	}
	
	public String PrintWeight() {
		String R="";
		for(int i=0; i<weight.getX();i++) {
			R+="w"+i+" : "+weight.values[i][0]+"\n";
		}
		R+="biais : "+biais;
		return(R);
	}
	
	//save data in a special file
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
	
	//load data an addapt the percetron size to the data
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
	
	//like load but raise an error if the data size doesn't match the perceptron size
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
