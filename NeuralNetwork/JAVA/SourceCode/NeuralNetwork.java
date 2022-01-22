import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InvalidClassException;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

/** if the Matrix and functions are in a separated library
import utilities.Matrix;
import utilities.functions;**/

//penser a ajouter des commentaires pour chaque fonction

//neural network class made by Physic Dev (physic gamer)

public class NeuralNetwork {
	private int inputLength;
	public int getInputLength() {
		return(inputLength);
	}
		
	private int outputLength;
	public int getOutputLength() {
		return(outputLength);
	}
		
	private int[] hiddenLength;
	public int getHiddenLength(int x){
		return(hiddenLength[x]);
	}
	
	private int[] structure;
	public int getSructure(int x){
		return(structure[x]);
	}
	
	private int layer;
	public int getLength(){
		return(layer);
	}
	
	//data of the neural network
	private Matrix[] weights;
	private Matrix[] values;
	private Matrix[] biais;
	
	//number of time the neural network was trained
	private int experience;
	public int getXp(){
		return(experience);
	}
	
	//learningRate
	public float learningRate=0.001f;

	//random variable 
	private Random rand=new Random();
	
	//activation function by layer
	private Function<Float, Float>[] activation;
	private Function<Float, Float>[] derivative;
	
	public void setActivation(Function<Float, Float> actf,Function<Float, Float> derf) {
		Arrays.fill(activation,actf);
		Arrays.fill(derivative,derf);
	}
	public void setActivation(Function<Float, Float> actf,Function<Float, Float> derf,
							  Function<Float, Float> outActf,Function<Float, Float> outDerf) {
		setActivation(actf,derf);
		activation[activation.length-1]=outActf;
		derivative[activation.length-1]=outDerf;
	}
	public void setActivation(Function<Float, Float>[] actArray,Function<Float, Float>[] derArray) throws IndexOutOfBoundsException{
		if(actArray.length != activation.length || derArray.length != derivative.length){
			throw new IndexOutOfBoundsException("the length of input array don't have the size of the neural network");			
		}
		for(int i=0;i<actArray.length;i++) {
			activation[i]=actArray[i];
			derivative[i]=derArray[i];
		}
	}
	
	public void setActivation(functions f) {setActivation(f.Function,f.Derivative);}
	public void setActivation(functions f,functions outf) {setActivation(f.Function,f.Derivative,outf.Function,outf.Derivative);}
	public void setActivation(functions[] actArray) throws IndexOutOfBoundsException{
		if(actArray.length != activation.length){
			throw new IndexOutOfBoundsException("the length of input array don't have the size of the neural network");			
		}
		for(int i=0;i<actArray.length;i++) {
			activation[i]=actArray[i].Function;
			derivative[i]=actArray[i].Derivative;
		}
	}
	
	//main constructor
	public NeuralNetwork(int In, int[] Hid,int Out) {
		
		//penser a ajouter des commentaires
		layer=Hid.length+2;
		hiddenLength=Arrays.copyOf(Hid,layer-2);
		structure=new int[layer];
		for(int i=0;i<Hid.length;i++) {
			structure[i+1]=Hid[i];
		}
		structure[0]=In;
		structure[layer-1]=Out;
		inputLength=In;
		outputLength=Out;
		values=new Matrix[layer];
		weights=new Matrix[layer-1];
		biais=new Matrix[layer-1];
		
		//input setup
		values[0]=new Matrix(1,In);
		weights[0]=new Matrix(In,Hid[0]);
		biais[0] = new Matrix(1,Hid[0]);
		
		//output setup
		weights[weights.length-1]=new Matrix(Hid[Hid.length-1],Out);
		biais[biais.length-1] = new Matrix(1,Out);
		values[values.length-1]=new Matrix(1,Out);
		
		for(int i=0;i<hiddenLength.length;i++) {
			values[i+1]=new Matrix(1,Hid[i]);
			if(i!=0) {
				weights[i]=new Matrix(Hid[i-1],Hid[i]);
				biais[i]=new Matrix(1,Hid[i]);
			}
		}
		experience=0;
		activation=(Function<Float, Float>[]) new Function[layer-1];
		derivative=(Function<Float, Float>[]) new Function[layer-1];
		setActivation(functions.Sigmoid);
	}
	
	//to redefine the structure of a neural network (used for the loading of a new neural network);
	private void redefine(int[] Structure) {
		
		//penser a ajouter des commentaires
		layer=Structure.length;
		structure=new int[layer];
		structure=Arrays.copyOf(Structure,layer);
		hiddenLength=new int[layer-2];
		for(int i=0;i<layer-2;i++) {
			hiddenLength[i]=Structure[i+1];
		}
		inputLength=Structure[0];
		outputLength=Structure[layer-1];
		values=new Matrix[layer];
		weights=new Matrix[layer-1];
		biais=new Matrix[layer-1];
		
		//input setup
		values[0]=new Matrix(1,inputLength);
		weights[0]=new Matrix(inputLength,hiddenLength[0]);
		biais[0] = new Matrix(1,hiddenLength[0]);
		
		//output setup
		weights[weights.length-1]=new Matrix(hiddenLength[hiddenLength.length-1],outputLength);
		biais[biais.length-1] = new Matrix(1,outputLength);
		values[values.length-1]=new Matrix(1,outputLength);
		
		for(int i=0;i<hiddenLength.length;i++) {
			values[i+1]=new Matrix(1,hiddenLength[i]);
			if(i!=0) {
				weights[i]=new Matrix(hiddenLength[i-1],hiddenLength[i]);
				biais[i]=new Matrix(1,hiddenLength[i]);
			}
		}
		experience=0;
		activation=(Function<Float, Float>[]) new Function[layer-1];
		derivative=(Function<Float, Float>[]) new Function[layer-1];
		setActivation(functions.Sigmoid);
	}
	
	//other constructor
	public NeuralNetwork(int[] structure) {
		this(structure[0],Arrays.copyOfRange(structure, 1, structure.length-1),structure[structure.length-1]);
	}
	//randomize weight and biais
	public void randomize(float start,float end){
		for(Matrix m: weights){
			m.randomize(start, end);
		}
		for(Matrix m: biais){
			m.randomize(start, end);
		}
		//reset experience because the perceptron is "brainwashed"
		experience=0;
	}
	public void randomize(){
		randomize(-1,1);
	}
	
	//generate random number between -(1/sqrt(n) and (1/sqrt(n)
	public void XavierInit(int n) {
		randomize(-1/(float)Math.sqrt(n),1/(float)Math.sqrt(n));
	}
	
	//for optimal n value
	public void XavierInit() {
		 for(int i=0;i<weights.length;i++) {
			 if(i==0) {
				 randomizeLayer(i,-1/(float)Math.sqrt(inputLength),1/(float)Math.sqrt(inputLength));
				 continue;
			 }
			 randomizeLayer(i,-1/(float)Math.sqrt(hiddenLength[i-1]),1/(float)Math.sqrt(hiddenLength[i-1]));
		 }
	}
	
	
	public void randomizeLayer(int layer) {
		randomizeLayer(layer,-1,1);
	}
	
	public void randomizeLayer(int layer,float start,float end) throws IndexOutOfBoundsException{
		if(layer >= weights.length) {
			throw new IndexOutOfBoundsException("the layer is out of bound");
		}
		weights[layer].randomize(start,end);
		biais[layer].randomize(start,end);
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
		values[0]=In;
	}
	
	public void setInput(float[] In) throws IndexOutOfBoundsException{
		if(In.length!=inputLength){
			throw new IndexOutOfBoundsException("array size didn't match perceptron's input size");
		}
		values[0]=new Matrix(In);
	}
	
	public void compute(Object In){
		setInput(In);
		compute();
	}
	
	public void compute(){
		for(int i=0;i<weights.length;i++){
			values[i+1]=weights[i].product(values[i]);
			values[i+1].map(activation[i]);
		}
	}
	
	public void backPropagation(Object target) throws IllegalArgumentException{
		Matrix Err = null;
		if(target.getClass() != Matrix.class && target.getClass() != float[].class) {
			throw new IllegalArgumentException("faut mettre du float[] ou de matrix");
		}
		if(target.getClass() == Matrix.class) {
			Err=((Matrix)target).clone();
		}else if(target.getClass() == float[].class){
			Err =new Matrix((float[])target);
		}
		
		Err.substract(values[values.length-1]);
		Function<Float,Float>Bruh=((x) -> x*(1-x));
		for(int i=0;i<weights.length;i++) {
			
			Matrix Nsum=values[values.length-1-i].clone();
			Nsum.map(derivative[derivative.length-1-i]);
			Nsum.fusion(Err);
			Matrix oldW=weights[weights.length-1-i].clone();
			Err = oldW.transpose().product(Nsum);
			
			Nsum.factor(learningRate);
			//getting old weight before changing them for the new error computation
			
			//new weight
			weights[weights.length-1-i].add(Nsum.product(values[values.length-2-i].transpose()));
			biais[biais.length-1-i].add(Nsum);
		}
	}
	
	public void backPropagation(Object In,Object target) {
		compute(In);
		backPropagation(target);
	}
	
	//train for every element of the array
	public void backPropagation(Object[] In,Object[] target) throws IndexOutOfBoundsException{
		if(In.length != target.length) {
			throw new IndexOutOfBoundsException("Inputs array length doesn't match target array length");
		}
		for(int i=0;i<In.length;i++) {
			backPropagation(In[i],target[i]);
		}
	}

	//train X time with random element in the dataBase
	public void backPropagation(Object[] dataBase,Object[] target,int train_number) throws IndexOutOfBoundsException{
		if(dataBase.length != target.length) {
			throw new IndexOutOfBoundsException("Inputs array length doesn't match target array length");
		}
		for(int i=0;i<train_number;i++) {
			int index=rand.nextInt(dataBase.length);
			backPropagation(dataBase[index],target[index]);
		}
	}
	
	public String toString() {
		return("<Class : NeuralNetwork ; inputs : "+inputLength+" ; output : "+outputLength+" ; Structure : "+Arrays.toString(structure)+" ; experience : "+experience+" ; learning Rate : "+learningRate+" >");
	}
	
	public String PrintOutput() {
		return(values[values.length-1].toString());
	}
	
	public String PrintIndexOutput() {
		return(""+values[values.length-1].max()[1]);
	}
	
	public String PrintWeight() {
		String R="";
		for(int i=0; i<layer-1;i++) {
			R+="weight layer "+i+" :\n";
			R+=weights[i].toString();

			R+="biais layer "+i+" :\n";
			R+=biais[i].toString();
		}
		return(R);
	}
	
	//save data in a special file ( !!! do not save activations function data)
	public void save(String path) {
		DataOutputStream out = null;
		try {
			out=new DataOutputStream(new FileOutputStream(new File(path).getAbsoluteFile()));
			out.writeUTF("NeuralNetworkDataFile :");
			out.writeInt(inputLength);
			out.writeInt(outputLength);
			out.writeInt(layer);
			for(int i=0;i<layer;i++){out.writeInt(structure[i]);}
			out.writeInt(experience);
			for(Matrix m:weights){
				m.writeInFile(out);
			}
			for(Matrix b:biais){
				b.writeInFile(out);
			}
			out.writeUTF("==>");
			out.flush();
			out.close();
		}
		catch(Exception e) {
			System.err.println("something went wrong ... ");
			e.printStackTrace(System.err);
		}
	}
		
	//load data an adapt the NeuralNetwork size to the data
	public void load(String path){
		DataInputStream in = null;
		try {
			in=new DataInputStream(new FileInputStream(new File(path).getAbsoluteFile()));
			String typeCheck=in.readUTF();
			if(0!=typeCheck.compareTo("NeuralNetworkDataFile :")) {
				throw new Exception("wrong or corrupt file header");
			}
			int Input = in.readInt();
			int Output= in.readInt();
			int Layer = in.readInt();
			int[] Structure = new int[layer];
			boolean createNew=false;
			for(int i=0;i<Layer;i++) {
				int length=in.readInt();
				if(length!=structure[i]) {
					createNew=true;
				}
				Structure[i]=length;
			}
			if(createNew){
				redefine(Structure);
			}
			int XP = in.readInt();
			for(Matrix m:weights){
				m.loadFromFile(in);
			}
			for(Matrix b:biais){
				b.loadFromFile(in);
			}
			String endCheck=in.readUTF();
			if(0!=endCheck.compareTo("==>")) {
				throw new Exception("wrong or corrupt end file");
			}
			experience=XP;
			in.close();
		}
		catch(Exception e) {
			System.err.println("something went wrong ... ");
			e.printStackTrace(System.err);
		}
	}
		
	//like load but raise an error if the data size doesn't match the neural network size
	public void strictLoad(String path) {
		DataInputStream in = null;
		try {
			in=new DataInputStream(new FileInputStream(new File(path).getAbsoluteFile()));
			String typeCheck=in.readUTF();
			if(0!=typeCheck.compareTo("NeuralNetworkDataFile :")) {
				throw new Exception("wrong or corrupt file header");
			}
			int Input = in.readInt();
			int Output= in.readInt();
			int Layer = in.readInt();
			int[] Structure = new int[layer];
			for(int i=0;i<Layer;i++) {
				int length=in.readInt();
				if(length!=structure[i]) {
					throw new IndexOutOfBoundsException("the neural network size doesn't match the current neural network");
				}
				Structure[i]=length;
			}
			int XP = in.readInt();
			for(Matrix m:weights){
				m.loadFromFile(in);
			}
			for(Matrix b:biais){
				b.loadFromFile(in);
			}
			String endCheck=in.readUTF();
			if(0!=endCheck.compareTo("==>")) {
				throw new Exception("wrong or corrupt end file");
			}
			experience=XP;
			in.close();
		}
		catch(Exception e) {
			System.err.println("something went wrong ... ");
			e.printStackTrace(System.err);
		}
	}
}
