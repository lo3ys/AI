import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

/** if the Matrix and functions are in a separated library
import utilities.Matrix;
import utilities.functions;**/
//neural network class made by Physic Dev (physic gamer)
/**
 * this class create fully connected neural network.
 * 
 * @version 1.1
 * @author Physic Dev
 * 
 */
public class NeuralNetwork {
	private int inputLength;
	/**
	 * return the number of input of the neural network
	 * @return the number of input
	 * @since 1.0
	 */
	public int getInputLength() {
		return(inputLength);
	}
		
	private int outputLength;
	/**
	 * return the number of output of the neural network
	 * @return the number of output
	 * @since 1.0
	 */
	public int getOutputLength() {
		return(outputLength);
	}
		
	private int[] hiddenLength;
	/**
	 * return the number of neurons of the corresponding hidden layer
	 * 
	 * @param x the hidden layer
	 * @return the number of neuron
	 * @since 1.0
	 * @see #getSructure(int)
	 */
	public int getHiddenLength(int x){
		return(hiddenLength[x]);
	}
	
	private int[] structure;
	/**
	 * return the number of neurones of the corresponding layer
	 * @param x the layer
	 * @return the number of neuron
	 * @since 1.0
	 * @see #getHiddenLength(int)
	 */
	public int getSructure(int x){
		return(structure[x]);
	}
	
	private int layer;
	/**
	 * return the total number of layer in the neural network (with input and output)
	 * @return the number of layer
	 */
	public int getLength(){
		return(layer);
	}
	
	//data of the neural network
	private Matrix[] weights;
	private Matrix[] values;
	private Matrix[] biais;
	/**
	 * 
	 * return a float array containing the output of the neural network
	 * 
	 * @return a float array containing the output of the neural network
	 * @since 1.2
	 */
	public float[] getOutput() {
		return(new Matrix(values[values.length-1]).values[0]);
	}
	
	//number of time the neural network was trained
	private int experience;
	/**
	 * return the number of time the neural network trained
	 * @return the experience of the neural network
	 * @since 1.0
	 */
	public int getXp(){
		return(experience);
	}
	
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
	
	//activation function by layer
	private Function<Float, Float>[] activation;
	private Function<Float, Float>[] derivative;
	
	/**
	 * define the activation function used by the neural network
	 * @param actf the activation function
	 * @param derf the derivative of the activation function
	 * @apiNote the derivative variable must be the activation function output
	 *
	 * @since 1.0
	 * @see #setActivation(functions)
	 * @see #setActivation(functions[])
	 * @see #setActivation(Function[], Function[])
	 * @see #setActivation(functions, functions)
	 * @see #setActivation(Function, Function, Function, Function)
	 */
	public void setActivation(Function<Float, Float> actf,Function<Float, Float> derf) {
		Arrays.fill(activation,actf);
		Arrays.fill(derivative,derf);
	}
	/**
	 * 
	 * define the activations functions used by the neural network:
	 * </br></br>
	 * all of the layer will use the first activation function except
	 * the output layer which will use the second one 
	 * 
	 * @param actf the first activation function
	 * @param derf the derivative of the first activation function
	 * @param outActf the second activation function
	 * @param outDerf the derivative of the second activation function
	 * 
	 * @apiNote the derivative variable must be the activation function output
	 * @since 1.0
	 * 
	 * @see #setActivation(functions)
	 * @see #setActivation(functions[])
	 * @see #setActivation(Function[], Function[])
	 * @see #setActivation(functions, functions)
	 * @see #setActivation(Function, Function)
	 */
	public void setActivation(Function<Float, Float> actf,Function<Float, Float> derf,
							  Function<Float, Float> outActf,Function<Float, Float> outDerf) {
		setActivation(actf,derf);
		activation[activation.length-1]=outActf;
		derivative[activation.length-1]=outDerf;
	}
	/**
	 * 
	 * define the activations functions used by the neural network.
	 * Each layer will have a different activation function
	 * 
	 * 
	 * @param actArray the array of activation function
	 * @param derArray the array of derivative function
	 * 
	 * @apiNote the derivative variable must be the activation function output
	 * @since 1.0
	 * 
	 * @see #setActivation(functions)
	 * @see #setActivation(functions[])
	 * @see #setActivation(Function, Function, Function, Function)
	 * @see #setActivation(functions, functions)
	 * @see #setActivation(Function, Function)
	 */
	public void setActivation(Function<Float, Float>[] actArray,Function<Float, Float>[] derArray) throws IndexOutOfBoundsException{
		if(actArray.length != activation.length || derArray.length != derivative.length){
			throw new IndexOutOfBoundsException("the length of input array don't have the size of the neural network");			
		}
		for(int i=0;i<actArray.length;i++) {
			activation[i]=actArray[i];
			derivative[i]=derArray[i];
		}
	}
	
	/**
	 * do the same thing as {@link #setActivation(Function, Function)} but with the built-in activation function
	 * 
	 * 
	 * @param f the activations functions
	 * @since 1.0
	 * @see #setActivation(Function, Function)
	 *
	 */
	public void setActivation(functions f) {setActivation(f.Function,f.Derivative);}
	/**
	 * do the same thing as {@link #setActivation(Function, Function,Function,Function)} but with the built-in activation function
	 * 
	 * 
	 * @param f the activations functions
	 * @param outf the output activation function
	 * @since 1.0
	 * @see #setActivation(Function, Function,Function,Function)
	 *
	 */
	public void setActivation(functions f,functions outf) {setActivation(f.Function,f.Derivative,outf.Function,outf.Derivative);}
	/**
	 * do the same thing as {@link #setActivation(Function[], Function[])} but with the built-in activation function
	 * 
	 * 
	 * @param actArray the activations functions
	 * @since 1.0
	 * @see #setActivation(Function[], Function[])
	 *
	 */
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
	/**
	 * create a fully connected neural network with multiple hidden layer of neurons.</br>
	 * the standard activation function are the sigmoid function.
	 * @param In the number of inputs
	 * @param Hid array containing the number of neurons for each hidden layer</br>
	 *        the number of hidden layer is given with the array length
	 * @param Out the number of outputs
	 * 
	 * @throws IllegalArgumentException if one or more values are negative or null
	 * @see #NeuralNetwork(int[])
	 * 
	 * @since 1.0
	 * @apiNote the constructor didn't randomize the matrix values so you have to use {@link #randomize()} in order to use the neural network.
	 */
	public NeuralNetwork(int In, int[] Hid,int Out) throws IllegalArgumentException{
		
		//check for invalid argument
		if(In<=0 || Out<=0) {
			throw new IllegalArgumentException("the number of input or output is null or negative");
		}
		for(int d:Hid) {if(d<=0) {throw new IllegalArgumentException("the number of input or output is null or negative");}}
		
		//create matrix array
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
		
		//building matrix
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
		
		//default activation function
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
	
	/**
	 * create a fully connected neural network with multiple hidden layer of neurons.</br>
	 * the standard activation function are the sigmoid function.
	 * @param structure array containing the number of neurons for each layer</br> 
	 * the first and the last value of the array are the input and the output, </br> the other value correspond to the hidden layers</br>
	 * 
	 * @throws IllegalArgumentException if one or more values are negative or null
	 * @see #NeuralNetwork(int, int[], int)
	 * 
	 * @since 1.0
	 * @apiNote the constructor didn't randomize the matrix values so you have to use {@link #randomize()} in order to use the neural network.
	 */
	public NeuralNetwork(int[] structure) {
		this(structure[0],Arrays.copyOfRange(structure, 1, structure.length-1),structure[structure.length-1]);
	}
	/**
	 * randomize weight and biais with value between two values, this methods reset the experience variable because the neuralNetwork is "brainwashed"
	 * @param start the minimum value
	 * @param end the maximum value
	 * 
	 * @see #randomize(float, float)
	 * @see #randomizeLayer(int)
	 * @see #randomizeLayer(int, float, float)
	 * @see #XavierInit()
	 * @since 1.0
	 */
	public void randomize(float start,float end){
		for(Matrix m: weights){
			m.randomize(start, end);
		}
		for(Matrix m: biais){
			m.randomize(start, end);
		}
		//reset experience because the neural network is "brainwashed"
		experience=0;
	}
	
	/**
	 * randomize weight and biais with value between -1 and 1, this methods reset the experience variable because the neuralNetwork is "brainwashed"
	 * 
	 * @see #randomize(float, float)
	 * @see #randomizeLayer(int)
	 * @see #randomizeLayer(int, float, float)
	 * @see #XavierInit()
	 * @since 1.0
	 */
	public void randomize(){
		randomize(-1,1);
	}
	
	/**
	 * randomize weight and biais with the xavier initialisation, this methods reset the experience variable because the neuralNetwork is "brainwashed"
	 * 
	 * @param n the parameter of the xavier distribution
	 * @since 1.0
	 * @see #XavierInit()
	 * @see #randomize()
	 * @see #randomize(float, float)
	 * @since 1.0
	 */
	public void XavierInit(int n) {
		randomize(-1/(float)Math.sqrt(n),1/(float)Math.sqrt(n));
	}
	
	/**
	 * randomize weight and biais with the xavier initialisation, this methods reset the experience variable because the neuralNetwork is "brainwashed"
	 * 
	 * @since 1.0
	 * @see #XavierInit(int)
	 * @see #randomize()
	 * @see #randomize(float, float)
	 * @since 1.0
	 */
	public void XavierInit() {
		 for(int i=0;i<weights.length;i++) {
			 if(i==0) {
				 randomizeLayer(i,-1/(float)Math.sqrt(inputLength),1/(float)Math.sqrt(inputLength));
				 continue;
			 }
			 randomizeLayer(i,-1/(float)Math.sqrt(hiddenLength[i-1]),1/(float)Math.sqrt(hiddenLength[i-1]));
		 }
	}
	

	/**
	 * randomize weight and biais of only one layer with value between -1 and 1.</br>
	 * This methods didn't reset the experience variable because the neuralNetwork keep a part of its training.
	 * @param layer the layer to randomize, input is 0
	 * 
	 * @see #randomize()
	 * @see #randomize(float, float)
	 * @see #randomizeLayer(int, float, float)
	 * @see #XavierInit()
	 * 
	 * @throws  IndexOutOfBoundException if the layer is negative or exceeds number of layer of the neural network.
	 * @since 1.0
	 */
	public void randomizeLayer(int layer) {
		randomizeLayer(layer,-1,1);
	}
	
	/**
	 * randomize weight and biais of only one layer with value between two value.,</br>
	 * This methods didn't reset the experience variable because the neuralNetwork keep a part of its training.
	 * @param layer the layer to randomize, input is 0
	 * @param start the minimum value
	 * @param end the maximum value
	 * 
	 * @see #randomize()
	 * @see #randomize(float, float)
	 * @see #randomizeLayer(int)
	 * @see #XavierInit()
	 * 
	 * @throws  IndexOutOfBoundException if the layer is negative or exceeds number of layer of the neural network.
	 * @since 1.0
	 */
	public void randomizeLayer(int layer,float start,float end) throws IndexOutOfBoundsException{
		if(layer >= weights.length) {
			throw new IndexOutOfBoundsException("the layer is out of bound");
		}
		weights[layer].randomize(start,end);
		biais[layer].randomize(start,end);
	}
	
	/**
	 * assign Input value to the neural network input
	 * 
	 * @param In the inputs values
	 * @throws IllegalArgumentException if the In parameter isn't a matrix or a float array
	 * @throws IndexOutOfBoundsException if the data length doesn't match the neural network input size.
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
			throw new IndexOutOfBoundsException("matrix size didn't match neural network input size");
		}
		values[0]=In;
	}
	
	private void setInput(float[] In) throws IndexOutOfBoundsException{
		if(In.length!=inputLength){
			throw new IndexOutOfBoundsException("array size didn't match neural network input size");
		}
		values[0]=new Matrix(In);
	}
	
	/**
	 * assign input value then compute the neural network output (feed forward).
	 * 
	 * @param In the inputs values
	 * @throws IllegalArgumentException if the In parameter isn't a matrix or a float array
	 * @throws IndexOutOfBoundsException if the data length doesn't match the neural network input size
	 * 
	 * @see #compute()
	 * @since 1.0
	 */
	public void compute(Object In){
		setInput(In);
		compute();
	}
	
	/**
	 * compute the neural network output (feed forward) with the current input.
	 * 
	 * @see #compute(Object)
	 * @since 1.0
	 */
	public void compute(){
		for(int i=0;i<weights.length;i++){
			values[i+1]=weights[i].product(values[i]);
			values[i+1].map(activation[i]);
		}
	}
	
	/**
	 * compute the backpropagation, change the value of the weights and the biais to get output closer to the target.
	 * @param target the target values
	 * @throws IllegalArgumentException if the target parameter isn't a matrix or a float array
	 * @throws IndexOutOfBoundsException if the data length doesn't match the neural network output size
	 * 
	 * 
	 * @see #backPropagation(Object, Object)
	 * @see #backPropagation(Object[], Object[])
	 * @see #backPropagation(Object[], Object[], int)
	 * @since 1.0
	 */
	public void backPropagation(Object target) throws IllegalArgumentException,IndexOutOfBoundsException{
		Matrix Err = null;
		if(target.getClass() != Matrix.class && target.getClass() != float[].class) {
			throw new IllegalArgumentException("faut mettre du float[] ou de matrix");
		}
		if(target.getClass() == Matrix.class) {
			if(((Matrix) target).getX()!=1 || ((Matrix) target).getY()!=outputLength) {
				throw new IndexOutOfBoundsException("matrix size didn't match neural network output size");
			}
			Err=((Matrix)target).clone();
		}else if(target.getClass() == float[].class){
			if(((float[])target).length!=outputLength){
				throw new IndexOutOfBoundsException("array size didn't match neural network input size");
			}
			Err =new Matrix((float[])target);
		}
		
		Err.substract(values[values.length-1]);
		for(int i=0;i<weights.length;i++) {

			//compute Gradient
			Matrix Nsum=values[values.length-1-i].clone();
			Nsum.map(derivative[derivative.length-1-i]);
			Nsum.fusion(Err);
			//to avoid Nan issue
			Nsum.clamp(10);

			//computing the new error before changing the weight
			Matrix oldW=weights[weights.length-1-i].clone();
			Err = oldW.transpose().product(Nsum);
			
			Nsum.factor(learningRate);
			
			//compute the new weights and biais
			weights[weights.length-1-i].add(Nsum.product(values[values.length-2-i].transpose()));
			biais[biais.length-1-i].add(Nsum);
		}
	}
	/**
	 * assign the inputs to the neural network, compute its output (feedforward) then
	 * compute the backpropagation. </br> it change the value of the weights and the biais to get output closer to the target.
	 * 
	 * @param In the inputs values
	 * @param target the target values
	 * 
	 * @see #backPropagation(Object)
	 * @see #backPropagation(Object[], Object[])
	 * @see #backPropagation(Object[], Object[], int)
	 * 
	 * @throws IllegalArgumentException if the In or target parameter isn't a matrix or a float array
	 * @throws IndexOutOfBoundsException if the data length doesn't match the neural network output size
	 * @since 1.0
	 */
	public void backPropagation(Object In,Object target) {
		compute(In);
		backPropagation(target);
	}
	
	//train for every element of the array
	/**
	 * train the neural network with every values from the database
	 * 
	 * @param In the inputs values
	 * @param target the target values
	 * 
	 * @see #backPropagation(Object)
	 * @see #backPropagation(Object, Object)
	 * @see #backPropagation(Object[], Object[], int)
	 * 
	 * @throws IllegalArgumentException if the In or target parameter isn't a matrix or a float array
	 * @throws IndexOutOfBoundsException if the data length doesn't match the target length
	 * @since 1.0
	 */
	public void backPropagation(Object[] In,Object[] target) throws IndexOutOfBoundsException{
		if(In.length != target.length) {
			throw new IndexOutOfBoundsException("Inputs array length doesn't match target array length");
		}
		for(int i=0;i<In.length;i++) {
			backPropagation(In[i],target[i]);
		}
	}

	/**
	 * train the neural network X time with random values from the database
	 * 
	 * @param In the inputs values
	 * @param target the target values
	 * @param train_number number of trains
	 * 
	 * @see #backPropagation(Object)
	 * @see #backPropagation(Object, Object)
	 * @see #backPropagation(Object[], Object[], int)
	 * 
	 * @throws IllegalArgumentException if the In or target parameter isn't a matrix or a float array
	 * @throws IndexOutOfBoundsException if the data length doesn't match the target length
	 * @since 1.0
	 */
	public void backPropagation(Object[] dataBase,Object[] target,int train_number) throws IndexOutOfBoundsException{
		if(dataBase.length != target.length) {
			throw new IndexOutOfBoundsException("Inputs array length doesn't match target array length");
		}
		for(int i=0;i<train_number;i++) {
			int index=rand.nextInt(dataBase.length);
			backPropagation(dataBase[index],target[index]);
		}
	}
	
	/**
	 * Return a String containing informations about the neural network.</br>
	 * <ul>informations contain in this representation are :
		 * <li>number of inputs</li>
		 * <li>number of output</li>
		 * <li>complete structure of the neural network</li>
		 * <li>experience</li>
		 * <li>learningRate</li>
	 * </ul>
	 * 
	 * @return the String representation of the neural network
	 * @since 1.0
	 */
	public String toString() {
		return("<Class : NeuralNetwork ; inputs : "+inputLength+" ; output : "+outputLength+" ; Structure : "+Arrays.toString(structure)+" ; experience : "+experience+" ; learning Rate : "+learningRate+" >");
	}
	
	/**
	 * Return a the matrix representation of the last layer of the neural network (the ouput layer)
	 * @return the matrix representation of the last layer of the neural network
	 * @since 1.0
	 */
	public String PrintOutput() {
		return(values[values.length-1].toString());
	}
	
	/**
	 * Return the value of the index of the max value of the neural network output
	 * @return the index of the max value of the neural network output
	 * @since 1.0
	 */
	public String PrintIndexOutput() {
		return(""+values[values.length-1].max()[1]);
	}

	/**
	 * Return a String containing all weight value (including the bias)
	 * 
	 * @return a string containing the weights values
	 * @since 1.0
	 */
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
	/**
	 * save data in a special file.</br></br>
	 * <p>the file contain : <ul>
	 * 		<li>a header containing :
	 * 			<ul>
	 * 			<li>a begining of file : "NeuralNetworkDataFile :" to indicate the nature of the file</li>
	 * 			<li>the number of input</li>
	 * 			<li>the number of output</li>
	 * 			<li>the number of layer follow by all the layer number of inputs</li>
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
		
	/**
	 * load data from file an addapt the Neural network size to the data
	 * 
	 * @param path of the file
	 * @since 1.0
	 */
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
		
	/**
	 * load data from file , fail if the Neural network size doesn't match the data in the file
	 * 
	 * @param path of the file
	 * @since 1.0
	 */
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
