package neuralnet;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

import utilities.functions;

/**
 * this class create standard neural network, less efficient that the NeuralNetwork class but allow more complex structure
 * 
 * this is inspired by the Neat AI algorithm
 * 
 * 
 * @see {@link neuralnet.NeuralNetwork}
 * @see {@link neuralnet.GeneticAlgorithm}
 * @author PhysicDev
 * 
 * @version 1.1
 *
 */
public class NeuronNetwork {

	
	private static final int Input = 0;
	private static final int Output = 1;
	private static final int Bias = 2;
	private static final int Hidden = 3;
	
	private static final int MaxNeurons = 100000;
	private static final float StructureFactor = 1f;
	private static final float WeightFactor = 1f;
	
	//for creation of new neurons
	private int nextId = 0;
	
	//random variable for generating random weight and other stuff that need to be random
	private Random R = new Random();
	
	//activation function
	private Function<Float, Float> activation;
	private Function<Float, Float> outputActivation;
	private Function<Float, Float> derivative;
	private Function<Float, Float> outputDerivative;
	
	//disabled neuron which isn't take in account for computation
	private ArrayList<Synapse> disabledSyn = new ArrayList<Synapse>();
	
	//performance variable
	private float fitness;
	private int tests=0;
	
	//info on the neural network structure
	private int inputLength;
	private int outputLength;
	private int totalNeurons;
	private int totalSynapses;
	/**
	 * this variable is used for the weight randomization, larger value mean mor espread out value
	 * @since 1.0
	 */
	public float weightDistrib = 10;
	
	/**
	 * learning rate is the amount of modification made by one backPropagation cycle, 
	 * larger value mean faster result but less precision.
	 * @since 1.0
	 */
	public float learningRate = 0.05f;
	
	/**
	 * if you want to control the randomness of the network (for the weight randomization for example)
	 * @param rand the random value
	 * @since 1.0
	 */
	public void SetRandom(Random rand) {
		R=rand;
	}
	
	/**
	 * 
	 * the Neuron Class is used by the neuronNetwork to store data relative to neuron
	 * 
	 * @author PhysicDev
	 * @since 1.0
	 */
	private class Neuron{
		
		/**
		 * unique Id
		 */
		public int Id;
		/**
		 * type of the Neuron:
		 * 
		 * 0: Input
		 * 1: Output
		 * 2: Bias
		 * 3: Hidden
		 */
		public int type;
		
		/**
		 * store the position of the Neuron in the NeuralNetowrk
		 */
		public int layer;
		
		/**
		 * Value of the Neuron (output)
		 */
		public float value=1;
		
		/**
		 * sum of the Neuron, useful if you don't have simple differential equation for the activation function
		 */
		public float sum;
		/**
		 * delta of the Neuron, used to propagate the error in the network
		 */
		public float delta=0;
		
		/**
		 * input Synapse.
		 */
		public ArrayList<Synapse> Synapses = new ArrayList<Synapse>();
		
		/**
		 * output Synapses.
		 */
		public ArrayList<Synapse> BackSynapses = new ArrayList<Synapse>();
		
		/**
		 * create a neuron
		 * 
		 * @param Type the type of the Neuron
		 * @param Layer the layer where the Neuron is
		 * @since 1.0
		 */
		public Neuron(int Type,int Layer){
			Id=nextId;
			nextId++;
			type=Type;
			layer=Layer;
		}

		/**
		 * compute the neuron output 
		 * @since 1.0
		 */
		public void compute() {
			sum=0;
			for(Synapse S: Synapses)
				if(S.enabled)
					sum+=S.weight*S.parent.value;
			value=(type==Output?outputActivation.apply(sum):activation.apply(sum));
		}
		
		/**
		 * update the weight of the output Synapses
		 * @since 1.0
		 * @see Synapse#updateWeight()
		 * @see #updateDelta()
		 */
		public void backPropagation() {
			for(Synapse S:BackSynapses)
				if(S.enabled)
					S.updateWeight();
		}
		
		/**
		 * update its delta (propagate the error)
		 * using the following formula :<br><br>
		 * <em><b> inputDelta = weight*F(output)*outputDelta </em></b><br>
		 * where F(y) is the derivative of the activation function and y is the output of the activation function
		 * 
		 * 
		 * @since 1.0
		 */
		public void updateDelta() {
			delta=0;
			for(Synapse S:BackSynapses) {
				if(!S.enabled)
					continue;//skip disabled synapses
				delta+=S.weight
				      *((S.children.type==Output)?outputDerivative.apply(S.children.value):derivative.apply(S.children.value))
					  *S.children.delta;
			}
		}
		
		/**
		 * Convert type value to String
		 * @param T type value
		 * @return the corresponding type or unknown if the type is not in the [0,3] interval 
		 */
		private String typeName(int T) {
			switch(T) {
				case Bias:
					return "Bias";
				case Input:
					return "Input";
				case Output:
					return "Output";
				case Hidden:
					return "Hidden";
			}
			return "Unknown";
				
		}

		/**
		 * Show the Neuron Info 
		 * and all the input synapses info
		 * 
		 * @return a String representation of the Neuron
		 */
		public String toString() {
			String output = "<Class : Neuron | Type : "+typeName(type)+" | Id : "+Id+" | value : "+value+" > "+layer+"\n";
			for(Synapse S:Synapses)
				output+="   "+S.toString()+" \n";
			return output;
		}
	}
	
	/**
	 * 
	 * the Synapse Class is used by the neuronNetwork to store data relative to the Synapses (Connection)
	 * 
	 * @author Physic Dev
	 *
	 */
	private class Synapse{
		/**
		 * unique Id generated with both parent and children Neuron Id
		 */
		public int Id;
		/**
		 * parent Neuron.
		 */
		public Neuron parent;
		/**
		 * childrenNeuron.
		 */
		public Neuron children;
		
		/**
		 * the weight of the Neuron is here
		 */
		public float weight;
		
		/**
		 * not used for now
		 */
		public boolean enabled = true;
		
		//on peut facilement créer des neurones récurent en créant un lien entre deux neurones ici
		
		/**
		 * Create a new Synapse between two Neuron
		 * @param Parent parent Neuron
		 * @param Children children Neuron
		 * @throws IllegalArgumentException if the Neuron are equal (a neuron cannot connect to itself)
	     * @since 1.0
		 */
		public Synapse(Neuron Parent,Neuron Children) throws IllegalArgumentException{
			if(Parent==Children)
				throw new IllegalArgumentException("A neuron cannot be connected to itself");
			parent=Parent;
			children=Children;
			Id = MaxNeurons*Parent.Id+Children.Id;
			weight = (R.nextFloat()-0.5f)*weightDistrib;
		}

		/**
		 * update the weight value using the following formula : <br><br>
		 * <em><b>newWeight=OldWeight+Input*delta*F(output)*lr</b></em> <br>
		 * where F(y) is the derivative of the activation function and y is the output of the activation function
	     * @since 1.0
		 * 
		 */
		public void updateWeight() {
			weight+=parent.value
					*children.delta
					*((children.type==Output)?outputDerivative.apply(children.value):derivative.apply(children.value))
					*learningRate;
		}

		/**
		 * check if two synapse are equal by comparing their Id
		 * @param S other synapse
		 * @return true if the synapse have the same Id
	     * @since 1.0
		 */
		public boolean equals(Synapse S) {
			return Id==S.Id;
		}
		
		/**
		 * 
		 * return a String representation of the Synapse
	     * @since 1.0
		 * 
		 * @return a string representation of the Synapse
		 */
		public String toString() {
			return "Parent : "+parent.Id+" ; Weight : "+weight;
		}
	}
	
	/**
	   * define the activation function used by the neural network
	   * 
	   * Warning ! the softMax function not working on this neural network
	   * 
	   * @param actf the activation function
	   * @param derf the derivative of the activation function
	   * @apiNote the derivative variable must be the activation function output
	   *
	   * @since 1.0
	   * @see #setActivation(functions)
	   * @see #setActivation(functions, functions)
	   * @see #setActivation(Function, Function, Function, Function)
	   */
	  public void setActivation(Function<Float, Float> actf,Function<Float, Float> derf) {
	    activation=actf;
	    outputActivation=activation;
	    derivative=derf;
	    outputDerivative=derivative;
	  }

	/**
	   * 
	   * define the activations functions used by the neural network:
	   * </br></br>
	   * all of the layer will use the first activation function except
	   * the output layer which will use the second one 
	   * 
	   * Warning ! the softMax function not working on this neural network
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
	   * @see #setActivation(functions, functions)
	   * @see #setActivation(Function, Function)
	   */
	  public void setActivation(Function<Float, Float> actf,Function<Float, Float> derf,
	                Function<Float, Float> outActf,Function<Float, Float> outDerf) {
	    setActivation(actf,derf);
	    outputActivation=outActf;
	    outputDerivative=outDerf;
	  }



	/**
	   * do the same thing as {@link #setActivation(Function, Function)} but with the built-in activation function
	   * 
	   * Warning ! the softMax function not working on this neural network
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
	   * Warning ! the softMax function not working on this neural network
	   * 
	   * @param f the activations functions
	   * @param outf the output activation function
	   * @since 1.0
	   * @see #setActivation(Function, Function,Function,Function)
	   *
	   */
	  public void setActivation(functions f,functions outf) {setActivation(f.Function,f.Derivative,outf.Function,outf.Derivative);}

	/**
	ArrayList<Neuron> Neurons = new ArrayList<Neuron>();//est-ce encore utile ce truc?
	
	ArrayList<Synapse> Synapses = new ArrayList<Synapse>();//est-ce encore utile ce truc?**/
	  
	  /**
	   * 
	   */
	protected ArrayList<ArrayList<Neuron>> Layers = new ArrayList<ArrayList<Neuron>>();

	public void addFitness(float fit) {
		fitness=(fitness*tests+fit)/(tests+1);
		tests++;
	}
	
	/**
	 * the fitness indicate how well the neuron is working
	 * used in genetic algorithm but can also be used to track the learning of the neural netowrk
	 * <br><br>
	 * higher value mean better result.
	 * 
	 * @return the fitness of the neural network
	 * @since 1.0
	 */
	public float getFitness() {
		return fitness;
	}
	/**
	 * reset the fitness value, useful to get rid of old test to see the current state of the network
	 * @since 1.0
	 */
	public void resetFitness() {
		fitness=0;
		tests=0;
	}
	
	/**
	 * return the number of input of the neural network
	 * @return the number of input
	 * @since 1.0
	 */
	public int getInputLength() {
	  return(inputLength);
	}
	  
	/**
	 * return the number of output of the neural network
	 * @return the number of output
	 * @since 1.0
	 */
	public int getOutputLength() {
	  return(outputLength);
	}
	
	/**
	 * 
	 * @return the total number of Neurons
	 * @since 1.0
	 */
	public int getSize() {
		return(totalNeurons);
	}
	
	/**
	 * @return the total number of Synapses
	 * @since 1.0
	 */
	public int Complexity() {
		return totalSynapses;
	}
	
	
	/**
	 * check if the network contain at least one neuron
	 * @return True if the network is empty
	 */
	public boolean isEmpty() {
		return totalNeurons==0;
	}
	
	/**
	 * check if the neuron network has at least one connection
	 * this is different from the method is empty because a network with input and output is not empty but can be simple.
	 * @return true if the neuron has 0 synapses
	 * @since 1.1
	 */
	public boolean isSimple() {
		return totalSynapses==0;
	}
	
	/**
	 * build an empty neural Network
	 * @throws IllegalArgumentException
	 * @since 1.1
	 */
	public NeuronNetwork() throws IllegalArgumentException{
		setActivation(functions.Sigmoid);//default activation function
	}
	
	/**
	 * Create a New Neuron network based on a fully connected network Structure.
	 * 
	 * the Integrity value control the amount of connection that will be create, 
	 * 0 mean no connection
	 * 1 mean all the connection (same as fully connected)
	 * 
	 * @param In number of Input neurons
	 * @param Hid number of hidden neurons by layer
	 * @param Out number of output neurons
	 * @param Integrity amount of connection created
	 * @throws IllegalArgumentException if one of the structure data is null or negative
	 * 
	 * @see #NeuronNetwork(int[])
	 * @see #NeuronNetwork(int[],float)
	 * @see #NeuronNetwork(int, int[], int)
	 * @since 1.0
	 */
	public NeuronNetwork(int In, int[] Hid,int Out,float Integrity) throws IllegalArgumentException{
	    //check for invalid argument
	    if(In<=0 || Out<=0)   throw new IllegalArgumentException("the number of input or output is null or negative");
	    for(int d:Hid)if(d<=0)throw new IllegalArgumentException("the number of neuron of one layer is null or negative");

	    inputLength = In;
	    outputLength = Out;
	    totalNeurons = In+Out;

    	for(int i=0;i<Hid.length+2;i++)
    		Layers.add(new ArrayList<Neuron>());

	    Neuron B = new Neuron(Bias,0);
    	Layers.get(0).add(B);
    	
	    //creating neuron
	    for(int i=0;i<In;i++)
	    	Layers.get(0).add(new Neuron(Input,0));
	    
    	
	    for(int j=0;j<Hid.length;j++){
	    	totalNeurons+=Hid[j];
		    for(int i=0;i<Hid[j];i++)
		    	Layers.get(j+1).add(new Neuron(Hidden,j+1));
	    }
	    
	    for(int i=0;i<Out;i++)
	    	Layers.get(Hid.length+1).add(new Neuron(Output,Hid.length+1));
	    
	    totalSynapses = 0;
	    //create connection
		for(ArrayList<Neuron> layer:Layers)
		    for(Neuron N:layer)
		    	if(N.type != Bias && N.type != Input && R.nextFloat()<Integrity) {
		    		Synapse S=new Synapse(B,N);
		    		N.Synapses.add(S);
		    		B.BackSynapses.add(S);
		    		totalSynapses++;
		    	}
		    	else
		    		continue;
		for(int i=1;i<Layers.size();i++)
			 for(Neuron N:Layers.get(i))
				 for(Neuron other:Layers.get(i-1))
					if(other.type != Bias && R.nextFloat()<Integrity) {
						Synapse S=new Synapse(other,N);
						N.Synapses.add(S);
						other.BackSynapses.add(S);
						totalSynapses++;
					}
		while(totalSynapses==0) {//to be sure there are at least one Synapse
			for(int i=1;i<Layers.size();i++)
				 for(Neuron N:Layers.get(i))
					 for(Neuron other:Layers.get(i-1))
						if(other.type != Bias && R.nextFloat()<Integrity) {
							Synapse S=new Synapse(other,N);
							N.Synapses.add(S);
							other.BackSynapses.add(S);
							totalSynapses++;
						}
		}
		
		
		setActivation(functions.Sigmoid);//default activation function
	}
	
	/**
	 * Create a New Neuron network based on a fully connected network Structure.
	 * 
	 * the Integrity value control the amount of connection that will be create, 
	 * 0 mean no connection
	 * 1 mean all the connection (same as fully connected)
	 * 
	 * @param structure the amount of neurons by layer
	 * @param Integrity amount of connection created
	 * @throws IllegalArgumentException if one of the structure data is null or negative
	 * 
	 * @see #NeuronNetwork(int, int[], int, float)
	 * @see #NeuronNetwork(int[])
	 * @see #NeuronNetwork(int, int[], int)
	 * @since 1.0
	 */
	public NeuronNetwork(int[] structure,float Integrity) {
	   this(structure[0],Arrays.copyOfRange(structure, 1, structure.length-1),structure[structure.length-1],Integrity);
	}

	/**
	 * Create a New Neuron network based on a fully connected network Structure.
	 * 
	 * 
	 * @param structure the amount of neurons by layer
	 * @throws IllegalArgumentException if one of the structure data is null or negative
	 * 
	 * @see #NeuronNetwork(int, int[], int, float)
	 * @see #NeuronNetwork(int[],float)
	 * @see #NeuronNetwork(int, int[], int)
	 * @since 1.0
	 */
	public NeuronNetwork(int[] structure) {
	   this(structure[0],Arrays.copyOfRange(structure, 1, structure.length-1),structure[structure.length-1]);
	}
	
	/**
	 * Create a New Neuron network based on a fully connected network Structure.
	 * 
	 * 
	 * @param In number of Input neurons
	 * @param Hid number of hidden neurons by layer
	 * @param Out number of output neurons
	 * @throws IllegalArgumentException if one of the structure data is null or negative
	 * 
	 * @see #NeuronNetwork(int[])
	 * @see #NeuronNetwork(int[],float)
	 * @see #NeuronNetwork(int, int[], int,float)
	 * @since 1.0
	 */
	public NeuronNetwork(int In, int[] Hid, int Out) {
		this(In,Hid,Out,1f);
	}

	//like the function addNode except it return the created neuron, useful for some intern method (and this is why it is private)
	private Neuron addNode_(int Type,int Layer) throws IllegalArgumentException{
		if(Layer<0)
			throw new IllegalArgumentException("Layer must be positive");
		if(Type == Input && Layer != 0)
			throw new IllegalArgumentException("Inputs must be on layer 0");
		if(Type == Output && Layer != Layers.size()-1)
			throw new IllegalArgumentException("Output must be on last layer");
		if(Type == Input)
			inputLength++;
		if(Type == Output)
			outputLength++;
		totalNeurons++;
		
		if(Layers.size()<Layer)
			throw new IllegalArgumentException("Layer too high");
		Neuron n = new Neuron(Type,Layer);
		int output = n.Id;
    	Layers.get(Layer).add(n);
    	return n;
	}
	
	/**
	 * add a neuron in the network
	 * @param Type the type of the neuron
	 * @param Layer the layer of the Neuron
	 * @throws IllegalArgumentException if the layer have incorrect value or if input or output neuron are placed in the wrong layer
	 * 
	 * @return the new Neuron Id
	 * 
	 * @see Neuron#Neuron(int, int)
	 * @since 1.0
	 */
	public int addNode(int Type,int Layer) {
		return(addNode_(Type,Layer).Id);
	}
	
	public void addNode(Neuron n) throws IllegalArgumentException {
		for(ArrayList<Neuron> layer:Layers)
			for(Neuron N:layer)
				if(N.Id==n.Id)
					throw new IllegalArgumentException("a neuron with the same id already exist");
		
		if(n.layer>Layers.size())
			throw new IllegalArgumentException("neuron layer is too high");
		Neuron N=addNode_(n.type,n.layer);
		N.Id=n.Id;
		nextId=Math.max(nextId,n.Id+1);//to avoid Id error for future neuron
		
	}

	/**
	 * 
	 * pas sur de ça
	public void InsertLayer(int layer,int neuron,float Integrity) {
		Layers.add(layer, new ArrayList<Neuron>());
		for(int i=0;i<neuron;i++)
			Layers.get(layer).add(new Neuron(Hidden,layer));
		
	}
	
	public void InsertLayer(int layer, int neuron) {
		InsertLayer(layer,neuron,1);
	}
	**/

	/**
	 * remove the Neuron with the corresponding Id
	 * @param Id the Id of the Neuron to Delete
	 * @since 1.0
	 */
	public void removeNode(int Id){

		for(ArrayList<Neuron> layer:Layers)
			for(Neuron N:layer) {
				for(int i=0;i<N.Synapses.size();i++)
					if(N.Synapses.get(i).parent.Id==Id) {
						N.Synapses.remove(i);i--;totalSynapses--;
					}
				for(int i=0;i<N.BackSynapses.size();i++)
					if(N.BackSynapses.get(i).parent.Id==Id) {
						N.BackSynapses.remove(i);i--;totalSynapses--;
					}
			}
		
		for(ArrayList<Neuron> layer:Layers)
			for(int i=0;i<layer.size();i++)
				if(layer.get(i).Id==Id) {
					if(layer.get(i).type == Input)
						inputLength--;
					if(layer.get(i).type == Output)
						outputLength--;
					totalNeurons--;
					
					layer.remove(i);
					return;
				}
	}
	
	/**
	 * 
	 * create a connection between two neuron  
	 * 
	 * @param Id1 Id of parent Neuron
	 * @param Id2 Id of children Neuron
	 * @param Weight the Synapse Weight
	 * @throws NullPointerException if the Neuron does not exist in the current Network
	 * 
	 * @see #addConnection(Neuron,Neuron)
	 * @since 1.1
	 */
	public void addConnection(int Id1,int Id2,float Weight)throws NullPointerException {
		addConnection(Id1,Id2,Weight,true);
	}
	
	/**
	 * 
	 * create a connection between two neuron  
	 * 
	 * @param Id1 Id of parent Neuron
	 * @param Id2 Id of children Neuron
	 * @param Weight the Synapse Weight
	 * @param enabled the Synapse state
	 * @throws NullPointerException if the Neuron does not exist in the current Network
	 * 
	 * @see #addConnection(Neuron,Neuron)
	 * @since 1.1
	 */
	public void addConnection(int Id1,int Id2,float Weight,boolean enabled)throws NullPointerException {
		Neuron N1=null,N2=null;
		for(ArrayList<Neuron> layer:Layers)
			for(Neuron N:layer) {
				if(Id1==N.Id)
					N1=N;
				if(Id2==N.Id)
					N2=N;
				if(N1!=null && N2!=null) {
					addConnection(N1,N2,Weight,enabled);
					return;
				}
			}
		if(N1==null || N2==null)
			throw new NullPointerException("there are no neuron with this Id "+N1+" "+N2);
	}

	/**
	 * 
	 * create a connection between two neuron  
	 * 
	 * @param Id1 Id of parent Neuron
	 * @param Id2 Id of children Neuron
	 * @throws NullPointerException if the Neuron does not exist in the current Network
	 * 
	 * @see #addConnection(Neuron,Neuron)
	 * @since 1.0
	 */
	public void addConnection(int Id1,int Id2){
		addConnection(Id1,Id2,R.nextFloat()*weightDistrib-(weightDistrib/2));
	}

	/**
	 * 
	 * create a connection between two neuron 
	 * 
	 * @param Parent the Parent Neuron
	 * @param Children the Children Neuron
	 * @throws IllegalArgumentException if the connection already exist or if the neurons are equal
	 * @throws NullPointerException if the Neuron does not exist
	 * 
	 * @see #addConnection(int, int)
	 * @since 1.0
	 */
	public void addConnection(Neuron Parent,Neuron Children) {
		addConnection(Parent,Children,R.nextFloat()*weightDistrib-(weightDistrib/2),true);
	}
	
	/**
	 * 
	 * create a connection between two neuron 
	 * 
	 * @param Parent the Parent Neuron
	 * @param Children the Children Neuron
	 * @param Weight the Synapse Weight
	 * @throws IllegalArgumentException if the connection already exist or if the neurons are equal
	 * @throws NullPointerException if the Neuron does not exist
	 * 
	 * @see #addConnection(int, int)
	 * @since 1.1
	 */
	public void addConnection(Neuron Parent,Neuron Children,float Weight) {
		addConnection(Parent,Children,Weight,true);
	}
	
	/**
	 * 
	 * create a connection between two neuron 
	 * 
	 * @param Parent the Parent Neuron
	 * @param Children the Children Neuron
	 * @param Weight the Synapse Weight
	 * @param enabled the Synapse state
	 * @throws IllegalArgumentException if the connection already exist or if the neurons are equal
	 * @throws NullPointerException if the Neuron does not exist
	 * 
	 * @see #addConnection(int, int)
	 * @since 1.1
	 */
	public void addConnection(Neuron Parent,Neuron Children,float Weight,boolean enabled)throws IllegalArgumentException,NullPointerException{
		
		if(Parent==Children)
			throw new IllegalArgumentException("A Neuron cannot be conncted to itself");
		for(Synapse S:Parent.BackSynapses)
			if(S.children==Children)
				throw new IllegalArgumentException("the connection already exist");
		for(Synapse S:Children.Synapses)
			if(S.parent==Parent)
				throw new IllegalArgumentException("the connection already exist");
		
		if(Children.type==Input || Children.type==Bias)
			throw new IllegalArgumentException("the Inputs or bias nodes cannot be children of another neuron");

		if(Parent.type==Output)
			throw new IllegalArgumentException("the output nodes cannot be parent of another neuron");
		
		if(Parent.layer==Children.layer)
			throw new IllegalArgumentException("cannot link two neuron of the same layer");
		
		boolean Ok = false;
		for(Neuron N:Layers.get(Parent.layer))
			if(Parent==N)
				Ok=true;
		if(!Ok)
			throw new NullPointerException("the Parent neuron does not exist");
		Ok=false;
		for(Neuron N:Layers.get(Children.layer))
			if(Children==N)
				Ok=true;
		if(!Ok)
			throw new NullPointerException("the Children neuron does not exist");
		
		if(Parent.layer>Children.layer)
			throw new IllegalArgumentException("cannot link a neuron from a higher layer to a lower layer");

		//maybe add some stuff to manage extending layers size to create new neuron
		//also note that you're not supposed to build your network by hand but with a genetic algorithm
		
		totalSynapses++;
		Synapse S = new Synapse(Parent,Children);
		S.weight=Weight;
		S.enabled=true;
		Parent.BackSynapses.add(S);
		Children.Synapses.add(S);
	}
	
	
	/**
	 * 
	 * create a connection between two Neuron
	 * 
	 * the Neuron position is indicate with a set of two value <br>
	 * the first indicate the layer while the second indicate the neuron position in this layer
	 * 
	 * @param parentCoord 
	 * @param childrenCoord
	 * 
	 * @throws NullPointerException if the Neuron does not exist
	 * 
	 * @see #addConnection(int, int)
	 * @see #addConnection(Neuron, Neuron)
	 * @since 1.0
	 */
	public void addConnection(int[] parentCoord,int[] childrenCoord) throws NullPointerException {
		if(parentCoord[0]>Layers.size() || childrenCoord[0]>Layers.size())
			throw new NullPointerException("Coord out of range");
		if(parentCoord[1]>Layers.get(parentCoord[0]).size() || childrenCoord[1]>Layers.get(childrenCoord[0]).size())
			throw new NullPointerException("Coord out of range");
		
		totalSynapses++;
		Synapse S = new Synapse(Layers.get(parentCoord[0]).get(parentCoord[1]),Layers.get(childrenCoord[0]).get(childrenCoord[1]));
		Layers.get(parentCoord[0]).get(parentCoord[1]).BackSynapses.add(S);
		Layers.get(childrenCoord[0]).get(childrenCoord[1]).Synapses.add(S);
	}
	

	/**
	 * remove the connection between two Neuron
	 * @param Parent the parent Neuron
	 * @param Children the Children Neuron
	 * 
	 * @see #removeConnection(int, int)
	 * @see #removeConnection(int[], int[])
	 * 
	 * @since 1.0
	 */
	public void removeConnection(Neuron Parent,Neuron Children){
		for(int i=0;i<Parent.BackSynapses.size();i++)
			if(Parent.BackSynapses.get(i).children.Id==Children.Id) {
				Parent.BackSynapses.remove(i);
				break;
			}
		for(int i=0;i<Children.Synapses.size();i++)
			if(Children.Synapses.get(i).parent.Id==Parent.Id) {
				Children.Synapses.remove(i);
				break;
			}
		
		for(int i=0;i<disabledSyn.size();i++)
			if(Children.Synapses.get(i).parent.Id==Parent.Id) {
				Children.Synapses.remove(i);
				break;
			}
		
		totalSynapses--;
	}
	
	/**
	 * 
	 * remove a connection between two neuron  
	 * 
	 * @param Id1 Id of parent Neuron
	 * @param Id2 Id of children Neuron
	 * @throws NullPointerException if the Neuron does not exist in the current Network
	 * 
	 * @see #addConnection(Neuron,Neuron)
	 * @see #removeConnection(Neuron, Neuron)
	 * @see #removeConnection(int[], int[])
	 * @since 1.1
	 */
	public void removeConnection(int ParentId,int ChildrenId) {
		Neuron N1=null,N2=null;
		for(ArrayList<Neuron> layer:Layers)
			for(Neuron N:layer) {
				if(ParentId==N.Id)
					N1=N;
				if(ChildrenId==N.Id)
					N2=N;
				if(N1!=null && N2!=null) {
					removeConnection(N1,N2);
					return;
				}
			}
		if(N1==null || N2==null)
			throw new NullPointerException("there are no neuron with this Id "+N1+" "+N2);
	}

	/**
	 * 
	 * remove a connection between two neuron  
	 * 
	 * @param parentCoord the parent coordinate in the network
	 * @param childrenCoord the children coordinate in the network
	 * 
	 * @throws NullPointerException if one of the Neuron does not exist
	 * 
	 * @see #addConnection(Neuron,Neuron)
	 * @see #removeConnection(Neuron, Neuron)
	 * @see #removeConnection(int, int)
	 * @since 1.1
	 */
	public void removeConnection(int[] parentCoord, int[] childrenCoord) {
		if(parentCoord[0]>Layers.size() || childrenCoord[0]>Layers.size())
			throw new NullPointerException("Coord out of range");
		if(parentCoord[1]>Layers.get(parentCoord[0]).size() || childrenCoord[1]>Layers.get(childrenCoord[0]).size())
			throw new NullPointerException("Coord out of range");
		
		totalSynapses++;
		removeConnection(Layers.get(parentCoord[0]).get(parentCoord[1]),Layers.get(childrenCoord[0]).get(childrenCoord[1]));
	}

	/**
	 * disable the Synapse S
	 * @param S the Synapse to disable
	 * @since 1.1
	 * @see #enableConnection(Synapse)
	 */
	public void disableConnection(Synapse S) {
		if(S.enabled) {
			S.enabled=false;
			disabledSyn.add(S);
		}
	}

	/**
	 * enable the Synapse S
	 * @param S the Synapse to enable
	 * @since 1.1
	 * @see #disableConnection(Synapse)
	 */
	public void enableConnection(Synapse S) {
		if(!S.enabled) {
			S.enabled=true;
			disabledSyn.remove(S);
		}
	}
	
	

	/**
	 * do some random little modification to the neural network to get random new behavior without losing the old ones.
	 * 
	 * this method will call mutateWeight, mutateNode and mutateConnection
	 * with the following parameter : 
	 * <ul>
	 * <li><b>Weights</b> : 0.8 0.1 WeightDistrib*0.2</li>
	 * <li><b>Connections</b> : 0.05 0.05 0.25 20</li>
	 * <li><b>Nodes</b> : 0.1 0.1</li>
	 * </ul>
	 * 
	 * @since 1.0
	 * 
	 * @see #mutateWeight(float, float, float)
	 * @see #mutateConnection(float, float, float, int)
	 * @see #mutateNode(float, float)
	 */
	public void mutate(){
		mutateWeight(0.8f,0.1f,weightDistrib*0.2f);	
		mutateConnection(0.05f,0.05f,0.25f,20);
		mutateNode(0.1f,0.1f);
	}
	
	/**
	 * mutate the network by adding a node in a synapse (the new neuron become the children of the inner neuron and the parent of the outetr neuron)
	 * or by removing a random neuron.
	 * <br>
	 * note that bias, output and input neuron cannot be removed or created.
	 * @param probNew probability of new neuron
	 * @param probMerge probability of removing a neuron
	 * @since 1.1
	 * @see #mutate()
	 */
	public void mutateNode(float probNew,float probMerge) throws IllegalArgumentException{
		if(probNew<0 || probNew>1)throw new IllegalArgumentException("probability(probNew) must be between 0 and 1");
		if(probMerge<0 || probMerge>1)throw new IllegalArgumentException("probability(probMerge) must be between 0 and 1");
		if(R.nextFloat()<probNew && totalSynapses!=0) {
			//take a random Synapse
			int Syn = R.nextInt(totalSynapses);
			int pos=0;
			Synapse S=null;
			out:for(ArrayList<Neuron> layer:Layers)
				for(Neuron N:layer) {
					pos+=N.BackSynapses.size();
					if(pos>Syn) {
						Syn -= pos;
						Syn += N.BackSynapses.size();
						for(Synapse Sy:N.BackSynapses) {
							Syn--;
							if(Syn<=0) {
								S=Sy;
								break out;
							}
						}
					}
				}
			
			//disable Synapse
			disableConnection(S);
			//System.out.println(S.parent.Id+" "+S.children.Id);
			
			//then update structure 
			//don't use Normalized because its a lot of useless computation
			correctLayer(S.children,S.children.layer+1);
			
			Neuron newNeuron = addNode_(Hidden,S.parent.layer+1);
			addConnection(S.parent,newNeuron,S.weight);
			addConnection(newNeuron,S.children);
			fixLayer();//to fix issue like having hidden in output layer or vice versa
		}
		

		if(R.nextFloat()<probMerge && totalSynapses!=0 && (totalNeurons-inputLength-outputLength-1)!=0){
			Neuron N=randomHidden();
			//connect the parent neuron to a random children from the output synapses (if the connection doesn't exist)
			//this allow to keep the connection between the input and output neuron while getting rid of the hidden neuron allowing to reduce the useless complexity that can occurs
			//it also avoid removing all connection by accident (if there was only 1 hidden neuron left for example)
			for(Synapse S:N.Synapses) {
				try {
					addConnection(S.parent,N.BackSynapses.get(R.nextInt(N.BackSynapses.size())).children,S.weight);
				}catch(Exception e) {
					//don't care
				}
			}
			removeNode(N.Id);
		}
	}
	
	private void correctLayer(Neuron N,int newLayer) {
		if(N.type==Output && Layers.size()-1>newLayer) {//ignore new layer and put it at the higher existing layer
			Layers.get(N.layer).remove(N);
			N.layer=Layers.size()-1;
			Layers.get(Layers.size()-1).add(N);
			return;
		}
		while(Layers.size()<=newLayer)//add layers if they didn't exist
			Layers.add(new ArrayList<Neuron>());
		Layers.get(N.layer).remove(N);
		N.layer=newLayer;
		Layers.get(newLayer).add(N);//updateLayer
		for(Synapse S:N.BackSynapses)
			if(S.children.layer<=newLayer)
				correctLayer(S.children);//correct chidren layer
	}
	
	private void correctLayer(Neuron N) {
		int newLayer=N.layer;//check if all parent neuron are in lower layer, if not update the layer
		for(Synapse S:N.Synapses)
			if(S.parent.layer>=newLayer)
				newLayer=S.parent.layer+1;
		if(newLayer!=N.layer)//correct layer only if its needed
			correctLayer(N,newLayer);
	}
	
	
	//remove empty layer and replace all output on last layer
	private void fixLayer() {
		//if there are hidden in the output layer we add a new layer for the output;
		for(Neuron N:Layers.get(Layers.size()-1)) {
			if(N.type!=Output) {
				Layers.add(new ArrayList<Neuron>());
				break;
			}
		}
		//we move all output to the last layer
		for(int i=1;i<Layers.size()-1;i++)
			for(int j=0;j<Layers.get(i).size();j++)
				if(Layers.get(i).get(j).type==Output) {
					Layers.get(i).get(j).layer=Layers.size()-1;
					Layers.get(Layers.size()-1).add(Layers.get(i).get(j));
					Layers.get(i).remove(j);
					j--;
				}
		
		//we remove all empty layer
		int offset=0;
		for(int i=1;i<Layers.size();i++)
			if(Layers.get(i).isEmpty()) {
				Layers.remove(i);i--;offset++;
			}else
				for(Neuron N:Layers.get(i))
					N.layer-=offset;

		
	}
	
	//take a random neuron in the network
	private Neuron randomNeuron() {
		int pos= R.nextInt(totalNeurons);
		for(ArrayList<Neuron> layer:Layers) {
			if(pos-layer.size()<=0)
				for(Neuron N:layer) {
					pos--;
					if(pos==0)
						return N;
				}
			pos-=layer.size();
		}
		return null;
	}
	
	//take a random hidden neuron in the network
	private Neuron randomHidden() {
		int pos= ((totalNeurons-inputLength-outputLength-1==1)?0:R.nextInt(totalNeurons-inputLength-outputLength-2))+inputLength+2;
		for(ArrayList<Neuron> layer:Layers) {
			if(pos-layer.size()<=0)
				for(Neuron N:layer) {
					pos--;
					if(pos==0)
						return N;
				}
			pos-=layer.size();
		}
		return null;
	}

	/**
	 * will try to add, remove, or activate connection
	 * @param probNew probability of new connection
	 * @param probDel probability of removing connection
	 * @param probAct probability of activating a connection
	 * @param attempt how many time it will try to create a connection
	 */
	public void mutateConnection(float probNew,float probDel,float probAct,int attempt) throws IllegalArgumentException{
		if(probNew<0 || probNew>1)throw new IllegalArgumentException("probability(probNew) must be between 0 and 1");
		if(probDel<0 || probDel>1)throw new IllegalArgumentException("probability(probDel) must be between 0 and 1");
		if(probAct<0 || probAct>1)throw new IllegalArgumentException("probability(probAct) must be between 0 and 1");
		if(attempt<0)throw new IllegalArgumentException("attempt must be positive");
		if(R.nextFloat()<probNew) {
			for(int i=0;i<attempt;i++){
				try {
					addConnection(randomNeuron(), randomNeuron());
					break;
				}catch(Exception e) {
					continue;
				}
			}
		}
		if(totalSynapses>1) {
			if(R.nextFloat()<probDel) {
				for(int i=0;i<attempt;i++){
					try {
						removeConnection(randomNeuron(), randomNeuron());
						break;
					}catch(Exception e) {
						continue;
					}
				}
			}
		}
		
		if(!disabledSyn.isEmpty() && R.nextFloat()<probAct)//maybe activate neuron
			enableConnection(disabledSyn.get(R.nextInt(disabledSyn.size())));
	}
	/**
	 * 
	 * mutate weight, take some weight and change a little bit the value of the neuron.
	 * it can also reset a connection (take a random value for its weights.
	 * 
	 * @param prob probability of changing a weight value
	 * @param resetProb probability that the change will reset the value (the probability of a connection reset is prob*resetProb)
	 * @param amount how much is weight changed.
	 */
	public void mutateWeight(float prob,float resetProb,float amount) throws IllegalArgumentException{
		if(prob<0 || prob>1)throw new IllegalArgumentException("probability(probDel) must be between 0 and 1");
		if(resetProb<0 || resetProb>1)throw new IllegalArgumentException("probability(probAct) must be between 0 and 1");
		if(amount<0)throw new IllegalArgumentException("attempt must be positive");
		for(ArrayList<Neuron> Layer:Layers)
			for(Neuron N:Layer)
				for(Synapse S:N.Synapses)
					if(R.nextFloat()<prob)
						if(R.nextFloat()<resetProb)
							S.weight=(R.nextFloat()-0.5f)*weightDistrib;
						else
							S.weight+=(R.nextFloat()-0.5f)*amount;
	}
	
	/**
	 * assign inputs to Inputs Neurons
	 * @param data the input data array
	 * @throws ArrayIndexOutOfBoundsException if the data array's size does not match the number of inputs
	 * @since 1.0
	 */
	public void setInputs(float[] data)throws ArrayIndexOutOfBoundsException{
		if(data.length != inputLength)
			throw new ArrayIndexOutOfBoundsException("Data array must have the size of the inputs number : "+data.length+" "+inputLength);
		int i=0;
		for(Neuron N:Layers.get(0))
			if(N.type==Input) {
				N.value=data[i];i++;
			}
	}
	
	/**
	 * compute the output of the Neural network
	 * @param data the data array
	 * @see #compute()
	 * @since 1.0
	 */
	public void compute(float[] data) {
		setInputs(data);
		compute();
	}
	
	/**
	 * compute the output of the Neural network with the data already loaded in the inputs neurons
	 * @see #compute(float[])
	 * @since 1.0
	 */
	public void compute() {
		int layer =1;
		while(layer<Layers.size()) {
			for(Neuron N:Layers.get(layer)) {
				N.compute();
			}
			layer++;
		}
	}
	
	/**
	 * give the outputs in a float array
	 * @return the outputs
	 * @since 1.0
	 */
	public float[] getOutputs(){
		float[] outputs = new float[outputLength];
		int i=0;
		for(Neuron N:Layers.get(Layers.size()-1))
			if(N.type==Output){
				outputs[i]=N.value;i++;
			}
		return outputs;
	}
	
	/**
	 * give the output sorted by their higher value;
	 * @return
	 */
	public int[] maxOutputs() {
		float[] output=getOutputs();
		int[] out = new int[outputLength];
		for(int i=0;i<outputLength;i++)
			out[i]=i;
		quickSort(out,output,0,outputLength-1);
		return out;
	}
	
	//made my own quick sort because i'm not sure playing with mapping and list method are very efficient
	private void quickSort(int[] index,float[] arr,int low,int high){
		if(low<high) {
			int p = partition(index,arr,low,high);
        	quickSort(index,arr,low,p-1);
        	quickSort(index,arr,p+1,high);
		}
	}
	
	private int partition(int[] index,float[] arr, int low, int high) {
		float pivot=arr[high];
		int Ppos=low-1;
		for(int i=low;i<=high-1;i++) {
			if(arr[i]>pivot) {//pour avoir un ordre déroissant (max in first)
				Ppos++;
				swap(index,arr,Ppos,i);
			}
		}
		swap(index,arr,Ppos+1,high);
		return Ppos+1;
	}
	
	private void swap(int[] index,float[] arr, int x, int y) {
		float temp=arr[x];
		arr[x]=arr[y];
		arr[y]=temp;
		int tempI=index[x];
		index[x]=index[y];
		index[y]=tempI;
	}

	/**
	 * 
	 * compute the fitness value of the neural network using its best value.
	 * 
	 * @since 1.0
	 * @param target the wanted output
	 * @return the fitness value for this input
	 */
	private float computeFitness(float[] target){
		//no error check because it's a private methods
		float result=0f;
		
		int i=0;
		for(float output:getOutputs()) {
			result += 1-(target[i]-output)*(target[i]-output);
			i++;
		}
		
		return result/outputLength;
	}

	/**
	 * train the Neural network using the database 
	 * @param data the data array
	 * @param target the wanted output for the data array 
	 * @throws ArrayIndexOutOfBoundsException if the data and target array does not have the same size or one of the data size does not match the inputs/outputs size
	 *
	 * @since 1.0
	 * @see #backPropagation(ArrayList, ArrayList, int)
	 * @see #backPropagation(float[], float[])
	 */
	public void backPropagation(ArrayList<float[]> data,ArrayList<float[]> target) throws ArrayIndexOutOfBoundsException{
		backPropagation(data,target,data.size()*100);
	}
	
	/**
	 * 
	 * train the Neural network using the database 
	 * @param data the data array
	 * @param target the wanted output for the data array 
	 * @param tests the number of test cycle (one test cycle is the size of the database)
	 * @throws ArrayIndexOutOfBoundsException if the data and target array does not have the same size or one of the data size does not match the inputs/outputs size
	 *
	 * @see #backPropagation(ArrayList, ArrayList)
	 * @since 1.0
	 * @see #backPropagation(float[], float[])
	 */
	public void backPropagation(ArrayList<float[]> data,ArrayList<float[]> target,int tests) throws ArrayIndexOutOfBoundsException{
		
		//errors check
		if(data.size()!=target.size())
			throw new ArrayIndexOutOfBoundsException("data and target array must have the same size");
		for(int i=0;i<data.size();i++) {
			if(data.get(i).length != inputLength)
				throw new ArrayIndexOutOfBoundsException("data arrays must have the size of the inputs number");
			if(target.get(i).length != outputLength)
				throw new ArrayIndexOutOfBoundsException("target arrays must have the size of the outputs number");
		}
		
		for(int i=0;i<tests;i++) {
			int index = R.nextInt(data.size());
			//System.out.println(Arrays.toString(data.get(index))+" "+Arrays.toString(target.get(index))); //debug
			backPropagation(data.get(index),target.get(index));
		}
	}
	
	/**
	 * 
	 * update the weight to give outputs closer to the target next time
	 * 
	 * 
	 * @param data the inputs values
	 * @param target the target values
	 * @throws ArrayIndexOutOfBoundsException if the data or target array size does not match the inputs and outputs size
	 *
	 * @since 1.0
	 * @see #backPropagation(ArrayList, ArrayList)
	 * @see #backPropagation(ArrayList, ArrayList, int)
	 */
	public void backPropagation(float[] data,float[] target)throws ArrayIndexOutOfBoundsException{
		
		//errors check
		if(data.length != inputLength)
			throw new ArrayIndexOutOfBoundsException("data arrays must have the size of the inputs number");
		if(target.length != outputLength)
			throw new ArrayIndexOutOfBoundsException("target arrays must have the size of the outputs number");
		
		compute(data);
		
		//seting initial deltas
		int i=0;
		for(Neuron N:Layers.get(Layers.size()-1))
			if(N.type==Output){
				//System.out.println(-N.value+target[i]+" "+Arrays.toString(data)+" "+target[i]);
				N.delta=-N.value+target[i];
				i++;
			}
		
		//backPropagating !
		int layer =Layers.size()-2;
		while(layer>=0) {
			//System.out.println(layer);
			for(Neuron N:Layers.get(layer)) {
				N.updateDelta();
				N.backPropagation();
			}
			layer--;
		}
	}

	/**
	 * test the neural network and add the result to the fitness value
	 * @param data the test inputs
	 * @param target the wanted outputs
	 * @since 1.0
	 * @throws ArrayIndexOutOfBoundsException if the data or target array size does not match the inputs and outputs size
	 */
	public void train(float[] data,float[] target) throws ArrayIndexOutOfBoundsException{
		
		//errors check
		if(data.length != inputLength)
			throw new ArrayIndexOutOfBoundsException("data arrays must have the size of the inputs number");
		if(target.length != outputLength)
			throw new ArrayIndexOutOfBoundsException("target arrays must have the size of the outputs number");
		
		float Fit=0;
		compute(data);
		Fit+=computeFitness(target);
		
		fitness=fitness*tests+Fit;
		tests++;
		fitness/=tests;
	}

	/**
	 * test the neural network and add the result to the fitness value
	 * 
	 * @param data the test ArrayList
	 * @param target the wanted outputs ArrayList
	 * @since 1.0
	 * @throws ArrayIndexOutOfBoundsException if the data or target array size does not match the inputs and outputs size or the size of the data and target arraylist are not the same
	 */
	public void train(ArrayList<float[]> data,ArrayList<float[]> target) throws ArrayIndexOutOfBoundsException{
		//errors check
		if(data.size()!=target.size())
			throw new ArrayIndexOutOfBoundsException("data and target array must have the same size");
		for(int i=0;i<data.size();i++) {
			if(data.get(i).length != inputLength)
				throw new ArrayIndexOutOfBoundsException("data arrays must have the size of the inputs number");
			if(target.get(i).length != outputLength)
				throw new ArrayIndexOutOfBoundsException("target arrays must have the size of the outputs number");
		}
		float Fit=0;
		for(int i=0;i<data.size();i++) {
			compute(data.get(i));
			Fit+=computeFitness(target.get(i));
		}
		
		fitness=fitness*tests+Fit*data.size();
		tests+=data.size();
		fitness/=tests*tests;
	}
	
	/**
	 * delete useless neuron and collapse the network in it minimal structure (by taking the minimum amount of layer)
	 *
	 * useful to compare two similar network that were build differently.
	 * @since 1.0
	 */
	public void normalizeNetwork() {
		
		//delete useless Neurons.
		int deletedNeurons=1;
		while(deletedNeurons!=0){
			deletedNeurons=0;
			for(ArrayList<Neuron> layer:Layers)
				for(int i=0;i<layer.size();i++)
					if(layer.get(i).type==Hidden
					&& layer.get(i).Synapses.isEmpty()){ //if the neuron has no input connection (only for hidden neuron to avoid external error)
						removeNode(layer.get(i).Id);i--;deletedNeurons++;
					}
		}
		
		
		//create temporary array with all neuron
		ArrayList<Neuron> tempNeuron = new ArrayList<Neuron>();
		for(ArrayList<Neuron> layer:Layers)
			for(Neuron N:layer) {
				N.layer=Integer.MAX_VALUE;//useful later
				tempNeuron.add(N);
			}
		
		ArrayList<ArrayList<Neuron>> newLayer = new ArrayList<ArrayList<Neuron>>();
		
		newLayer.add(new ArrayList<Neuron>());
		
		ArrayList<Neuron> Outputs = new ArrayList<Neuron>();
		
		for(int i=0;i<tempNeuron.size();i++) {
			if(tempNeuron.get(i).type==Input||tempNeuron.get(i).type==Bias){
				tempNeuron.get(i).layer=0;
				newLayer.get(0).add(tempNeuron.remove(i));i--;
			}
			else if(tempNeuron.get(i).type==Output) {
				Outputs.add(tempNeuron.remove(i));i--;
			}
		}
		
		int l=1;
		while(tempNeuron.size()!=0) {
			newLayer.add(new ArrayList<Neuron>());
			loop: for(int i=0;i<tempNeuron.size();i++) {
				for(Synapse S:tempNeuron.get(i).Synapses) 
				{
					if(S.parent.layer>l-1) {
						continue loop;
					}
				}
				tempNeuron.get(i).layer=l;
				newLayer.get(l).add(tempNeuron.remove(i));
				i--;//pour compensser le remove
			}
			l++;
		}
		
		
		for(Neuron N:Outputs)
			N.layer=l;
		newLayer.add(Outputs);
		Layers=newLayer; //good
	
	}
	
	/**
	 * Reassign Neuron and Synapse Id
	 * using this method with the normalize method make two network with the same topology almost exactly equal
	 * 
	 * @since 1.1
	 */
	public void reasignId() {
		
		// not sure keeping that is good for the code
		totalSynapses = 0; //to recompute the Synapse number
		//reasign Neurons Id;
		boolean FirstBias=true;
		int currentId=1; //we skip Id 0 because it's the first Bias Id
		for(ArrayList<Neuron> layer:Layers) {
			for(Neuron N:layer)
				if(N.type==Bias && FirstBias) {
					N.Id=0;FirstBias=false;
				}
				else {
					totalSynapses+=N.Synapses.size();
					N.Id=currentId;currentId++;
				}
		}
		nextId=currentId;
		
		//reasign Synapses Id;
		for(ArrayList<Neuron> layer:Layers)
			for(Neuron N:layer)
				for(Synapse S:N.Synapses)
					S.Id=MaxNeurons*S.parent.Id+S.children.Id;
	}
	
	/**
	 * 
	 * compare the structure and the weight of two Neuron Network, 0 mean the networks are exactly the same,
	 * greater value mean the networks are more different
	 * 
	 * @since 1.0
	 * @return the comparison value
	 */
	public float compare(NeuronNetwork NN) {
		//get All Synapses for easier compute
		ArrayList<Synapse> Synapses=new ArrayList<Synapse>();
		for(ArrayList<Neuron> layer:Layers)
			for(Neuron N:layer)
				for(Synapse S:N.Synapses) {
					Synapses.add(S);
				}

		int SynapseError=0;
		float WeightError = 0f;
		int SynapseCommun = 0;
		
		for(ArrayList<Neuron> layer:NN.Layers)
			for(Neuron N:layer)
				loop : for(Synapse S:N.Synapses) {
					for(int i=0;i<Synapses.size();i++)
						if(Synapses.get(i).equals(S)) {
							SynapseCommun++;
							WeightError=Math.abs(S.weight-Synapses.get(i).weight);
							Synapses.remove(i);
							continue loop;
						}
					SynapseError++;
				}
		
		SynapseError=Math.max(this.Complexity(),NN.Complexity())-SynapseCommun;	
		SynapseError/=Math.max(Math.max(this.Complexity(),NN.Complexity()),1);//au cas où il n'y a aucune synapse  dans les deux réseaux...
		if(SynapseCommun==0)
			SynapseError*=(1+WeightFactor/StructureFactor);//to compensate the Weight error which is at 0 but not for the good reason
		else
			WeightError/=SynapseCommun;
		
		return SynapseError*StructureFactor+WeightError*WeightFactor;
	}
	
	/**
	 * return a simple representation of the Neural network
	 * @return a String representation of the Network
	 * 
	 * @since 1.0
	 */
	public String Header() {
		return("<Class : NeuronNetwork | Size : "+totalNeurons+" | Synapses : "+totalSynapses+" >");
	}

	/**
	 * return a complete representation of the Neural network
	 * @return a String representation of the Network
	 * 
	 * @since 1.0
	 */
	public String toString() {
		String output = Header();
		output+="\n Structure : ";
		String data = "\n\nData :\n";
		int i=0;
		for(ArrayList<Neuron> layer:Layers) {
			output+=layer.size()+" ; ";
			data+="\nlayer "+i+" : \n";
			for(Neuron N:layer)
				data+=N.toString()+"\n";
			i++;
		}
		return(output+data);
	}
	
	/**
	 * return an Independent copy of the network
	 * 
	 * I don't know why but the clone function is very sensible to bug, if there are one bug in the neuron network, there are a high chance it will trigger an error in this
	 * 
	 * @return
	 */
	public NeuronNetwork clone() {
		NeuronNetwork output = new NeuronNetwork();
		int i=0;
		for(ArrayList<Neuron> L:Layers) {
			output.Layers.add(new ArrayList<Neuron>());
			for(Neuron N:L) {
				output.addNode(N);
				for(Synapse S:N.Synapses) {
					output.addConnection(S.parent.Id,S.children.Id,S.weight,S.enabled);
				}
			}
			i++;
		}
		
		output.setActivation(activation, derivative, outputActivation, outputDerivative);
		return output;
	}
	
	/**
	 * create a copy of the network and take randomly the weight of the second network if the synapse are the same
	 * @return
	 */
	public NeuronNetwork merge(NeuronNetwork NN) {
		NeuronNetwork output = clone();//get All Synapses for easier compute
		
		ArrayList<Synapse> Synapses=new ArrayList<Synapse>();
		for(ArrayList<Neuron> layer:output.Layers)
			for(Neuron N:layer)
				for(Synapse S:N.Synapses) {
					Synapses.add(S);
				}
		
		for(ArrayList<Neuron> layer:NN.Layers)
			for(Neuron N:layer)
				loop : for(Synapse S:N.Synapses) {
					for(int i=0;i<Synapses.size();i++)
						if(Synapses.get(i).equals(S) && R.nextBoolean()) {
							Synapses.get(i).weight=S.weight;
							continue loop;
						}
				}
		return output;
	}
	
	public int save(String path) throws IOException{
	      DataOutputStream out=new DataOutputStream(new FileOutputStream(new File(path).getAbsoluteFile()));
	      out.writeUTF("NeuronStructure");
	      out.writeInt(totalNeurons);
	      out.writeInt(totalSynapses);
	      out.writeInt(inputLength);
	      out.writeInt(outputLength);
	      out.writeInt(nextId);
	      out.writeInt(Layers.size());
	      //write all neuron
	      for(ArrayList<Neuron> layer:Layers) {
	    	  out.writeInt(layer.size());
	    	  for(Neuron N:layer) {
	    		  out.writeInt(N.Id);
	    		  out.writeByte(N.type);
	    	  }
	      }
	      //write all synapses
	      for(ArrayList<Neuron> layer:Layers) {
	    	  for(Neuron N:layer) {
	    		  for(Synapse S:N.Synapses) {
	    			  out.writeChar('S');
	    			  out.writeInt(S.parent.Id);
	    			  out.writeInt(S.parent.layer);
	    			  out.writeInt(S.children.Id);
	    			  out.writeInt(S.children.layer);
	    			  out.writeBoolean(S.enabled);
	    			  out.writeFloat(S.weight);
	    		  }
	    	  }
	      }

		  out.writeChar('E');
	      out.writeUTF("==>");//end of file
	      out.flush();
	      out.close();
	      return out.size();
	}
	
	public int load(String path) throws IOException {
		File file=new File(path).getAbsoluteFile();
		DataInputStream in=new DataInputStream(new FileInputStream(file));
	    String typeCheck=in.readUTF();
	    if(0!=typeCheck.compareTo("NeuronStructure"))
	    	throw new IOException("wrong file header... error");
	    
	    int TN=in.readInt();
	    int TS=in.readInt();
	    int In=in.readInt();
	    int Out=in.readInt();
	    int NI=in.readInt();
	    
	    int L=in.readInt();
	    
	    int CheckIn=0;
	    int CheckOut=0;
	    
	    ArrayList<ArrayList<Neuron>> newLayer=new ArrayList<ArrayList<Neuron>>();
	    for(int i=0;i<L;i++) {
	    	newLayer.add(new ArrayList<Neuron>());
	    	int layerSize=in.readInt();
		    for(int j=0;j<layerSize;j++){
		    	//System.out.println("load 1");
		    	int Id=in.readInt();
		    	int type=in.readByte();
		    	if(type==Input)
		    		CheckIn++;
		    	if(type==Output)
		    		CheckOut++;
		    	Neuron N=new Neuron(type,i);
		    	N.Id=Id;
		    	nextId--;
		    	newLayer.get(i).add(N);
		    }
	    }
	    if(In!=CheckIn||Out!=CheckOut)
	    	throw new IOException("network input/output neuron doesn't match the header values");
	    
	    while(in.readChar()=='S'){
	    	int PI=in.readInt();
	    	int PL=in.readInt();
	    	int CI=in.readInt();
	    	int CL=in.readInt();
	    	boolean ena=in.readBoolean();
	    	float weight=in.readFloat();
	    	
	    	Neuron Parent=null;
	    	Neuron Children=null;
	    	for(Neuron N:newLayer.get(PL))
	    		if(N.Id==PI)
	    			Parent=N;
	    	for(Neuron N:newLayer.get(CL))
	    		if(N.Id==CI)
	    			Children=N;
	    	Synapse S=new Synapse(Parent,Children);
	    	S.enabled=ena;
	    	S.weight=weight;
	    	Children.Synapses.add(S);
	    	Parent.BackSynapses.add(S);
	    }

	    String endOfFileCheck=in.readUTF();
	    if(0!=endOfFileCheck.compareTo("==>"))
	    	throw new IOException("wrong end of file ... error");
	    
	    //if everything go right : 
	    totalNeurons=TN;
	    totalSynapses=TS;
	    nextId=NI;
	    Layers=newLayer;
	    
	    return (int)file.length();
	}
}
