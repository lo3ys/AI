package neuralnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import neuralnet.NeuralNetwork;
/**
 * Genetic algorithm that adapt a network structure to solve a problem.
 * genetic algorithm inspired by neat AI algorithm
 * @author PhysicDev
 * @see {@link neuralnet.NeuronNetwork}
 * @version 1.1
 *
 */
public class GeneticAlgorithm {
	
	
	//data
	private int gen=0;
	private int pop;
	
	//random variable for random generation
	private Random R = new Random();
	
	//Structure data and info
	//initial Structure for new NeuralNetwork
	private int[] initialStructure;
	private float initialIntegrity;
	private int input;
	private int output;

	//fitness data
	private float fitness=0;
	private float bestFitness=-Float.MAX_VALUE;
	private float adjustedFitness=0;//rework fitness to avoid one species to crush the other
	private NeuronNetwork Best=null;
	
	//network data
	private ArrayList<NeuronNetwork> population = new ArrayList<NeuronNetwork>();//All the population
	private ArrayList<Species> SpeciesArray = new ArrayList<Species>();//the same network as above but distributed in species

	/**
	 * return the actual generation.
	 * @return the actual generation
	 * @since 1.1
	 */
	public int getGen() {
		return gen;
	}
	
	/**
	 * return the amount of network per generation
	 * @return the amount of network per generation
	 * @since 1.1
	 */
	public int getPop() {
		return pop;
	}
	
	/**
	 * use this method to set a custom random variable to the genetic algorithm,
	 * this variable control everything that is randomly generated in the network.
	 * @param r the random variable
	 * @since 1.1
	 */
	public void setRandom(Random r) {
		R=r;
	}
	
	/**
	 * return the average fitness of the whole population
	 * @return the average fitness
	 * @since 1.1
	 */
	public float getFitness() {
		return fitness;
	}
	
	/**
	 * return the fitness of every network in the population in a float array[]
	 * @return the fitness of every network
	 * @since 1.1
	 */
	public float[]getFitnessArray(){
		float[] output = new float[population.size()];
		for(int i=0;i<population.size();i++) {
			output[i]=population.get(i).getFitness();
		}
		return output;
	}
	
	/**
	 * return the fitness of the best network in the actual population</br> i.e. max({@link #getFitnessArray()})
	 * @return the fitness of the best network
	 * @since 1.1
	 */
	public float getBestFitness() {
		return bestFitness;		
	}
	
	/**
	 * return a copy of the best network in the actual population
	 * @return the best network of this generation
	 * @since 1.1
	 */
	public NeuronNetwork getBest() {
		return Best.clone();
	}
	
	/**
	 * return the average adjusted fitness of the whole population.</br>
	 * the adjusted fitness is the fitness divide by the amount of network in the network's species.
	 * this fitness is used to prevent a species to crush the other.
	 * @return the average adjusted fitness
	 * @since 1.1
	 */
	public float getAdjustedFitness() {
		return adjustedFitness;
	}
	
	/**
	 * the wanted amount of species
	 */
	private int speciesTarget=4;
	
	/**
	 * the wanted amount of species is not always the amount of species that we get by generation but the threshold value
	 * is change every generation to get closer to that amount of species.
	 * @return
	 */
	public int getSpeciesTarget() {
		return speciesTarget;
	}
	
	/**
	 * to change the wanted amount of species
	 * @param SpeciesTarget the wanted amount of species 
	 * @throws IllegalArgumentException if SpceciesTarget is negative or null
	 */
	public void setSpeciesTarget(int SpeciesTarget) throws IllegalArgumentException {
		if(speciesTarget<1)
			throw new IllegalArgumentException("SpceisTarget must be a positive not null value");
		speciesTarget=SpeciesTarget;
	}

			/**
			 * the maximum number of generation a species can live without progressing
			 */
	
	private int ImprovementLimit = 20;

	/**
	 * the maximum number of generation a species can live without progressing
	 * if a species doesn't do any progress within this amount of generation.
	 * it's deleted and replace with new network.
	 * @return the improvement limit
	 */
	public int getImprovementLimit() {
		return ImprovementLimit;
	}
	
	/**
	 * to change the maximum number of generation a species can live without progressing
	 * @param IL the maximum number of generation a species can live without progressing
	 * @throws IllegalArgumentException if IL is negative or null
	 */
	public void setImprovementLimit(int IL) throws IllegalArgumentException {
		if(ImprovementLimit<1)
			throw new IllegalArgumentException("IL must be a positive not null value");
		ImprovementLimit=IL;
	}
	
	/**
	 * threshold value indicate how similar two network must be to be in the same species
	 * @since
	 */
	public float threshold=0.5f;
	
	/**
	 * 
	 * @author PhysicDev
	 *
	 */
	private class Species extends ArrayList<NeuronNetwork>{
		/**
		 * average fitness of the species
		 */
		public float fitness=0;
		/**
		 * average adjusted fitness of the species
		 */
		public float adjustedFitness=0;
		/**
		 * total Fitness (not very useful outside of the class)
		 */
		public float totalFitness=0;
		
		/**
		 * generation without improvment in fitness
		 */
		public int lastImprove = 0;
		/**
		 * the best fitness of the species
		 */
		public float BestFitness = -Float.MAX_VALUE;
		/**
		 * the best netowkr of the speices
		 */
		public NeuronNetwork Best;
		
		/**
		 * update the fitness variable see above with the fitness of each network in the spcecies
		 */
		public void updateFitness() {
			totalFitness=0;
			BestFitness=-Float.MAX_VALUE;
			for(NeuronNetwork NN:this) {
				totalFitness+=NN.getFitness();
				if(NN.getFitness()>BestFitness) {
					BestFitness=NN.getFitness();
					Best=NN;
				}
			}
			fitness =totalFitness/this.size();
			adjustedFitness =fitness/this.size();//Adjusted fitness
		}
		
		/**
		 * useless
		 */
		public Species(){
			super();
			//bruh nothing here
		}
		
		/**
		 * return a random Network from the population weighted by their respective fitness
		 * @return a random Network from the population weighted by their respective fitness
		 */
		private NeuronNetwork randomPop() {
			float val = R.nextFloat()*this.totalFitness;
			float sum=0;
			int i=0;
			while(val<sum) {
				sum+=this.get(i).getFitness();
				i++;
			}
			return this.get(i);
		}
		
		/**
		 * Create a children from this species by merging two random parents selected by {@link #randomPop()}
		 * @return the children network
		 */
		public NeuronNetwork CreateChildren() {
			NeuronNetwork child=null;
			
			//get the two parent
			NeuronNetwork Parent1 = randomPop();
			NeuronNetwork Parent2 = randomPop();
			if(Parent1.getFitness()>Parent2.getFitness())
				child = Parent1.merge(Parent2);
			else 
				child = Parent2.merge(Parent1);
			return child;
		}
	}
	
	/**
	 * return the amount of species
	 * @return the amount of species
	 */
	public int getDiversity() {
		return SpeciesArray.size();
	}
	
	/**
	 * @return the input size of the network
	 */
	public int getInput() {
		return input;
	}
	/**
	 * @return the output size of the network
	 */
	public int getOutput() {
		return output;
	}
	
	/**
	 * generate the first generation of the genetic algorithm.
	 * @param Pop the amount of network by generation
	 * @param In the network number of input
	 * @param Hid the network hidden structure (fully connected)
	 * @param Out the network number of output
	 * @param Integrity the amount of connection kept from the fully connected network associated to this structure
	 * 
	 * @since 1.1
	 * @see #GeneticAlgorithm(int, int[])
	 * @see #GeneticAlgorithm(int, int[], float)
	 * @see #GeneticAlgorithm(int, int, int[], int)
	 */
	public GeneticAlgorithm(int Pop,int In,int[] Hid,int Out,float Integrity) {
		pop=Pop;
		initialIntegrity=Integrity;
		initialStructure = new int[2+Hid.length];
		for(int i=0;i<Hid.length;i++)
			initialStructure[1+i]=Hid[i];
		initialStructure[0]=In;
		initialStructure[1+Hid.length]=Out;
		for(int i=0;i<pop;i++) {
			NeuronNetwork NN=new NeuronNetwork(In,Hid,Out,Integrity);
			NN.normalizeNetwork(); 
			population.add(NN);
		}
	}
	
	/**
	 * generate the first generation of the genetic algorithm.
	 * @param Pop the amount of network by generation
	 * @param Structure the Structure of the network (fully connected)
	 * @param Integrity the amount of connection kept from the fully connected network associated to this structure
	 * 
	 * @since 1.1
	 * @see #GeneticAlgorithm(int, int[])
	 * @see #GeneticAlgorithm(int, int, int[], int)
	 * @see #GeneticAlgorithm(int, int, int[], int, float)
	 */
	public GeneticAlgorithm(int Pop,int[]Structure,float Integrity) {
		this(Pop,Structure[0],Arrays.copyOfRange(Structure, 1, Structure.length-1),Structure[Structure.length-1],Integrity);
	}

	/**
	 * generate the first generation of the genetic algorithm.
	 * @param Pop the amount of network by generation
	 * @param Structure the Structure of the network (fully connected)
	 * 
	 * @since 1.1
	 * @see #GeneticAlgorithm(int, int[], float)
	 * @see #GeneticAlgorithm(int, int, int[], int)
	 * @see #GeneticAlgorithm(int, int, int[], int, float)
	 */
	public GeneticAlgorithm(int Pop,int[] Structure) {
		this(Pop,Structure,1f);
	}

	/**
	 * generate the first generation of the genetic algorithm.
	 * @param Pop the amount of network by generation
	 * @param In the network number of input
	 * @param Hid the network hidden structure (fully connected)
	 * @param Out the network number of output
	 * 
	 * @since 1.1
	 * @see #GeneticAlgorithm(int, int[])
	 * @see #GeneticAlgorithm(int, int[], float)
	 * @see #GeneticAlgorithm(int, int, int[], int, float)
	 */
	public GeneticAlgorithm(int Pop,int In,int[] Hid,int Out) {
		this(Pop,In,Hid,Out,1f);
	}
	
	
	/**
	 * Speciate the population ie distribute the network in category depending of their similarities.
	 * this method is different from {@link #speciate()} because it care of the last Species array to generate the next one.
	 * @see #speciate()
	 * @see #speciate(float)
	 * @see #forceSpeciate(int)
	 */
	private void speciateArr() {
		//create copy of the Neural network because we gonna remove value
		ArrayList<NeuronNetwork> temp = (ArrayList<NeuronNetwork>) population.clone();
		ArrayList<Species> nextSpeciesArray = new ArrayList<Species>();
		
		//we recreate each Species
		for(Species S:SpeciesArray) {
			int specRefId = R.nextInt(S.size());
			NeuronNetwork specRef = S.get(specRefId);
			temp.remove(specRef);
			nextSpeciesArray.add(new Species());
			if(S.fitness>S.BestFitness) {
				nextSpeciesArray.get(nextSpeciesArray.size()-1).BestFitness=S.fitness;
				nextSpeciesArray.get(nextSpeciesArray.size()-1).lastImprove=0;
			}else {
				nextSpeciesArray.get(nextSpeciesArray.size()-1).BestFitness=S.BestFitness;
				nextSpeciesArray.get(nextSpeciesArray.size()-1).lastImprove=S.lastImprove+1;
			}
			for(int i=0;i<temp.size();i++)
				if(specRef.compare(temp.get(i))<threshold) {
					nextSpeciesArray.get(nextSpeciesArray.size()-1).add(temp.get(i));
					temp.remove(i);
					i--;
				}
			if(nextSpeciesArray.get(nextSpeciesArray.size()-1).isEmpty())
				nextSpeciesArray.remove(nextSpeciesArray.size()-1);
		}
		
		//create new species to replace the old ones
		while(temp.size()!=0) {
			//create new SpeciesArray
			nextSpeciesArray.add(new Species());
			//take random neural network among the non selected ones
			//it will be the reference of the new SpeciesArray
			int specRefId = R.nextInt(temp.size());
			NeuronNetwork specRef = temp.get(specRefId);
			//add the new reference
			nextSpeciesArray.get(nextSpeciesArray.size()-1).add(temp.get(specRefId));
			temp.remove(specRefId);
			//compare network and add similar network
			for(int i=0;i<temp.size();i++)
				if(specRef.compare(temp.get(i))<threshold) {
					nextSpeciesArray.get(nextSpeciesArray.size()-1).add(temp.get(i));
					temp.remove(i);
					i--;
				}
		}
		SpeciesArray=nextSpeciesArray;//replace the array
	}

	/**
	 * Speciate the population ie distribute the network in category depending of their similarities.
	 *
	 * @see #speciate()
	 * @see #speciate(float)
	 * @see #forceSpeciate(int)
	 */
	public void speciate() {
		speciate(threshold);
	}
	
	/**
	 * Speciate the population ie distribute the network in category depending of their similarities.
	 *
	 * @param threshold indicate how similar two network must be to be in the same species
	 */
	public void speciate(float threshold) {
		//create copy of the Neural network because we gonna remove value
		ArrayList<NeuronNetwork> temp = (ArrayList<NeuronNetwork>) population.clone();
		SpeciesArray = new ArrayList<Species>();
		while(temp.size()!=0) {
			//create new SpeciesArray
			SpeciesArray.add(new Species());
			//take random neural network among the non selected ones
			//it will be the reference of the new SpeciesArray
			int specRefId = R.nextInt(temp.size());
			NeuronNetwork specRef = temp.get(specRefId);
			//add the new reference
			SpeciesArray.get(SpeciesArray.size()-1).add(temp.get(specRefId));
			temp.remove(specRefId);
			//compare network and add similar network
			for(int i=0;i<temp.size();i++)
				if(specRef.compare(temp.get(i))<threshold) {
					SpeciesArray.get(SpeciesArray.size()-1).add(temp.get(i));
					temp.remove(i);
					i--;
				}
		}
	}
	
	/**
	 * not very useful, personally i don't use it.
	 * 
	 * take random Neural Networks to be the reference of their SpeciesArray
	 * then sort the remaining Neural Network in the SpeciesArray that is the closest to him (using the compare function on the reference)
	 * @param SpeciesArrayAmount the wanted number of species
	 */
	public void forceSpeciate(int SpeciesArrayAmount) {
		//create copy of the Neural network because we gonna remove value
		ArrayList<NeuronNetwork> temp = (ArrayList<NeuronNetwork>) population.clone();
		SpeciesArray=new ArrayList<Species>(SpeciesArrayAmount);
		for(int i=0;i<SpeciesArrayAmount;i++) {
			int specRefId = R.nextInt(temp.size());
			NeuronNetwork specRef = temp.get(specRefId);
			SpeciesArray.add(new Species());
			SpeciesArray.get(i).add(specRef);
			temp.remove(specRefId);
		}
		
		//sort the remaining network
		for(NeuronNetwork NN:temp) {
			float min=Float.MAX_VALUE;
			int SpecPos = -1;
			for(int i=0;i<SpeciesArrayAmount;i++) {
				float dif=NN.compare(SpeciesArray.get(i).get(0));
				if(dif<min) {
					min=dif;
					SpecPos=i;
				}
			}
			SpeciesArray.get(SpecPos).add(NN);
		}
	}
	
	/**
	 * compute the next generation of neuralNetwork
	 */
	public void nextGen() {
		gen++;
		
		//update threshold
		if(SpeciesArray.size()<speciesTarget) 
			threshold-=0.05;
		else
			threshold+=0.05;
		
		//look for best Network
		float Fit  = -Float.MAX_VALUE;
		NeuronNetwork Best = null;
		int SpecBest = -1;
		for(int i=0;i<SpeciesArray.size();i++)
			for(NeuronNetwork NN:SpeciesArray.get(i))
				if(Fit<NN.getFitness()) {
					Fit=NN.getFitness();
					Best=NN;
					SpecBest=i;					
				}
		
		//compute new species.
		float Fsum = 0;
		int Isum = 0;
		population=new ArrayList<NeuronNetwork>();
		for(int i=0;i<SpeciesArray.size();i++) {			
			Fsum += (SpeciesArray.get(i).adjustedFitness/adjustedFitness*SpeciesArray.get(i).size()); // to solve rounding problems
			int nextSize = (int)Fsum-Isum;
			Isum=(int)Fsum;
			for(int j=0;j<nextSize;j++) {
				if(j==0 && i==SpecBest) {
					//if this is the best species we keep the best member of this species
					population.add(Best.clone());
					continue;
				}
				if(SpeciesArray.get(i).Best.Complexity()==0 || SpeciesArray.get(i).lastImprove>ImprovementLimit) {
					//if the species is not progressing or if it's in a soft lock configuration (as 0 complexity network for example), then we generate new networks
					NeuronNetwork NN=new NeuronNetwork(initialStructure,initialIntegrity);
					NN.normalizeNetwork(); 
					population.add(NN);
				}
				 
				NeuronNetwork newPop = SpeciesArray.get(i).CreateChildren();
				newPop.mutate();
				population.add(newPop);
			}
		}
		speciateArr();//respeciate for the next generation
	}
	
	/**
	 * test all neuralNetwork (evaluate their fitness)
	 * then update the species update (you must call this method before using nextGen)
	 * @param data the database of tests
	 * @param target the wanted result value for this tests
	 */
	public void tests(ArrayList<float[]> data,ArrayList<float[]> target) {
		fitness=0;
		adjustedFitness=0;
		bestFitness=-Float.MAX_VALUE;
		for(NeuronNetwork NN : population) {
			NN.resetFitness();
			NN.train(data, target);
			fitness+=NN.getFitness();
		}
		for(Species S:SpeciesArray) {
			S.updateFitness();
			for(NeuronNetwork NN:S)
				adjustedFitness+=NN.getFitness()/S.size();
			if(bestFitness<=S.BestFitness) {
				Best=S.Best;
				bestFitness=S.BestFitness;
			}
		}
		adjustedFitness/=pop;
		fitness/=pop;
	}
	

	/**
	 * do the same as above except it also train the network (so the algorithm don't wander randomly to find good weights through generation and can learn a bit with time)
	 * @param data the database of tests
	 * @param target the wanted result value for this tests
	 * @param testCycle number of tests (the total number of test is S*testCycle where S is the size of the database of tests)
	 */
	public void train(ArrayList<float[]> data,ArrayList<float[]> target,int testCycle) {
		fitness=0;
		adjustedFitness=0;
		for(NeuronNetwork NN : population) {
			NN.resetFitness();
			NN.backPropagation(data, target,data.size()*testCycle);
			NN.train(data, target);
			fitness+=NN.getFitness();
		}
		for(Species S:SpeciesArray) {
			S.updateFitness();
			adjustedFitness+=S.adjustedFitness*S.size();
		}
		adjustedFitness/=pop;
		fitness/=pop;
	}
	
	/**
	 * 
	 * @return a compact string representation of the genetic algorithm
	 */
	public String toString() {
		return("<Class : Genetic Algorithm | generation : "+gen+" | population : "+pop+">\n");
	}
	
	/**
	 * @return a string representation of the genetic algorithm with every neural network in the actual generation
	 */
	public String longInfo() {
		String output = toString();
		for(NeuronNetwork NN:population) {
			output+=NN.toString();
		}
		return output;
	}
	
	/**
	 * give info about every species with all the network in the species
	 * @return info about every species with all the network in the species
	 */
	public String infoSpeciesArray() {
		String output="<SpeciesArray Info | Number of Species : "+SpeciesArray.size()+" >\n\n";
		int i=1;
		for(ArrayList<NeuronNetwork> Spec:SpeciesArray) {
			output+="<SpeciesArray "+i+" population : "+Spec.size()+" >\n\n"+Spec.get(0).toString();i++;
			}
		return output;
	}
	
	/**
	 * same as {@link #infoSpeciesArray()} but without the network info
	 * @return info about the species
	 */
	public String infoSpeciesArrayShort() {
		String output="<SpeciesArray Info | Number of Species : "+SpeciesArray.size()+" >\n\n";
		int i=1;
		for(ArrayList<NeuronNetwork> Spec:SpeciesArray) {
			output+="<SpeciesArray "+i+" population : "+Spec.size()+" >\n\n";i++;
			}
		return output;
		
	}
	
}
