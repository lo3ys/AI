package NeuralNet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import NeuralNet.NeuralNetwork;
/**
 * genetic algorithm inspired by neat AI video
 * @author utilisateur
 *
 */
public class GeneticAlgorithm {
	
	public static final int ImprovementLimit = 20;
	
	private int gen=0;
	public int getGen() {
		return gen;
	}
	
	private int pop;
	public int getPop() {
		return pop;
	}
	
	private Random R = new Random();
	public void setRandom(Random r) {
		R=r;
	}
	
	//initial Structure for new NeuralNetwork
	private int[] initialStructure;
	private float initialIntegrity;
	
	private float fitness=0;
	public float getFitness() {
		return fitness;
	}
	public float[]getFitnessArray(){
		float[] output = new float[population.size()];
		for(int i=0;i<population.size();i++) {
			output[i]=population.get(i).getFitness();
		}
		return output;
	}
	
	private float bestFitness=-Float.MAX_VALUE;
	public float getBestFitness() {
		return bestFitness;		
	}
	
	private NeuronNetwork Best=null;
	public NeuronNetwork getBest() {
		return Best.clone();
	}
	
	private float adjustedFitness=0;
	public float getAdjustedFitness() {
		return adjustedFitness;
	}
	
	public int speciesTarget=4;
	public float threshold=0.5f;
	
	private class Species extends ArrayList<NeuronNetwork>{
		
		public float fitness=0;
		public float adjustedFitness=0;
		public float totalFitness=0;
		public int lastImprove = 0;
		public float BestFitness = -Float.MAX_VALUE;
		public NeuronNetwork Best;
		
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
		
		public Species(){
			super();
			//bruh nothing here
		}
		
		/**
		 * return a random Network from the population weighted by their respective fitness
		 * @return
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
	
	
	private ArrayList<NeuronNetwork> population = new ArrayList<NeuronNetwork>(); 
	private ArrayList<Species> SpeciesArray = new ArrayList<Species>();
	public int getDiversity() {
		return SpeciesArray.size();
	}
	
	private int input;
	public int getInput() {
		return input;
	}
	private int output;
	public int getOutput() {
		return output;
	}
	
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
			NN.setId(i);
			population.add(NN);
		}
	}
	
	public GeneticAlgorithm(int Pop,int[]Structure,float Integrity) {
		this(Pop,Structure[0],Arrays.copyOfRange(Structure, 1, Structure.length-1),Structure[Structure.length-1],Integrity);
	}
	
	public GeneticAlgorithm(int Pop,int[] Structure) {
		this(Pop,Structure,1f);
	}
	
	public GeneticAlgorithm(int Pop,int In,int[] Hid,int Out) {
		this(Pop,In,Hid,Out,1f);
	}
	
	
	/**
	 * 
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

	public void speciate() {
		speciate(threshold);
	}
	
	/**
	 * speciate BRUH
	 * 
	 * @param threshold
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
	 * take random Neurals Networks to be the reference of their SpeciesArray
	 * then sort the remaining Neural Network in the SpeciesArray that is the closest to him (using the compare functio on the reference)
	 * @param SpeciesArrayAmount
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
	
	public void nextGen() {
		float Fit  = -Float.MAX_VALUE;
		
		//update threshold
		if(SpeciesArray.size()<speciesTarget) 
			threshold-=0.05;
		else
			threshold+=0.05;
		
		
		
		//look for best Network
		NeuronNetwork Best = null;
		int SpecBest = -1;
		for(int i=0;i<SpeciesArray.size();i++)
			for(NeuronNetwork NN:SpeciesArray.get(i))
				if(Fit<NN.getFitness()) {
					Fit=NN.getFitness();
					Best=NN;
					SpecBest=i;					
				}
		gen++;
		
		
		float Fsum = 0;
		int Isum = 0;
		population=new ArrayList<NeuronNetwork>();
		for(int i=0;i<SpeciesArray.size();i++) {			
			Fsum += (SpeciesArray.get(i).adjustedFitness/adjustedFitness*SpeciesArray.get(i).size()); // to solve rounding problems
			int nextSize = (int)Fsum-Isum;
			//System.out.println("size : "+nextSize);
			Isum=(int)Fsum;
			for(int j=0;j<nextSize;j++) {
				if(j==0 && i==SpecBest) {
					//if this is the best species we keep the best member of this species
					population.add(Best.clone());
					continue;
				}
				if(SpeciesArray.get(i).Best.Complexity()==0 || SpeciesArray.get(i).lastImprove>ImprovementLimit) {//improvment 
					NeuronNetwork NN=new NeuronNetwork(initialStructure,initialIntegrity);
					NN.normalizeNetwork(); 
					NN.setId(i);
					population.add(NN);
				}
				//if the species is not progressing or if it is in a soft lock configuration (as 0 complexity network for example), then we generate new network 
				NeuronNetwork newPop = SpeciesArray.get(i).CreateChildren();
				newPop.mutate();
				population.add(newPop);
			}
		}
		speciateArr();//respeciate for the next generation
	}
	
	/**
	 *test all neuralNetwork
	 * @param data
	 * @param target
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
	 *train all neuralNetwork
	 * @param data
	 * @param target
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
	
	public String header() {
		return("<Class : Genetic Algorithm | generation : "+gen+" | population : "+pop+">\n");
	}
	
	public String toString() {
		String output = header();
		for(NeuronNetwork NN:population) {
			output+=NN.toString();
		}
		return output;
	}
	
	public String infoSpeciesArray() {
		String output="<SpeciesArray Info | Number of Species : "+SpeciesArray.size()+" >\n\n";
		int i=1;
		for(ArrayList<NeuronNetwork> Spec:SpeciesArray) {
			output+="<SpeciesArray "+i+" population : "+Spec.size()+" >\n\n"+Spec.get(0).toString();i++;
			}
		return output;
	}
	
	public String infoSpeciesArrayShort() {
		String output="<SpeciesArray Info | Number of Species : "+SpeciesArray.size()+" >\n\n";
		int i=1;
		for(ArrayList<NeuronNetwork> Spec:SpeciesArray) {
			output+="<SpeciesArray "+i+" population : "+Spec.size()+" >\n\n";i++;
			}
		return output;
		
	}
	
}
