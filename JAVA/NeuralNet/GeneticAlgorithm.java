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
	
	private float fitness=0;
	public float getFitness() {
		return fitness;
	}
	
	public int speciesTarget=4;
	public float threshold;
	private class Species extends ArrayList<NeuronNetwork>{
		
		public float fitness=0;
		public float totalFitness=0;
		public int LastImprovment =0;
		
		public void updateFitness() {
			totalFitness=0;
			for(NeuronNetwork NN:this)
				totalFitness+=NN.getFitness();
			fitness =totalFitness/this.size()*this.size();//adjusted fitness
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
		for(int i=0;i<pop;i++) {
			NeuronNetwork NN=new NeuronNetwork(In,Hid,Out,Integrity);
			NN.normalizeNetwork();
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
	 * speciate BRUH
	 * 
	 * @param threshold
	 */
	public void speciate(float threshold) {
		//create copy of the Neural network because we gonna remove value
		ArrayList<NeuronNetwork> temp = (ArrayList<NeuronNetwork>) population.clone();
		
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
					System.out.println(specRef.compare(temp.get(i)));
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
		gen++;
		int[] nextSize = new int[SpeciesArray.size()];
		float Fsum = 0;
		int Isum = 0;
		for(int i=0;i<SpeciesArray.size();i++) { 
			 Fsum += (SpeciesArray.get(i).fitness/fitness*SpeciesArray.get(i).size()); // to solve rounding problems
			 nextSize[i] = (int)Fsum-Isum;
			 Isum=(int)Fsum;
			Species nextPop = new Species();
			for(int j=0;j<nextSize[i];j++) {
				nextPop.add(SpeciesArray.get(i).CreateChildren());
			}
		}
	}
	
	/**
	 *test all neuralNetwork
	 * @param data
	 * @param target
	 */
	public void tests(ArrayList<float[]> data,ArrayList<float[]> target) {
		fitness=0;
		for(NeuronNetwork NN : population) {
			NN.resetFitness();
			NN.train(data, target);
			fitness+=NN.getFitness();
		}
		for(Species S:SpeciesArray)
			S.updateFitness();
		fitness/=pop;
	}
	
		
	
	public String header() {
		return("<Class : Genetic Algorithm | generation : NaN | population : "+pop+">\n");
	}
	
	public String toString() {
		String output = header();
		for(NeuronNetwork NN:population) {
			output+=NN.toString();
		}
		return output;
	}
	
	public String infoSpeciesArray() {
		String output="<SpeciesArray Info | Number of SpeciesArray : "+SpeciesArray.size()+" >\n\n";
		int i=1;
		for(ArrayList<NeuronNetwork> Spec:SpeciesArray) {
			output+="<SpeciesArray "+i+" population : "+Spec.size()+" >\n\n"+Spec.get(0).toString();i++;
			}
		return output;
	}
	
	public String infoSpeciesArrayShort() {
		String output="<SpeciesArray Info | Number of SpeciesArray : "+SpeciesArray.size()+" >\n\n";
		int i=1;
		for(ArrayList<NeuronNetwork> Spec:SpeciesArray) {
			output+="<SpeciesArray "+i+" population : "+Spec.size()+" >\n\n";i++;
			}
		return output;
		
	}
	
}
