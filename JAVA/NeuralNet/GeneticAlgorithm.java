package NeuralNet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import NeuralNet.NeuralNetwork;

public class GeneticAlgorithm {
	private int pop;
	
	private Random R = new Random();
	public void setRandom(Random r) {
		R=r;
	}
	
	public int getPop() {
		return pop;
	}
	
	private ArrayList<NeuronNetwork> population = new ArrayList<NeuronNetwork>(); 
	private ArrayList<ArrayList<NeuronNetwork>> Species = new ArrayList<ArrayList<NeuronNetwork>>();
	public int getDiversity() {
		return Species.size();
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
			//create new species
			Species.add(new ArrayList<NeuronNetwork>());
			//take random neural network among the non selected ones
			//it will be the reference of the new species
			int specRefId = R.nextInt(temp.size());
			NeuronNetwork specRef = temp.get(specRefId);
			//add the new reference
			Species.get(Species.size()-1).add(temp.get(specRefId));
			temp.remove(specRefId);
			//compare network and add similar network
			for(int i=0;i<temp.size();i++)
				if(specRef.compare(temp.get(i))<threshold) {
					System.out.println(specRef.compare(temp.get(i)));
					Species.get(Species.size()-1).add(temp.get(i));
					temp.remove(i);
					i--;
				}
		}
	}
	
	/**
	 * take random Neurals Networks to be the reference of their species
	 * then sort the remaining Neural Network in the species that is the closest to him (using the compare functio on the reference)
	 * @param SpeciesAmount
	 */
	public void forceSpeciate(int SpeciesAmount) {
		//create copy of the Neural network because we gonna remove value
		ArrayList<NeuronNetwork> temp = (ArrayList<NeuronNetwork>) population.clone();
		Species=new ArrayList<ArrayList<NeuronNetwork>>(SpeciesAmount);
		for(int i=0;i<SpeciesAmount;i++) {
			int specRefId = R.nextInt(temp.size());
			NeuronNetwork specRef = temp.get(specRefId);
			Species.add(new ArrayList<NeuronNetwork>());
			Species.get(i).add(specRef);
			temp.remove(specRefId);
		}
		
		//sort the remaining network
		for(NeuronNetwork NN:temp) {
			float min=Float.MAX_VALUE;
			int SpecPos = -1;
			for(int i=0;i<SpeciesAmount;i++) {
				float dif=NN.compare(Species.get(i).get(0));
				if(dif<min) {
					min=dif;
					SpecPos=i;
				}
			}
			Species.get(SpecPos).add(NN);
		}
	}
	
	public void tests(ArrayList<float[]> data,ArrayList<float[]> target) {
		for(NeuronNetwork NN : population)
			NN.train(data, target);
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
	
	public String infoSpecies() {
		String output="<Species Info | Number of species : "+Species.size()+" >\n\n";
		int i=1;
		for(ArrayList<NeuronNetwork> Spec:Species) {
			output+="<Species "+i+" population : "+Spec.size()+" >\n\n"+Spec.get(0).toString();i++;
			}
		return output;
	}
	
	public String infoSpeciesShort() {
		String output="<Species Info | Number of species : "+Species.size()+" >\n\n";
		int i=1;
		for(ArrayList<NeuronNetwork> Spec:Species) {
			output+="<Species "+i+" population : "+Spec.size()+" >\n\n";i++;
			}
		return output;
		
	}
	
}
