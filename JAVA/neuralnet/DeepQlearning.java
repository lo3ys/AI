package neuralnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Stack;

import utilities.Playable;

/**
 * 
 * custom fully connected neural network that learn how to choose move in turn-based games without database. 
 * 
 * @author physic dev
 *
 */
public class DeepQlearning<K extends Playable> extends NeuralNetwork{
	
	private K game;
	
	/**
	 * !! warning !! this is NOT LSTM neural network
	 * the words short and long term memory are used only to make better explanation 
	 */
	/**
	 * NN short term memory to save state when there are no winner (so we don't know which one did good)
	 */
	private Stack<float[]> inputStack;
	private Stack<float[]> outputStack;
	private Stack<Integer> players;
	
	/**
	 * NN long memory if you want to store all of the past short term memory in a database (if you want to train the network with backpropagation)
	 */
	private ArrayList<float[]> dataBaseI;
	private ArrayList<float[]> dataBaseO;
	
	/**
	 * this boolean value indicate if you want to save or not save the state in a database
	 */
	public boolean saveData=false;
	
	/**
	 * constructor
	 * @param Hid the hidden layer
	 * @param gameTemplate the game on which the neural network will work (use a dedicated instance for the network, do not mess with this instance or it may cause bug)
	 */
	public DeepQlearning(int[] Hid,K gameTemplate) {
		super(gameTemplate.inputs(),Hid,gameTemplate.outputs());
		game=gameTemplate;
	}

	/**
	 * play N game against itself and use the output from the winning network to train itself
	 * @param games
	 */
	public void learn(int games) {
		for(int i=0;i<games;i++) {
			//init stacks
			inputStack=new Stack<float[]>();
			outputStack=new Stack<float[]>();
			players=new Stack<Integer>();
			//reset the game
			game.reset();
			int outcome=Integer.MIN_VALUE;
			//while there are no winner, play the game
			while(outcome==Integer.MIN_VALUE) {
				//store state in stacks
				inputStack.add(game.makeInputs());
				players.add(game.currentPlayer());
				
				//guessing Q value
				this.compute(inputStack.peek());
				int[]out=this.maxOutputs();
				int id=0;
				//attempt to play in the possition
				while(!game.playMove(out[i]))
					id++;
				//create the theoretical output value (0 everywhere except at the move value)
				float[] Out=new float[this.getOutputLength()];
				Out[id]=1;
				//add output to stack
				outputStack.add(Out);
				//update winner variable
				outcome=game.winner();
			}
			//if its a draw don't care
			if(outcome==game.draw)
				continue;
			while(!inputStack.isEmpty()){
				//if the player that perform the move was the winning player train on its input
				if(players.pop()==outcome) {
					if(saveData) {
						dataBaseI.add(inputStack.peek());
						dataBaseO.add(outputStack.peek());
					}
					this.backPropagation(inputStack.pop(), outputStack.pop());
				}else {
					inputStack.pop();outputStack.pop();
				}	
			}
		}
	}
}
