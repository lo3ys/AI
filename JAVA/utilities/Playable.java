package utilities;
/**
 * 
 * @author physic gamer
 *
 * this interface is used to make turn-based game with fixed amount of player compatible with neural network (so link neural network to game is easier)
 */
public interface Playable {
	
	public static final int draw=0;
	public static final int inProgress=Integer.MIN_VALUE;
	
	/**
	 * @return the amount of inputs needed
	 */
	public int inputs();
	/**
	 * @return the amount of outputs needed
	 */
	public int outputs();
	
	/**
	 * attempt a move
	 * @param choices the chosen move
	 * @return false if the move failed (mostly caused by illegal move, game ended), true if not
	 */
	public boolean playMove(int choices);
	/**
	 * @return a list of all the possible move with the actual state 
	 */
	public int[] possibleMove();
	
	/**
	 * cancel the last move
	 * @param lastChoice the last move played
	 */
	public void cancelMove(int lastChoice);
	
	/**
	 * 
	 * @return the current player id
	 */
	public int currentPlayer();
	
	/**
	 * @return the winner of the game, 0 if its a draw, minimum value if the game is not finished.
	 */
	public int winner();
	
	/**
	 * reset the game
	 */
	public void reset();
	
	/**
	 * @return a float array for the neural network
	 */
	public float[] makeInputs();
}
