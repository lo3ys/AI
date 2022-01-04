import java.util.function.Function;

/**
 * enum for standard activation function
 * </br>
 * contain activations functions with their expression and derivative
 * (derivative's variable are the function itself and not the function input)
 * 
 * @author Physic Dev
 * @version 1.2
 */
public enum functions implements Function<Float, Float>{
	
	/**
	 * map value between 0 and 1
	 * 
	 * @since 1.0
	 */
	Sigmoid((x) -> 1/(float)(1+Math.exp(-x)),
			//(x) -> (float)(Math.exp(-x)/Math.pow((1+Math.exp(-x)),2))), //derivative expression
			(x) -> x*(1-x)),
	
	/**
	 * map value between -1 and 1
	 * @since 1.0
	 */
	Tanh((x) -> (float)Math.tanh(x),
		 //(x) -> 1/(float)Math.pow(Math.cosh(x),2)),
		 (x) -> 1-x*x),
	/**
	 * 0 for negative identity for positive
	 * @since 1.0
	 */
	ReLu((x) -> Math.max(0,x), 
		 (x) -> x>0?1f:0f),
	
	/**
	 * 0 for negative identity for positive
	 * @since 1.0
	 */
	LeakyReLu((x) -> Math.max(0.1f*x,x),
			  (x) -> x>0f?1f:0.1f),
	
	/**
	 * identity, using this function is like using no function
	 * @since 1.0
	 */
	Linear((x) -> x, 
		   (x) -> 1f),
	//change the step and absStep derivative to make them work
	//maybe there are something i don't understand in the use these functions
	
	/**
	 * return -1 or 1
	 * @since 1.0
	 */
	Step((x) -> x>0?1f:-1f,
		 (x) -> x>0?1f:-1f),
	
	/**
	 * return 0 or 1
	 * @since 1.0
	 */
	AbsStep((x) -> x>0f?1f:0f,
			(x) -> x>0?1f:-1f);

	public final Function<Float, Float> Function;
	public final Function<Float, Float> Derivative;
	functions(Function<Float, Float> function,Function<Float, Float> derivative) {
		Function=function;
		Derivative=derivative;
	}

	@Override
	public Float apply(Float value) {
		return Function.apply(value);
	}
	
	/**
	 * to apply the derivative function
	 * @since 1.0
	 */
	public Float derivative(Float value) {
		return Derivative.apply(value);
	}
}

