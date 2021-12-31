import java.util.function.Function;

public enum functions implements Function<Float, Float>{
	
	//activations functions with their expression and derivative
	//(derivative variable are the function itself and not the function input)
	Sigmoid((x) -> 1/(float)(1+Math.exp(-x)),
			//(x) -> (float)(Math.exp(-x)/Math.pow((1+Math.exp(-x)),2))), //derivative expression
			(x) -> x*(1-x)),
	Tanh((x) -> (float)Math.tanh(x),
		 //(x) -> 1/(float)Math.pow(Math.cosh(x),2)),
			(x) -> 1-x*x),
	ReLu((x) -> Math.max(0,x), 
		 (x) -> x>0?1f:0f), 
	LeakyReLu((x) -> Math.max(0.1f*x,x),
			  (x) -> x>0f?1f:0.1f),
	Linear((x) -> x, 
		   (x) -> 1f),
	//change the step and absStep derivative to make them work
	//maybe there are something i don't understand in these functions
	Step((x) -> x>0?1f:-1f,
		 (x) -> x>0?1f:-1f),
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
	
	public Float derivative(Float value) {
		return Derivative.apply(value);
	}
}

