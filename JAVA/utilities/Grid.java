package utilities;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Hashtable;

/**
 * 
 * @author physic gamer
 * 
 *         This is a basic square grid structure class that you can use to
 *         define more complex grid based structure (chess, tic tac toe, tileset
 *         game etc...)
 * 
 */
public class Grid {

	protected static final String fileHeader="GridData";
	protected int X, Y;
	/**
	 * the core of the class : this is where are store all the grid values
	 * 
	 * @see #set(int, int, int)
	 */
	protected int[] grid;

	/**
	 * translation Hashtable, associate int value to String (i'm using String and
	 * not char because it can be usefull in some case)
	 * 
	 * @see #getSymbols()
	 * @see #setSymb(Hashtable)
	 */
	protected Hashtable<Integer, String> symbols = new Hashtable<Integer, String>();

	/**
	 * change one of the grid value
	 * 
	 * this method is the equivalent of : <b> grid[x][y]=value;</b>
	 * 
	 * @param x     the x index
	 * @param y     the y index
	 * @param value the new value
	 */
	protected void set(int x, int y, int value) {
		grid[x+y*X] = value;
	}

	public int get(int i) {
		return grid[i];
	}
	public int get(int x,int y) {
		return get(X*y+x);
	}

	public int getX() {
		return X;
	}
	public int getY() {
		return Y;
	}
	/**
	 * fill the grid with 0 (recreate an empty array)
	 */
	public void reset() {
		grid=new int[X*Y];
	}

	/**
	 * this String is the default symbol, it's the symbols used when there are no
	 * translation available in the symbols HashTable
	 * 
	 * @see #getSymbols()
	 */
	public char empty_symbol = '.';

	public Grid(int s)throws IllegalArgumentException{
		this(s,s);
	}
	
	/**
	 * Constructor, build a 2D int array of size x*y
	 * 
	 * @param x the length of the array
	 * @param y the height or the array or the length of the subarray.
	 */
	public Grid(int x, int y)throws IllegalArgumentException{
		if((long)x*(long)y>Integer.MAX_VALUE)
			throw new IllegalArgumentException("the grid is too big !!");
		X = x;
		Y = y;
		grid=new int[x*y];
	}

	/**
	 * 
	 * @return the length of the grid (correspond to grid.length)
	 * 
	 * @see #getHeight()
	 */
	public int getLength() {
		return (X);
	}

	/**
	 * 
	 * @return the height of the grid (correspond to grid[0].length)
	 * 
	 * @see #getLength()
	 */
	public int getHeight() {
		return (Y);
	}

	/**
	 * the symbols variable store the different value the grid can take and
	 * associate them with symbols (String).
	 * 
	 * @return a copy of the symbol Hashtable.
	 * @see #setSymb(Hashtable)
	 * @see #symbols
	 */
	public Hashtable<Integer, String> getSymbols() {
		return (Hashtable<Integer, String>) (symbols.clone());
	}

	/**
	 * replace the symbols Hastable with a new one
	 * 
	 * @param symbol the new symbols Hastable
	 * @see #getSymbols()
	 */
	public void setSymb(Hashtable<Integer, String> symbol) throws IllegalArgumentException{
		for(String val:symbol.values())
			if(val.length()!=1)
				throw new IllegalArgumentException("symbol must be 1 character string, got "+val);
		symbols = (Hashtable<Integer, String>) symbol.clone();
	}

	/**
	 * return a simple header describing the grid (its size)
	 * 
	 * This method is separated from toString because it's more likely to change in
	 * child class (for example, in a morpion class, we keep the grid representation
	 * but we write Morpion instead of Grid in the header)
	 * 
	 * @return basics grid info in a String
	 * @see #toString()
	 */
	protected String header() {
		return ("<Grid " + X + " x " + Y + " >\n");
	}

	/**
	 * return the grid representation and a header
	 * 
	 * @return the String representation
	 * @see #Header()
	 */
	public String toString() {
		String R = header();
		for(int i=0;i<X;i++){
			for(int j=0;j<Y;j++)
				R += (!symbols.containsKey(grid[i+X*j])) ? empty_symbol : (symbols.get(grid[i+X*j]) == null ? grid[i+X*j] : symbols.get(grid[i+X*j]));
			R += "\n";
		}
		return (R);
	}

	/**
	 * return a HTML representation of the grid, usefull for easy web application,
	 * each cell is a span with GridSlot class and has variable storing their
	 * coordinate and their value.
	 * 
	 * @return a HTML representation of the grid.
	 */
	public String HTML() {
		String R = "";
		for (int i = 0; i < Y; i++) {
			for (int j = 0; j < X; j++) 
				R += "<span class=\"GridSlot\" X=" + j + " Y=" + i + " value=" + grid[j+X*i] + " >"
						+ ((symbols.containsKey(grid[j+X*i])) ? empty_symbol
								: (symbols.get(grid[j+X*i]) == null ? grid[j+X*i] : symbols.get(grid[j+X*i])))
						+ "</span>";
			R += "</br>";
		}
		return (R);
	}
	
	public void saveData(String path) throws IOException{
	    DataOutputStream out = null;
	    out=new DataOutputStream(new FileOutputStream(new File(path).getAbsoluteFile()));
	    out.writeUTF(fileHeader+"_DO");//DO for Data Only
	    WriteData(out);
	    out.writeUTF("%==");
	    out.close();
	    
	}
	
	public void save(String path) throws IOException{
	    DataOutputStream out = null;
	    out=new DataOutputStream(new FileOutputStream(new File(path).getAbsoluteFile()));
	    out.writeUTF(fileHeader+"_WS");//WS for with symbols
	    out.writeInt(symbols.size());
	    out.writeChar(empty_symbol);
	    for(int key:symbols.keySet()) {
	    	out.writeInt(key);
	    	out.writeChars(symbols.get(key));
	    }
	    WriteData(out);
	    out.writeUTF("%==");
	    out.close();
	}
	
	public void loadData(String path) throws IOException {
		DataInputStream in = new DataInputStream(new FileInputStream(new File(path).getAbsoluteFile()));
		String header=in.readUTF();
		if(header.equals(fileHeader+"_WS"))
			ReadSymbol(in);
		ReadData(in);
		in.close();
	}
	
	private void WriteData(DataOutputStream out) throws IOException {
	    out.writeInt(X);
	    out.writeInt(Y);
		for(int val:grid)
			out.writeInt(val);
	}

	private void ReadData(DataInputStream in) throws IOException {
		int x=in.readInt();
		int y=in.readInt();
		int[] grid_=new int[x*y];
		for(int i=0;i<x*y;i++)
			grid_[i]=in.readInt();
		X=x;Y=y;grid=grid_;
	}
	
	private void ReadSymbol(DataInputStream in) throws IOException {
		int nsymb=in.readInt();
		char ES=in.readChar();
		for(int i=0;i<nsymb;i++) {
			
		}
	}

}