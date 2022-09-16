**<h1>NeuralNet Library</h1>**

<h2>the Class present on the library are the following :</h2>
- Perceptron
- NeuralNetwork
- NeuronNetwork 
- GeneticAlgorithm

<h2>Perceptron : <span style="color:#7eec9c">USABLE [1.0]</span></h2>

pretty much nothing to say about it. it's a single neuron. see more on that in the javadocs

<h2>NeuralNetwork : <span style="color:#7eec9c">USABLE [1.3]</span></h2>

<h4>this is the new fully connected neural network class |VERSION :1.3|.</h4>

I will surely improve the library with time to make the network handle more complex structure (as injecting input in the hidden layer) and training (adaptative learning rate, genetic algorithm etc ...).

**EDIT :** I don't think I will add more complex structure as there are now the free structure neural network

Don't forget to download the [utilities](https://github.com/netscape-swega/AI/tree/master/library/java) library with the NeuralNet library or the library will not work.

the library jar file was moved [here](https://github.com/netscape-swega/AI/tree/master/library/java).

<h2>NeuronNetwork : <span style="color:#d6ca5b">ALPHA [1.0]</span></h2>
<h4>this is the <b>free connected neural network</b> class <b>|VERSION :1.0|</b>.</h4>

I will try to add genetic algorithm soon for this network.

I did not test it (except for very simple tests as XOR)
so it may have some issue.

also do not try to connect neuron to higher layer neuron, it will surely mess up the network.

I didn't do error check everywhere (because it's a pain to do).
so you cany do some stuff that the network cannot handle.

also the new SoftMax activation function that i add to the Utilities Library did not work for this networks.

Don't forget to download the [utilities](https://github.com/netscape-swega/AI/tree/master/library/java) library with the NeuralNet library or the library will not work.

the structure of the network is inspired by this [video](https://www.youtube.com/watch?v=NmCtSidJ7aY).

<h2>GeneticAlgorithm : <span style="color:#d67f5b">NOT USABLE [/]</span></h2>

this class is supposed to be used with NeuronNetwork to optimize their structure

still in development for now



<h1>Utilities library</h1>

<h2>the Class present on the library are the following :</h2>
- Matrix
- Function

<h2>Matrix : <span style="color:#7eec9c">USABLE [1.3]</span></h2>

this class represent the matrix mathematical object, fill with float.
may have some update (like 3D generalisation) in the future.

usefull for the fully connected neural network.

<h2>Function : <span style="color:#7eec9c">USABLE [1.3]</span></h2>

create custom variable for the most common activation function so you can avoid playing with the java Function class.

<h2>Future of the project (in order of priority) : </h2>
- implementation of a geneticAlgorithm
- implementation of Convolutional Neural Network
- implementation of recurent neuron

you can get more info on these in the javadocs.
(I update the javadoc only for big update so it may have some stuff in the code that is not documented yet)

***library made by physic Dev***

<!--6d96ee-->











