#**NeuralNet Library**

##the Class present on the library are the following :
- Perceptron
- NeuralNetwork
- NeuronNetwork 
- GeneticAlgorithm

##Perceptron : <span style="color:#7eec9c">USABLE [1.0]</span>

pretty much nothing to say about it. it's a single neuron. see more on that in the javadocs

##NeuralNetwork : <span style="color:#7eec9c">USABLE [1.3]</span>

####this is the new **fully connected neural network** class **|VERSION :1.3|**.

I will surely improve the library with time to make the network handle more complex structure (as injecting input in the hidden layer) and training (adaptative learning rate, genetic algorithm etc ...).

**EDIT :** I don't think I will add more complex structure as there are now the free structure neural network

Don't forget to download the [utilities](https://github.com/netscape-swega/AI/tree/master/library/java) library with the NeuralNet library or the library will not work.

the library jar file was moved [here](https://github.com/netscape-swega/AI/tree/master/library/java).

##NeuronNetwork : <span style="color:#d6ca5b">ALPHA [1.0]</span>
###this is the **free connected neural network** class **|VERSION :1.0|**.

I will try to add genetic algorithm soon for this network.

I did not test it (except for very simple tests as XOR)
so it may have some issue.

also do not try to connect neuron to higher layer neuron, it will surely mess up the network.

I didn't do error check everywhere (because it's a pain to do).
so you cany do some stuff that the network cannot handle.

also the new SoftMax activation function that i add to the Utilities Library did not work for this networks.

Don't forget to download the [utilities](https://github.com/netscape-swega/AI/tree/master/library/java) library with the NeuralNet library or the library will not work.

the structure of the network is inspired by this [video](https://www.youtube.com/watch?v=NmCtSidJ7aY).

##GeneticAlgorithm : <span style="color:#d67f5b">NOT USABLE [/]</span>

this class is supposed to be used with NeuronNetwork to optimize their structure

still in development for now



#Utilities library

##the Class present on the library are the following :
- Matrix
- Function

##Matrix : <span style="color:#7eec9c">USABLE [1.3]</span>

this class represent the matrix mathematical object, fill with float.
may have some update (like 3D generalisation) in the future.

usefull for the fully connected neural network.

##Function : <span style="color:#7eec9c">USABLE [1.3]</span>

create custom variable for the most common activation function so you can avoid playing with the java Function class.

##Future of the project (in order of priority) : 
- implementation of a geneticAlgorithm
- implementation of Convolutional Neural Network
- implementation of recurent neuron

you can get more info on these in the javadocs.
(I update the javadoc only for big update so it may have some stuff in the code that is not documented yet)

***library made by physic Dev***

<!--6d96ee-->











