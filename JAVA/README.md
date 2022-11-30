**<h1>NeuralNet Library</h1>**

<h2>the Class present on the library are the following :</h2>
<ul>
<li>- Perceptron</li>
<li>- NeuralNetwork</li>
<li>- NeuronNetwork </li>
<li>- GeneticAlgorithm</li>
<li>- DeepQlearning</li>
</ul>

<h2>Perceptron : <span style="color:#7eec9c">USABLE [1.0]</span></h2>

pretty much nothing to say about it. it's a single neuron. see more on that in the javadocs

<h2>NeuralNetwork : <span style="color:#7eec9c">USABLE [1.3]</span></h2>

<h4>this is the new fully connected neural network class |VERSION :1.3|.</h4>

I will surely improve the library with time to make the network handle more complex structure (as injecting input in the hidden layer) and training (adaptative learning rate, genetic algorithm etc ...).

**EDIT :** I don't think I will add more complex structure as there are now the free structure neural network

Don't forget to download the [utilities](https://github.com/netscape-swega/AI/tree/master/library/java) library with the NeuralNet library or the library will not work.

the library jar file was moved [here](https://github.com/netscape-swega/AI/tree/master/library/java).

<h2>NeuronNetwork : <span style="color:#6d96ee">BETA [1.1]</span></h2>
<h4>this is the <b>free connected neural network</b> class <b>|VERSION :1.1|</b>.</h4>

this network is suposed to be used with the genetic algorithm neural network

I did a few test on it (but it might still have some bugs)

be carefull when you're working directly with the structure, you can end up with a broken network

also the new SoftMax activation function that i add to the Utilities Library did not work for this networks.

the structure of the network is inspired by this [algorithm](https://fr.wikipedia.org/wiki/Algorithme_NEAT).

<h2>GeneticAlgorithm : <span style="color:#d6ca5b">ALPHA[1.1]</span></h2>

this class is supposed to be used with NeuronNetwork to optimize their structure

did not a lot of testing so might be broken but for fine for the teest i've done.

the class is more messy than the other because the algorithm is quite complex

<h2>DeepQlearning : <span style="color:#d67f5b">UNSTABLE [0.1]</span></h2>

Reinforcement neural network designed to work with turn based game. 

learn to evaluate Q value in a turn based game.

this class inherit from standard fully connected neural network, I might make a version compatible with the class NeuronNetwork for some reinforcement/genetics application.

I have no idea if its working, I did not do any testing.


<h1>Utilities library</h1>

this library store code which is not directly machine learning but is usefull tool to create neural network or neural network application.
<h2>the Class/Interface present on the library are the following :</h2>
<ul>
<li>- Matrix</li>
<li>- Function</li>
<li>- Grid</li>
<li>- playable</li>
</ul>
<h2>Matrix : <span style="color:#7eec9c">USABLE [1.3]</span></h2>

this class represent the matrix mathematical object, fill with float.
may have some update (like 3D generalisation) in the future.

usefull for the fully connected neural network.

<h2>Function : <span style="color:#7eec9c">USABLE [1.3]</span></h2>

create custom variable for the most common activation function so you can avoid playing with the java Function class.

<h2>Grid : <span style="color:#7eec9c">USABLE [1.0]</span></h2>

some kind of integer matrix, used to create grid game like connect4 or chess. has some method designed to make graphics easier (translation hashmap to match integer value to char)

<h2>Playable : <span style="color:#6d96ee">BETA [1.0]</span></h2>

interface to make the link between a turn-based game and a neural network (DeepQLearning network only for now).

<h2>Future of the project (in order of priority) : </h2>
<ul>
<li>- implementation of Reinforcement NeuralNetwork</li>
<li>- implementation of Convolutional Neural Network</li>
<li>- implementation of recurent neuron</li>
</ul>
</br>
</br>

you can get more info on these in the javadocs.
(I update the javadoc only for big update so it may have some stuff in the code that is not documented yet)

***library made by physic Dev***

<!-- 7eec9c 6d96ee d6ca5b d67f5b-->











