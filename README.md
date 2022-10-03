## NLP-Chatbot
A responsive chatbot trained on Sequence-to-Sequence modelling, deployed on Heroku.

Are you guys fascinated by chatbots? Wonder how these machines understand what you exactly mean and respond to you in human language?
This bot is built using Seq-2-Seq model, which I've explained in details in this medium post.

### What is Sequence-to-Sequence Modelling?
It is a special class of RNNs that take a sequence as an input and return a new sequence as the output. These sequences can be of different lengths and these models are well known to capture the essence of sequences with variable lengths. eg :<br />
Conversation: Hi, how are you? - - → I am good!<br />
Translation: Hi, how are you? - - → Salut, ça va?<br />
<p align="center">
<img src="https://cdn-images-1.medium.com/max/1000/1*O7a0ShsYNCjxs-lNeux8MA.png" width="50%" height="50%"><br />
</p>
In the above example, the model is divided into two parts.<br />
Encoder: It takes the input sequence in vectorized form and feeds it to an RNN that returns an output at each level along with its hidden states. These hidden states are then fed to the next level and the process repeats itself until it reaches the end. The final output of the encoder(in this case, 4) is its own output and the hidden states, which are dependent upon previous levels(words). The outputs are disregarded because we're only interested in the final context vector. The final hidden states are then sent to the decoder network.<br />
Decoder: It starts with a token implying the start of the decoding process(in this case, <GO>). Whenever the model reaches this token, it starts generating the predictions. On the first level of the decoder, the token passes through RNN to predict an output. This output along with hidden states of the previous level is then fed as an input to the next level to predict further words. Finally, the output of all these levels is combined to generate a sequence.

### Steps
1. Install the dependencies
```
pip install -r requirements.txt
```
2. Run the flask app
```
python app.py
```

#### The final output is like this -
```
- - - - - - - - - - - - - - - - - - - - - - - - - - - - 
.................# Welcome to Chatbot #................
- - - - - - - - - - - - - - - - - - - - - - - - - - - - 
You: hi let's watch a movie
Bot: no it is not like this 
=======================================================
You: don't you wanna watch a movie?
Bot: i do not want to talk about it 
=======================================================
You: why not?
Bot: i do not know 
=======================================================
You: you're rude
Bot: i am afraid i do not know how to put it in your ear 
=======================================================
You: i hate you
Bot: you do not know 
=======================================================
You: bye
Bot: bye 
=======================================================
You: quit
```
