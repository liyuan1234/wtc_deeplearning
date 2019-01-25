# Deep question answering with the World Tree Explanation Corpus
### Introduction
In this project I use deep learning to predict answers to multiple choice science questions for which explanations are provided. These questions and explanations are found in the [world tree explanation corpus](https://arxiv.org/abs/1802.03052) made available by Peter Jansen (2018).

Two example question and explanations:

Question|Explanation|Ans
---|---
*The Sun appears to move across the sky each day, rising in the east and setting in the west. What causes this apparent motion? (A) the rotation of Earth on its axis (B) the revolution of the Sun around Earth (C) the Earth's distance from the Sun (D) the properties of Earth's atmosphere*| the Earth rotating on its axis causes the sun to appear to move across the sky during the day. the sun rises in the east. the sun sets in the west. rising is a kind of motion. setting is a kind of motion. if a human is on a rotating planet then other celestial bodies will appear to move from that human 's perspective. Earth is a kind of planet. the Sun is a kind of star. a star is a kind of celestial object and celestial body. the Earth rotates on its axis on its axis.|A
*Which of the following represents a chemical reaction? (A) a sugar cube dissolving in water. (B) ice cubes forming in a freezer. (C) ice cream melting in a bowl. (D) a cake baking in an oven.*|cooking causes a chemical reaction. baking is similar to cooking. chemical reactions cause chemical change.|D

### Dataset
Unprocessed dataset can be downloaded at http://cognitiveai.org/explanationbank/

The version I use here is the "with mercury" dataset, i.e. Worldtree_Explanation_Corpus_V1_Sept2017_withMercury.zip
After processing this dataset consists of 1663 examples, which I split into 1363,150,150 training, validation and test examples.

### Models
#### basic model

![](/home/liyuan/Desktop/base model.png)  

 - Average pooling after RNN

#### cnn
![](/home/liyuan/Desktop/cnn.png)








### Results



### Some statistics:


### Other resources:
Peter Jansen: What's in an Explanation? Toward Explanation-centered Inference for Science Exams  
https://www.youtube.com/watch?v=EneqL2sr6cQ

### extensions/ideas:
- instead of negative sampling every iteration, maybe change model to calculate the hinge loss for each of the three wrong answers, then minimize the sum of the three hinge loss. I think this will reduce randomness and might help train faster.
- try adding batch normalization - batch normalization makes training less sensitive to hyperparameter settings
- try regularization..
