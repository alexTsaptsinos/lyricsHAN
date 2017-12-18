---
layout: post
title: Classifying Music Genres via Lyrics using a Hierarchical Attention Network
description: As part of [CS224N](http://web.stanford.edu/class/cs224n/) here at [Stanford](https://www.stanford.edu/) I began learning about the various uses of deep learning in natural language processing. As part of this course, I decided to begin a project to try and classify music genre using lyrics only which has typically been a tough problem in the music information retrieval (MIR) field. At the culmination of the course I was so invested in the course that I continued working on it and eventually published this research in ISMIR 2017, held in Suzhou, China.
tags: []
author: author
---
To begin the project I took inspiration from [the paper by Yang et al.](http://www.aclweb.org/anthology/N16-1174) using a Hierachical Attention Network (HAN) to classify documents. Similarly to documents, lyrics contain a hierachical structure: words go into lines, lines into sections (verse/chorus/...), and sections then form the whole song. Further, from the attention mechanism we can then extract and visualise where the network is applying its weights.

Using intact lyrics the song can be split into layers. At each layer we apply a bidirectional recurrent neural network (RNN) to obtain hidden state representations. The attention mechanism is then applied to form a weighted sum of that layers hidden representations i.e. a weighted sum of the word hidden representation vectors forms the line vector. We have now passed to a higher layer and can repeat the process until we finally end up with a vector which summarizes the whole song, from which we can classify via a softmax activation.

The structure of the network can be seen for the example song of 'Happy Birthday' below.
![HAN Architecture](/assets/img/lyricsHAN/network_image.pdf)
<!--![HAN Architecture]({{"/assets/img/lyricsHAN/network_image.pdf"}})-->

The HAN model was written using TensorFlow, with Keras as the top layer whenever possible. [This great blog post by Richard Liao](https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/) really helped guide me, since I was just starting out with TensorFlow then.

I was incredibly lucky to have been provided intact song lyrics from [LyricFind](http://www.lyricfind.com/), without which this project would not have been possible. After pre-processing the lyrics and tokenizing them we were able to train the HAN and test its performance. The HAN was compared to several baseline model and performed well compared to previous research, although the LSTM outperformed in in the 20 genre case. Results can be seen here:
![HAN results](/assets/img/lyricsHAN/results.png)

HN is the HAN network without the attention mechanism. HAN-L is the HAN model with layers at the word and line level. HAN-S is the HAN model with layers at the word and section level.

As is evident, classifying solely by lyrics remains a hard task! It's hard to compare to previous research, with varying number of genres used, varying genres in those lists, and no real standardisation of genres. However, we believe that these scores were some of the best reported!

The benefit of using the attention mechanism is the abilitiy to now feed in lyrics and visualise where the netowrk is applying heavy weights. In other words, we can see which words, or lines, the network deems important to classifying the genre. Below are some examples where the network correctly predicts the genre.
![HAN results](/assets/img/lyricsHAN/country_example.pdf)
![HAN results](/assets/img/lyricsHAN/rap_example.pdf)
![HAN results](/assets/img/lyricsHAN/rock_example.pdf)

We can see heavy weights on classic country words 'baby', 'driveway', 'ai', etc. Similarly, for the hip-hop/rap song we see mispellings and lite-cuss words heavily weighted. When heavier cuss words are used the HAN similarly applied heavy weights. One interesting pattern in rock we noticed was the heavy weighting of second-person pronouns such as 'you' or 'your'. This is contrasted by the weighting of first-person pronouns 'me', 'mine' in hip-hop/rap!

Of course, the network didn't get it right all the time. Below are some examples of the HAN incorrectly classifying the genre, although in each case you can see why it has done so.
![HAN results](/assets/img/lyricsHAN/popwrong_country_example.pdf)
![HAN results](/assets/img/lyricsHAN/popwrong_example.pdf)
![HAN results](/assets/img/lyricsHAN/rapwrong_pop_example.pdf)

Classification by lyrics alone is always going to be tricky, with vague genre boundaries and cover songs hardening matters. However, in combination with audio or symbolic data we believe the HAN could help boost genre classification accuracies. 

All the code is avalable [here](https://github.com/alexTsaptsinos/lyricsHAN), although I apologise in advance for not being formatted very cleanly.

For all the gory details you can read the full paper [here](https://alextsaptsinos.github.io/papers/lyricspaper.pdf).

Please get in touch if you have any questions!
