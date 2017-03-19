In this project, a hierarchical attention network was built to music genres from lyrics. The structure of this network is seen for the example song of 'Happy Birthday' below.
![HAN Network](https://github.com/alexTsaptsinos/cs224nProject/blob/master/images/network_image.png)

This was compared to several baseline models and performed superior in all case except when it was beaten by an LSTM over 20 genres. Results can be seen here:
![Results Table](https://github.com/alexTsaptsinos/cs224nProject/blob/master/images/resultstable.png)

As is evident, classifying solely by lyrics is a hard task!

A confusion matrix for the top 5 genres is seen here:
![Confusion Matrix](https://github.com/alexTsaptsinos/cs224nProject/blob/master/images/confusion.png)

We can extract the weights that the model applies to each line and words for new unseen songs. We take the top 5 most heavily weighted lines in a song and produce visualisations of the line and word weights. The line weights are on the left and the words are weighed according to the respective colorers. An example of this is seen in this correctly predicted Country song. More examples can be found in the ‘images’ folder.
![Country Visualisation](https://github.com/alexTsaptsinos/cs224nProject/blob/master/images/country_example.png)

At the time of writing, the code is horribly unstructured and lacks any good documentation. This therefore makes it hard for reusability. If I have time I will go through to clean everything up, if not and you see something you’re unsure of - please do send me a message.
