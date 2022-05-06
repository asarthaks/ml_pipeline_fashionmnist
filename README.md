### Fashion MNIST Classification and Its ML Pipeline using Apache Kafka


Part 1 :
The Part1 of the problem is to develop a cnn model, that is included in the folder AssignmentNotebooks.
Although the code in notebook is not flexible enough, but the same code is integrated into part3 which will work on other datasets as well (some changes will obviously be required)

Part 2 :
I haven't included specific code for part 2 of the assignment, but it can be achieved by using different bootstrap-server parameter in kafka-consumers and kafka-producers I wasn't able to try that out yet.

Part 3 :
The whole ml pipeline code is in this repo.

### How to run
1) install dependecies first

pip install -r requirements.txt

2) run python initialize.py

3) add fashion_mnist data in data/fashion folder (this part is not automated yet)

4) run predictor.py, trainer.py and sample.py in three different shells and then you can see the magic. (Make sure your zookeeper and kafka services are running)

I've also included code for retraining of the model after some predictions. I'll write in detail about the project later.


References :
https://towardsdatascience.com/putting-ml-in-production-i-using-apache-kafka-in-python-ce06b3a395c8
https://www.kaggle.com/code/pavansanagapati/a-simple-cnn-model-beginner-guide
