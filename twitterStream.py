from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt


def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
   
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)


def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    listpositive = []
    listnegative = []
    timestep = []
    i = 0
    for data in counts:
      listpositive.append(data[0][1])
      listnegative.append(data[1][1])
      timestep.append(i)
      i=i+1
    
    plt.plot(timestep,listpositive, marker='o', label='positive')
    plt.plot(timestep,listnegative, marker='o', label='negative')
    plt.axis([-1, len(timestep)+3, 0, max(max(listpositive), max(listnegative))+100])
    plt.xticks(np.arange(0,len(timestep)+1,1.0))
    plt.legend(loc = 'upper left')
    plt.xlabel('Time Step')
    plt.ylabel('Word Count')
    plt.show()
    
    


def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    text = open(filename)
    store_words= []
    input = text.read().split('\n')
    for word in input:
       store_words.append(word)
    return store_words

def updateFunction(newValues, runningCount):
   if runningCount is None:
       runningCount = 0
   return sum(newValues, runningCount)


def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))
    words = tweets.flatMap(lambda l:l.split(" "))
    pos = words.map(lambda l:('positive',1) if l in pwords else ('positive',0))
    neg = words.map(lambda l:('negative',1) if l in nwords else ('negative',0))    
    total = pos.union(neg)
    total = total.reduceByKey(lambda l,x:l+x)
    
   
    runningCounts = total.updateStateByKey(updateFunction)
    runningCounts.pprint()
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # Let the counts variable hold the word counts for all time steps
    counts = []
    
    total.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)
    return counts


if __name__=="__main__":
    main()
