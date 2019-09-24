#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark

import matplotlib.pyplot as plt 
# In[131]:


from wordcloud import WordCloud


# In[2]:


import csv


# In[121]:


import numpy as np


# In[3]:


import pandas as pd


# In[4]:


from pyspark import SparkConf,SparkContext,SQLContext


# In[5]:


from pyspark import SparkConf,SparkContext,SQLContext


# In[6]:


conf=SparkConf()


# In[7]:


context=SparkContext(conf=conf)


# In[8]:


from pyspark.ml.feature import VectorAssembler


# In[9]:


from pyspark.ml.feature import FeatureHasher


# In[10]:


from pyspark.ml.feature import Tokenizer,HashingTF,IDF


# In[11]:


from pyspark.ml.classification import NaiveBayes, NaiveBayesModel


# In[12]:


from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel


# In[13]:


from pyspark.mllib.util import MLUtils


# In[14]:


from pyspark.ml import Pipeline


# In[15]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[16]:


from pyspark.sql.functions import col, lit


# In[17]:


sql=SQLContext(context)


# In[18]:


rddnew=context.textFile("/home/salman/Downloads/spamraw.csv")


# In[19]:


def vectorize_data(inputStr) :
    attribute_split = inputStr.split(",")
    spam_or_ham = 0.0 if attribute_split[0] == "ham" else 1.0
    return [spam_or_ham, attribute_split[1]]


# In[20]:


vectorized = rddnew.map(vectorize_data)


# In[21]:


dfnew = sql.createDataFrame(vectorized, ["label", "message"])


# In[22]:


dfnew.show()


# In[43]:


(training_data, test_data) = dfnew.randomSplit([0.7, 0.3])


# In[100]:


df=dfnew.toPandas()


# In[101]:


spam_words = ' '.join(list(df[df['label'] == 1]['message']))
spam_wc = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[102]:


ham_words = ' '.join(list(df[df['label'] == 0]['message']))
ham_wc = WordCloud(width = 512,height = 512).generate(ham_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(ham_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[44]:


tokenizer = Tokenizer(inputCol="message",outputCol="tokenized")


# In[45]:


hasher = HashingTF(inputCol = tokenizer.getOutputCol(), outputCol = "frequency")


# In[46]:


idf = IDF(inputCol = hasher.getOutputCol(), outputCol = "features")


# In[47]:


from pyspark.ml.classification import RandomForestClassifier


# In[48]:


from pyspark.ml.classification import LinearSVC


# In[49]:


from pyspark.ml.classification import NaiveBayes


# In[50]:


lsvc = LinearSVC(maxIter=10, regParam=0.1)


# In[51]:


rf = RandomForestClassifier(labelCol="label",                             featuresCol="features",                             numTrees = 100,                             maxDepth = 4,                             maxBins = 32)


# In[52]:


nb = NaiveBayes(smoothing=1.0, modelType="multinomial")


# In[53]:


pipelinerf = Pipeline(stages=[tokenizer,hasher,idf,rf])


# In[54]:


pipelinelsvc1 = Pipeline(stages=[tokenizer,hasher,idf,lsvc])


# In[55]:


pipelinenb = Pipeline(stages=[tokenizer,hasher,idf,nb])


# In[56]:


modelrf = pipelinerf.fit(training_data)


# In[57]:


modellsvcc = pipelinelsvc1.fit(training_data)


# In[58]:


modelnb = pipelinenb.fit(training_data)


# In[59]:


predictionrf = modelrf.transform(test_data)


# In[60]:


predictionlsvc= modellsvcc.transform(test_data)


# In[61]:


predictionnb = modelnb.transform(test_data)


# In[62]:


selectedrf = predictionrf.select("label", "message", "probability", "prediction")


# In[63]:


evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")


# In[64]:


accuracyrf = evaluator.evaluate(predictionrf)


# In[65]:


accuracylsvc= evaluator.evaluate(predictionlsvc)


# In[66]:


accuracynb= evaluator.evaluate(predictionnb)


# In[67]:


print("Test set accuracy = " + str(accuracyrf))


# In[68]:


print("Test set accuracy = " + str(accuracylsvc))


# 

# In[69]:


print("Test set accuracy = " + str(accuracynb))


# In[ ]:





# In[70]:


pdd=pd.DataFrame([["", ""]], columns=("label","message"))


# In[71]:


data=sql.createDataFrame(pdd)


# In[72]:


predict=modelnb.transform(data)


# In[73]:


predict.show()


# In[128]:


accuracies=[accuracyrf,accuracynb,accuracylsvc]


# In[130]:


names = ['accuracyrf', 'accuracynb', 'accuracylsvc']
values = accuracies
import matplotlib.pyplot as plt 
plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()


# In[1]:





# In[2]:




# In[ ]:




