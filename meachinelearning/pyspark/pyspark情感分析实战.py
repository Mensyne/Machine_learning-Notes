from __future__ import print_function
import json
import re
import string
import numpy as np

from pyspark import SparkContext, SparkConf
from pyspark import SQLContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon

def doc2vec(document):
    doc_vec = np.zeros(100)
    tot_words = 0

    for word in document:
        try:
            vec = np.array(lookup_bd.value.get(word)) + 1
            # print(vec)
            if vec is not None:
                doc_vec += vec
                tot_words += 1
        except:
            continue

    vec = doc_vec / float(tot_words)
    return vec


remove_spl_char_regex = re.compile('[%s]' % re.escape(string.punctuation))  # regex to remove special characters
stopwords = [u'rt', u're', u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your',
             u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers',
             u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what',
             u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were',
             u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a',
             u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by',
             u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after',
             u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under',
             u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all',
             u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not',
             u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don',
             u'should', u'now']

def tokenize(text):
    tokens = []
    text = text.encode('ascii', 'ignore')  # to decode
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                  text)  # to replace url with ''
    text = remove_spl_char_regex.sub(" ", text)  # Remove special characters
    text = text.lower()

    for word in text.split():
        if word not in stopwords \
                and word not in string.punctuation \
                and len(word) > 1 \
                and word != '``':
            tokens.append(word)
    return tokens

def res_visulization(pred_result):

    popdensity_ori = {'New Jersey':  0., 'Rhode Island': 0., 'Massachusetts': 0., 'Connecticut': 0.,
                      'Maryland': 0.,'New York': 0., 'Delaware': 0., 'Florida': 0., 'Ohio': 0., 'Pennsylvania': 0.,
                      'Illinois': 0., 'California': 0., 'Hawaii': 0., 'Virginia': 0., 'Michigan':    0.,
                      'Indiana': 0., 'North Carolina': 0., 'Georgia': 0., 'Tennessee': 0., 'New Hampshire': 0.,
                      'South Carolina': 0., 'Louisiana': 0., 'Kentucky': 0., 'Wisconsin': 0., 'Washington': 0.,
                      'Alabama':  0., 'Missouri': 0., 'Texas': 0., 'West Virginia': 0., 'Vermont': 0.,
                      'Minnesota':  0., 'Mississippi': 0., 'Iowa': 0., 'Arkansas': 0., 'Oklahoma': 0.,
                      'Arizona': 0., 'Colorado': 0., 'Maine': 0., 'Oregon': 0., 'Kansas': 0., 'Utah': 0.,
                      'Nebraska': 0., 'Nevada': 0., 'Idaho': 0., 'New Mexico':  0., 'South Dakota':	0.,
                      'North Dakota': 0., 'Montana': 0., 'Wyoming': 0., 'Alaska': 0.}
    popdensity  = {'New Jersey':  0., 'Rhode Island': 0., 'Massachusetts': 0., 'Connecticut': 0.,
                      'Maryland': 0.,'New York': 0., 'Delaware': 0., 'Florida': 0., 'Ohio': 0., 'Pennsylvania': 0.,
                      'Illinois': 0., 'California': 0., 'Hawaii': 0., 'Virginia': 0., 'Michigan':    0.,
                      'Indiana': 0., 'North Carolina': 0., 'Georgia': 0., 'Tennessee': 0., 'New Hampshire': 0.,
                      'South Carolina': 0., 'Louisiana': 0., 'Kentucky': 0., 'Wisconsin': 0., 'Washington': 0.,
                      'Alabama':  0., 'Missouri': 0., 'Texas': 0., 'West Virginia': 0., 'Vermont': 0.,
                      'Minnesota':  0., 'Mississippi': 0., 'Iowa': 0., 'Arkansas': 0., 'Oklahoma': 0.,
                      'Arizona': 0., 'Colorado': 0., 'Maine': 0., 'Oregon': 0., 'Kansas': 0., 'Utah': 0.,
                      'Nebraska': 0., 'Nevada': 0., 'Idaho': 0., 'New Mexico':  0., 'South Dakota':	0.,
                      'North Dakota': 0., 'Montana': 0., 'Wyoming': 0., 'Alaska': 0.}
    idx = 0
    for obj in rawTst_data['results']:
        user_location = obj['user_location']
        popdensity_ori[user_location] += (obj['polarity'] - 1)
        popdensity[user_location] += (pred_result[idx] - 1)
        idx += 1
    print('popdensity_ori')
    print(popdensity_ori)
    print("---------------------------------------------------------")
    print('popdensity')
    print(popdensity)
    print("---------------------------------------------------------")

    # Lambert Conformal map of lower 48 states.
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_axes([0.05, 0.20, 0.40, 0.75])
    ax3 = fig.add_axes([0.50, 0.20, 0.40, 0.75])
    m1 = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95, ax = ax1)
    # draw state boundaries.
    # data from U.S Census Bureau
    # http://www.census.gov/geo/www/cob/st2000.html
    shp_info = m1.readshapefile('st99_d00','states',drawbounds=True)
    cities = ['WA', 'OR', 'CA', 'NV', 'MT','ID','WY','UT',
          'CO', 'AZ', 'NM', 'ND', 'SD', 'NE', 'KS',
          'OK', 'TX', 'MN', 'IA', 'MO', 'AR', 'LA',
          'WI', 'IL', 'MI', 'IN', 'OH', 'KY', 'TN',
          'MS',	'AL', 'PA', 'WV', 'GA', 'ME', 'VT',
          'NY', 'VA', 'NC', 'SC', 'FL', 'AL']
    lat = [47.40, 44.57, 36.12, 38.31, 46.92, 44.24,
           42.75, 40.15, 39.06, 33.73, 34.84, 47.53,
           44.30, 41.125, 38.526, 35.565, 31.05,
           45.69, 42.01, 38.46, 34.97, 31.17, 44.27,
           40.35, 43.33, 39.85, 40.39, 37.67, 35.75,
           32.74, 61.37, 40.59, 38.49, 33.04, 44.69,
           44.045, 42.165, 37.77, 35.63, 33.86, 27.77,
           32.81]
    lon = [-121.49, -122.07, -119.68, -117.05, -110.45,
           -114.48, -107.30, -111.86, -105.31, -111.43,
           -106.25, -99.93, -99.44, -98.27, -96.726,
           -96.93, -97.56, -93.90, -93.21, -92.29,
           -92.37, -91.87, -89.62, -88.99, -84.54,
           -86.26, -82.76, -84.67, -86.70, -89.68,
           -152.40, -77.21, -80.95, -83.64, -69.38,
           -72.71, -74.95, -78.17, -79.81, -80.945,
           -81.67, -86.79]
    print(shp_info)
    # choose a color for each state based on population density.
    colors={}
    colors2 = {}
    statenames=[]
    cmap = cm.GMT_polar

    inverse = [(value, key) for key, value in popdensity_ori.items()]
    vmin = min(inverse)[0]
    vmax = max(inverse)[0]  # set range.
    inverse = [(value, key) for key, value in popdensity.items()]
    vmin_pred = min(inverse)[0]
    vmax_pred = max(inverse)[0]
    print('vmax:')
    print(vmax)
    print(m1.states_info[0].keys())
    for shapedict in m1.states_info:
        statename = shapedict['NAME']
        # skip DC and Puerto Rico.
        if statename not in ['District of Columbia','Puerto Rico']:
            pop = popdensity_ori[statename]
            pop_pred = popdensity[statename]
            # calling colormap with value between 0 and 1 returns
            # rgba value.  Invert color range (hot colors are high
            # population), take sqrt root to spread out colors more.
            if pop == 0:
               colors[statename] = cmap(0.5)[:3]
            elif pop < 0:
               colors[statename] = cmap(1.0 - np.sqrt((pop - vmin)/(0-vmin)))[:3]
            else:
               colors[statename] = cmap(0.5 - np.sqrt((pop - 0)/(vmax-0)))[:3]

            if pop_pred == 0:
                colors2[statename] = cmap(0.5)[:3]
            elif pop_pred < 0:
                colors2[statename] = cmap(1.0 - np.sqrt((pop_pred - vmin_pred) / (0 - vmin_pred)))[:3]
            else:
                colors2[statename] = cmap(0.5 - np.sqrt((pop_pred - 0) / (vmax_pred - 0)))[:3]
        statenames.append(statename)
    # cycle through state names, color each one.
    #ax = plt.gca() # get current axes instance
    for nshape,seg in enumerate(m1.states):
        # skip DC and Puerto Rico.
        if statenames[nshape] not in ['District of Columbia','Puerto Rico']:
            color = rgb2hex(colors[statenames[nshape]])
            #print(statenames[nshape])
            poly = Polygon(seg,facecolor=color,edgecolor=color)
            ax1.add_patch(poly)
    m1.drawparallels(np.arange(25,65,20),labels=[1,0,0,0])
    m1.drawmeridians(np.arange(-120,-40,20),labels=[0,0,0,1])
    x, y = m1(lon, lat)
    for city, xc, yc in zip(cities, x, y):
        ax1.text(xc - 60000, yc - 50000, city)
    ax1.set_title('Twitter-based sentiment analysis about Hillary ')

    m2 = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95, ax = ax3)
    m2.readshapefile('st99_d00', 'states', drawbounds=True)
    for nshape, seg in enumerate(m2.states):
        # skip DC and Puerto Rico.
        if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
            color = rgb2hex(colors2[statenames[nshape]])
            # print(statenames[nshape])
            poly = Polygon(seg, facecolor=color, edgecolor=color)
            ax3.add_patch(poly)

    ax3.set_title('Random Forest prediction')
    m2.drawparallels(np.arange(25,65,20),labels=[1,0,0,0])
    m2.drawmeridians(np.arange(-120,-40,20),labels=[0,0,0,1])
    x, y = m2(lon, lat)
    for city, xc, yc in zip(cities, x, y):
        ax3.text(xc - 60000, yc - 50000, city)

    ax2 = fig.add_axes([0.05, 0.10, 0.9, 0.05])
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal',
                                    ticks=[-1, 0, 1])
    cb1.ax.set_xticklabels(['negative', 'natural', 'positive'])
    cb1.set_label('Sentiment')

    plt.show()

conf = SparkConf().setAppName("sentiment_analysis")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

sqlContext = SQLContext(sc)
lookup = sqlContext.read.parquet("word2vecM_simple/data").alias("lookup")
print("------------------------------------------------------")
lookup.printSchema()
lookup_bd = sc.broadcast(lookup.rdd.collectAsMap())
print("------------------------------------------------------")

with open('tweets.json', 'r') as f:
    rawTrn_data = json.load(f)
    f.close()

trn_data = []
for obj in rawTrn_data['results']:
    token_text = tokenize(obj['text'])
    tweet_text = doc2vec(token_text)
    trn_data.append(LabeledPoint(obj['polarity'], tweet_text))

trnData = sc.parallelize(trn_data)
#print(trnData)

with open('hillary.json', 'r') as f:
    rawTst_data = json.load(f)
    f.close()

tst_data = []
for obj in rawTst_data['results']:
    token_text = tokenize(obj['text'])
    tweet_text = doc2vec(token_text)
    tst_data.append(LabeledPoint(obj['polarity'], tweet_text))

tst_dataRDD = sc.parallelize(tst_data)

model = RandomForest.trainClassifier(trnData, numClasses=3, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

predictions = model.predict(tst_dataRDD.map(lambda x: x.features))
labelsAndPredictions = tst_dataRDD.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(tst_dataRDD.count())
print('Test Error = ' + str(testErr))
print('Learned classification tree model:')
print(model.toDebugString())
# visulization the result
res_visulization(predictions.collect())
sc.stop()

