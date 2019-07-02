# from pytrends.request import TrendReq
# import pytrends
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from datetime import date, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from joblib import Parallel, delayed
import itertools
import random
import time
import math
import os

def readWordList(filename):
    """
    read in word list to array
    """
    words = []
    with open(filename, 'r') as f:
        for line in f:
            words += line.split()

    return words

def combineCsvs(path, wordList):
    df = pd.DataFrame()
    for wordCount in range(0, len(wordList), 5):
        filename = path + "/" + wordList[wordCount : wordCount + 5].__str__() + ".csv"
        tempDF = pd.read_csv(filename)
        tempDF = tempDF.set_index('date')
        df = pd.concat([df, tempDF], axis=1)
    return df


def getTrendsData(wordList):
    """
    get trends data for wordlist into dataframe
    """
    pytrend = TrendReq(hl='en-US', tz=360)
    df = pd.DataFrame()

    for wordCount in range(0, len(wordList), 5):
        while True:
            # try:
            print( wordCount)
            print( wordList[wordCount : wordCount + 5])
            df = df.append(getTrendsDataHelper(wordList[wordCount : wordCount + 5], pytrend))
            print( "did one..")
            break
            # except:
            print( "delay...")
            time.sleep(60)
            continue

    df.to_csv("./csvs/final.csv")
    return df


# def getTrendsDataHelper(wordList, pytrends):
#     """
#     get trends data for 5 words into dataframe
#     """
#     td269 = timedelta(days=269)
#     td1 = timedelta(days=1)

#     date1 = date(2004, 01, 01)
#     date2 = date1 + td269
#     endDate = date(2018, 02, 02)

#     df = pd.DataFrame()
#     while date2 != endDate:
#         timeframe = date1.__str__() + " " + date2.__str__()
#         pytrends.build_payload(wordList, cat=0, timeframe=timeframe, geo='US', gprop='')
#         df.append(pytrends.interest_over_time())
#         date1 = date1 + td1
#         date2 = date2 + td1

#     print( df)
#     print( df.loc["2004-9-26"])
#     return df


def getTrendsDataHelper(wordList, pytrends):
    """
    get trends data for 5 words into dataframe
    """
    td269 = timedelta(days=269)
    td254 = timedelta(days=254)
    td1 = timedelta(days=1)

    date1 = date(2004, 1, 1)
    date2 = date1 + td269
    endDate = date(2018, 2, 2)

    df = pd.DataFrame()
    dfs = []
    while date2 < endDate:
        timeframe = date1.__str__() + " " + date2.__str__()
        # csvFile = "./csvs/" + timeframe + "---" + wordList.__str__() + ".csv"
        # fh = open(csvFile, "a")
        # fh = open(csvFile, "r")

        # if fh.read() == '':
        pytrends.build_payload(wordList, cat=0, timeframe=timeframe, geo='US', gprop='')
        dfs += [pytrends.interest_over_time().drop("isPartial", axis=1)]
            # df.to_csv(csvFile)

        date1 = date1 + td254
        date2 = date1 + td269

    df = normalizeData(dfs)

    csvFile = "./csvs/" + wordList.__str__() + ".csv"
    fh = open(csvFile, "a")
    fh = open(csvFile, "r")
    df.to_csv(csvFile)
    return df

def normalizeData(dfs):
    # print( dfs)
    df1 = dfs[0]
    for i in range(1, len(dfs)):
        df2 = dfs[i]
        df1 = normalizeDataHelper(df1, df2)

    # print( df1)
    return df1 

def normalizeDataHelper(df1, df2):
    """
    given two time series normalize the data to eachother
    """

    # get overlapping periods
    overlappingIndices = df1.index.intersection(df2.index)

    # loop over each series in the data frame
    seriesL = []
    numCols = len(df1.iloc[0, :]) # -1 due to extra uneeded column
    for col in range(numCols):
        series1 = df1.iloc[:, col]
        series2 = df2.iloc[:, col]

        # print( "Series1:", series1)
        # print( "Series2:", series2)
        # print()


        # determine series with larger values
        firstOverlap = overlappingIndices[0] #could do this by averaging all values in overlappiung indices to determine the actually larger period
        # print( "First overlap 1:", series1.loc[firstOverlap])
        # print( "First overlap 2:", series2.loc[firstOverlap])
        # print()
        firstIsLarger = series1.loc[firstOverlap] > series2.loc[firstOverlap]

        # determine scaling factor by averaging the overlap
        # why not just average the overlapping values for each period and then multiply the smaller period by the ratio...
        # if no pairs have non-zero values then remove word?
        #       MIGHT NEED TO CHANNGE ^ to something less biased
        valueSum = 0
        numValues = 0
        for overlappingIndex in overlappingIndices:
            value1 = series1.loc[overlappingIndex]
            value2 = series2.loc[overlappingIndex]
            if value1 != 0 and value2 != 0:
                # print( "Value1:", value1)
                # print( "Value2:", value2)
                if firstIsLarger:
                    valueSum += value1 * 1.0 / value2
                else:
                    valueSum += value2 * 1.0 / value1

                numValues += 1

        # if no overlapping values worked, return all 0's for a word and continue to next word
        if numValues == 0:
            print( "[WARNING] All overlapping values for " + series1.name + " contained a 0, making all values for series 0")
            # series1[:] = 0
            # series2[:] = 0
            # seriesL += [series1.append(series2.drop(overlappingIndices))]
            # continue
            scalingFactor = 0
        else:
            scalingFactor = valueSum * 1.0 / numValues

        # scale smaller series
        if firstIsLarger:
            series2 = series2 * scalingFactor
        else:
            series1 = series1 * scalingFactor

        # remove overlapping indices from smaller series
        if firstIsLarger:
            series2 = series2.drop(overlappingIndices)
        else:
            series1 = series1.drop(overlappingIndices)

        # join series of same word together
        joinedSeries = series1.append(series2)
        # print( joinedSeries)

        # add series to list of series
        seriesL += [joinedSeries]

    # join series in list to single dataframe with all words and both sets of data
    normalizedDf = pd.concat(seriesL, axis=1)
    # print( "Normalized DF", normalizedDf)
    
    # return joined df

    return normalizedDf


def determineCoeff(df):
    X = df.as_matrix()
    columns = np.concatenate((df.columns.values, ['index']))

    pca = PCA()
    pca.fit(X)
    firstComp = pca.components_[0]

    vals = np.concatenate((firstComp, ['coeff']))
    coefDF = pd.DataFrame([vals], columns=columns)
    coefDF = coefDF.set_index('index')

    # print( coefDF.to_string())

    df = pd.concat([df, coefDF])

    return df

def createIndex(posWords, negWords, df, daysAverage=1):
    indexDF = pd.DataFrame(0, index=df.drop(['coeff']).index, columns=["index"])

    for index in indexDF.index:
        for pw in posWords:
            coef = float(df.loc["coeff"][pw])

            indexDF.loc[index] += coef * df.loc[index][pw]

        for nw in negWords:
            coef = float(df.loc["coeff"][nw])
            indexDF.loc[index] -= coef * df.loc[index][nw]


    indexDF.to_csv("./csvs/index.csv")
    return indexDF
    

def compareIndices(index1, freq1, index2, freq2):
    pass
    

def plotIndex(data, groundTruth="ics"):

    pass

def updateValues(index, r, rf, a, value, date, negate=False):
    indexVal = index.loc[date][0]
    rVal = float(r.loc[date][0])
    rfVal = float(rf.loc[date][0])
    w = (indexVal * a)**4
    if negate:
        w = 1-w

    if w > 1.0 or w < 0:

        print(("W was greater or less than 1.0, a is", a, "w is", w, "index val is", indexVal))
        exit(1)

    rShares = int(value * (1 - w) / rVal)
    rfShares = int(value * w / rfVal)

    return rShares, rVal, rfShares, rfVal, w

def sell(value, rShares, r, rfShares, rf, date):
    rVal = r.loc[date][0]
    rfVal = rf.loc[date][0]
    return value + rShares * rVal + rfShares * rfVal

def buy(value, rShares, rVal, rfShares, rfVal):
    return value - rShares * rVal - rfShares * rfVal

def checkDate(date, index):
    return index.index.contains(date)

def tradingStrat(r, rf, index, a, negate=False):
    # i have the index data
    # i can split it into x pre-determined portfolio splits
    # 
    #
    #
    # two week buy/sell horizon
    # based on the current index value, readjust portolio
    # regress current index value on weight of risky asset
    # can i make my own cost function to maximize ?
        # technically by trying a bunch of ideas and getting the one that maximizes my return
    # every 2 weeks
    # w = index * a
    # portfolio = w*market + (1-w) * rf (resell / buy to reach this split)
            # maximize sum of this ^
    # do stochastic gradient descent on the - value
    # could just try various a's (stopping at some value / w <=1) 
    # what do i need
    # restructure() 
    # data on current w, index, a, market values, rf values
    # getValue()
    

    # start with 100k = value
    # loop
    # do w = index * a, perhaps with an intercept term or some other form of scaling
    # sell all i have, add value to a list 
    # purchase r/rf using money
    # skip forward two weeks

    # what I need
    # ability to sell / purchase / lookup values of r/rf
    value = 100000
    values = {}
    ws = {}
    rSharesD = {}
    rfSharesD = {}
    stockDate = date(2006, 6, 21)
    endDate = date(2017, 12, 13)
    twoWeeks = timedelta(days=7)

    values[stockDate] = value

    # print(type(stockDate))
    # print(stockDate)
    # print(index.index[0])
    # print(type(index.index[0]))
    # print(index.loc[stockDate][0])
    rShares, rVal, rfShares, rfVal, w = updateValues(index, r, rf, a, value, stockDate, negate=negate)
    ws[stockDate] = w
    rSharesD[stockDate] = rShares
    rfSharesD[stockDate] = rfShares
    value = buy(value, rShares, rVal, rfShares, rfVal)

    stockDate += twoWeeks

    while(stockDate <= endDate):
        if not checkDate(stockDate, index):
            stockDate += twoWeeks
            continue

        # sell
        value = sell(value, rShares, r, rfShares, rf, stockDate)

        # add value to dict
        values[stockDate] = value

        # get new values
        rShares, rVal, rfShares, rfVal, w = updateValues(index, r, rf, a, value, stockDate, negate=negate)

        ws[stockDate] = w
        rSharesD[stockDate] = rShares
        rfSharesD[stockDate] = rfShares
        # buy
        value = buy(value, rShares, rVal, rfShares, rfVal)

        # update date
        stockDate += twoWeeks

    # sell and get final value????????
    value = sell(value, rShares, r, rfShares, rf, endDate)
    values[endDate] = value

    return values, ws, rSharesD, rfSharesD

def holdStrat(r):

    value = 100000
    values = {}

    stockDate = r.index[0].to_pydatetime().date()
    endDate = date(2017, 12, 13)
    twoWeeks = timedelta(days=7)

    values[stockDate] = value

    marketVal = float(r.loc[stockDate][0])
    rShares = int(value / marketVal)
    leftover = value - marketVal * rShares
    
    stockDate += twoWeeks

    while(stockDate <= endDate):
        if not checkDate(stockDate, r):
            stockDate += timedelta(days=1)
            continue

        marketVal = float(r.loc[stockDate][0])
        # sell
        value = marketVal * rShares + leftover

        # add value to dict
        values[stockDate] = value

        # update date
        stockDate += twoWeeks

    # sell and get final value????????
    marketVal = float(r.loc[endDate][0])
    value = marketVal * rShares + leftover
    values[endDate] = value

    return values

def copyFormat(master, slave):
    idxMaster = master.index
    idxSlave = slave.index
    differentIndices = idxSlave.difference(idxMaster)
    return slave.drop(differentIndices)

def setupIndex():
    index = pd.read_csv("./csvs/index.csv")
    dateIndex = pd.to_datetime(index['Unnamed: 0'], format='%Y-%m-%d', errors='ignore')
    index = index.set_index(dateIndex).drop(['Unnamed: 0'], axis=1)
    return index.rolling(30).mean()

def setupVIX():
    vix = pd.read_csv("./csvs/VIX.csv")
    dateIndex = pd.to_datetime(vix["Date"], errors='ignore')
    vix = vix.set_index(dateIndex).drop(['Date'], axis=1).rename({"Value": "VIX"}, axis='columns')
    return vix.rolling(30).mean()

def setupMarket():
    market = pd.read_csv("./csvs/market.csv")
    dateIndex = pd.to_datetime(market["Date"], errors='ignore')
    market = market.set_index(dateIndex).drop(['Date'], axis=1).rename({"Value": "Market"}, axis='columns')
    return market

def setupRf():
    rf = pd.read_csv("./csvs/rf.csv")
    dateIndex = pd.to_datetime(rf["Date"], errors='ignore')
    rf = rf.set_index(dateIndex).drop(['Date'], axis=1).rename({"Value": "Rf"}, axis='columns')
    return rf

def setupShort():
    rf = pd.read_csv("./csvs/short.csv")
    dateIndex = pd.to_datetime(rf["Date"], errors='ignore')
    rf = rf.set_index(dateIndex).drop(['Date'], axis=1).rename({"Value": "Rf"}, axis='columns')
    return rf

def setupCCI():
    cci = pd.read_csv("./csvs/CCI.csv")
    dateIndex = pd.to_datetime(cci["Time"], format='%Y-%m-%d', errors='ignore')
    cci = cci.set_index(dateIndex).drop(['Time'], axis=1).rename({"Value": "CCI"}, axis='columns')
    cci.index = cci.index.to_period('M').to_timestamp('M')
    return cci

def setupAllWords():
    df = pd.read_csv("./csvs/combined.csv")
    dateIndex = pd.to_datetime(df["date"], format='%Y-%m-%d', errors='ignore')
    return df.set_index(dateIndex).drop(['date'], axis=1)

def setupAllPrunedWords():
    df = pd.read_csv("./csvs/prunedCombined.csv")
    dateIndex = pd.to_datetime(df["date"], format='%Y-%m-%d', errors='ignore')
    return df.set_index(dateIndex).drop(['date'], axis=1)

class RealTimeStrategy:
    def __init__(self, posWords=[], negWords=[], stratPower=4, startVal=100000, startTrainDate=None, endTrainDate=None, r=None, rf=None):
        self.posWords = posWords
        self.negWords = negWords
        self.words = self.posWords + self.negWords
        self.wordsDF = pd.DataFrame(index=np.array([startTrainDate + timedelta(days=i) for i in range(1, (endTrainDate - startTrainDate).days + 20*365)]), columns=[word for word in self.words]) # training data loaded into here
        self.wordWs = None
        self.stratPower = stratPower
        # self.startDate = startDate
        # self.curDate = curDate
        self.portfolioVal = startVal
        self.portfolioVals = {}
        self.rShares = 0
        self.rfShares = 0 
        self.index = pd.DataFrame(index=np.array([startTrainDate + timedelta(days=i) for i in range(1, (endTrainDate - startTrainDate).days + 20*365)]), columns=["index"])
        self.counter = 0 # ignore resizing for now
        self.r = pd.DataFrame(index=np.array([startTrainDate + timedelta(days=i) for i in range(1, (endTrainDate - startTrainDate).days + 20*365)]), columns=["r"])
        self.rf = pd.DataFrame(index=np.array([startTrainDate + timedelta(days=i) for i in range(1, (endTrainDate - startTrainDate).days + 20*365)]), columns=["rf"])
        self.oneDay = timedelta(days=1)
        self.oneMonth = timedelta(days=30)
        self.delay = 42*self.oneMonth
        # in use:
        self.startTrainDate = startTrainDate
        self.endTrainDate = endTrainDate
        self.updateWDay = self.startTrainDate + self.delay
        self.updateStratDay = self.endTrainDate + self.oneDay
        self.leftover = 0

        # might need to initialize date or weights or something
        # perhaps some other stuff


    def initW(self, trainingWords):
        curDate = self.startTrainDate + self.delay
        self.wordsDF.loc[self.startTrainDate:self.endTrainDate] = trainingWords

        self.updateWDay = curDate
        self.updateW(self.startTrainDate)


    def initS(self):
        # create index for as many days as we currently have words
        d = self.startTrainDate + self.oneDay
        while d <= self.endTrainDate:
            self.index.loc[d] = 0
            for pw in self.posWords:
                coef = abs(float(self.wordWs.loc[pw][0]))
                self.index.loc[d] -= coef * self.wordsDF.loc[d][pw]

            for nw in self.negWords:
                coef = abs(float(self.wordWs.loc[nw][0]))
                self.index.loc[d] += coef * self.wordsDF.loc[d][nw]

            d += self.oneDay


    def trainIndex(self, trainingWords, trainingR, trainingRf):
        curDate = self.startTrainDate + self.delay

        self.wordsDF.loc[self.startTrainDate:self.endTrainDate] = trainingWords
        self.r.loc[self.startTrainDate:self.endTrainDate] = trainingR
        self.rf.loc[self.startTrainDate:self.endTrainDate] = trainingRf

        while curDate <= self.endTrainDate:
            if curDate == self.updateWDay:
                self.updateW(curDate - self.delay)

            self.index.loc[curDate] = 0

            for pw in self.posWords:
                coef = abs(float(self.wordWs.loc[pw][0]))
                self.index.loc[curDate] -= coef * self.wordsDF.loc[curDate][pw]

            for nw in self.negWords:
                coef = abs(float(self.wordWs.loc[nw][0]))
                self.index.loc[curDate] += coef * self.wordsDF.loc[curDate][nw]

            curDate += self.oneDay
            # self.counter += 1

    def trainStrat(self, sd, ed):
        curDate = sd
        normIndex = self.rollingNormalize()

        indexVal = normIndex.loc[curDate][0]
        rVal = self.r.loc[curDate][0]
        rfVal = self.rf.loc[curDate][0]
        w = (indexVal)**(self.stratPower)

        self.rShares = int(self.portfolioVal * (1 - w) / rVal)
        self.rfShares = int(self.portfolioVal * w / rfVal)

        self.leftover = self.portfolioVal - self.rShares * rVal - self.rfShares * rfVal
        self.portfolioVals[curDate] = self.portfolioVal

        curDate += self.oneDay
        while curDate <= ed:
            indexVal = normIndex.loc[curDate][0]
            rVal = self.r.loc[curDate][0]
            rfVal = self.rf.loc[curDate][0]
            w = (indexVal)**(self.stratPower)

            self.portfolioVal = self.rShares * rVal + self.rfShares * rfVal + self.leftover
            self.portfolioVals[curDate] = self.portfolioVal
            self.rShares = int(self.portfolioVal * (1 - w) / rVal)
            self.rfShares = int(self.portfolioVal * w / rfVal)

            self.leftover = self.portfolioVal - self.rShares * rVal - self.rfShares * rfVal

            curDate += self.oneDay

    def handleData(self, wordData, rDF, rfDF, curDate):
        #adds a new datapoint to our model. 
        # add data to wordsDF
        self.wordsDF.loc[curDate] = wordData

        # update counter
        # self.counter += 1

        # if self.counter = 

        # check if we need to update W
        if curDate == self.updateWDay:
            self.updateW(curDate - self.delay)

        # compute index using data and W
        self.index.loc[curDate] = 0
        for pw in self.posWords:
            coef = abs(float(self.wordWs.loc[pw][0]))
            self.index.loc[curDate] -= coef * self.wordsDF.loc[curDate][pw]

        for nw in self.negWords:
            coef = abs(float(self.wordWs.loc[nw][0]))
            self.index.loc[curDate] += coef * self.wordsDF.loc[curDate][nw]

        # check if we need to trade stock
        # print(rDF.index.contains(curDate))
        # print(curDate)
        # print(self.updateStratDay)
        if curDate >= self.updateStratDay and rDF.index.contains(curDate):
            # print(rDF.index.contains(curDate))
            # print(rDF.loc[curDate])
            self.r.loc[curDate] = rDF.loc[curDate][0]
            self.rf.loc[curDate] = rfDF.loc[curDate][0]
            self.updateStrat(curDate)



    def rollingNormalize(self):
        # normalizes the index using min max normalization
        index = self.index.rolling(30).mean()
        return (index - index.min())/(index.max() - index.min()) #normalize!

    def initStrat(self, curDate, initValue, rVal, rfVal):
        self.portfolioVal = initValue
        normIndex = self.rollingNormalize()
        indexVal = normIndex.loc[curDate][0]
        rVal = rVal
        rfVal = rfVal
        w = (indexVal)**(self.stratPower)

        self.rShares = int(self.portfolioVal * (1 - w) / rVal)
        self.rfShares = int(self.portfolioVal * w / rfVal)

        self.leftover = self.portfolioVal - self.rShares * rVal - self.rfShares * rfVal

    def updateStrat(self, curDate):
        # assumes the provided date is a trading day
        normIndex = self.rollingNormalize()
        indexVal = normIndex.loc[curDate][0]

        rVal = self.r.loc[curDate][0]
        rfVal = self.rf.loc[curDate][0]

        self.portfolioVals[curDate] = self.portfolioVal # might lead to an off by 1 week error
        self.portfolioVal = self.rShares * rVal + self.rfShares * rfVal + self.leftover
        w = (indexVal)**(self.stratPower)

        self.rShares = int(self.portfolioVal * (1 - w) / rVal)
        self.rfShares = int(self.portfolioVal * w / rfVal)

        self.leftover = self.portfolioVal - self.rShares * rVal - self.rfShares * rfVal

        self.updateStratDay = curDate + 7*self.oneDay


    def updateW(self, startDate):
        # update weights
        endDate = self.updateWDay
        pastData = self.wordsDF.loc[startDate:endDate]
        # print(pastData)
        self.wordWs = self.determineW(pastData).transpose()
        # print(self.wordWs)

        self.updateWDay = endDate + self.oneDay
        

    def determineW(self, df):
        X = df.as_matrix()
        columns = np.concatenate((df.columns.values, ['index']))

        pca = PCA()
        pca.fit(X)
        firstComp = pca.components_[0]

        vals = np.concatenate((firstComp, ['coeff']))
        coefDF = pd.DataFrame([vals], columns=columns)
        coefDF = coefDF.set_index('index')

        return coefDF

