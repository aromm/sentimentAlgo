from util import *
import multiprocessing

vix = setupVIX()
cci = setupCCI()

negwords = readWordList("wordslist_neg_econ.txt")
poswords = readWordList("wordslist_pos_econ.txt")
words = negwords + poswords

def monteCarloIndices():
    # # df getTrendsData(words)
    
    df = setupAllPrunedWords()
    print(df)
    prunedWords = list(df.columns)

    def loop(counter):
        subset = list(random.sample(prunedWords, 10))

        subPos = []
        subNeg = []
        for word in subset:
            if word in negwords:
                subNeg += [word]
            else:
                subPos += [word]

        subsetDF = df[list(subset)]

        length = len(subsetDF)
        trainingDF = subsetDF.iloc[0:int(length / 3 * 2)]

        rts = RealTimeStrategy(subPos, subNeg, trainingDF, startTrainDate=trainingDF.index[0], endTrainDate=trainingDF.index[-1])
        rts.trainIndex()
        index = rts.index.rolling(30).mean()

        # coefficients = determineCoeff(subsetDF)
        # index = createIndex(subPos, subNeg, coefficients)
        
        indexVIX = copyFormat(vix, index)
        indexCCI = copyFormat(cci, index)

        cciComp = pd.concat([indexCCI, cci], axis=1)
        vixComp = pd.concat([indexVIX, vix], axis=1)

        cciCor = cciComp.corrwith(cciComp['CCI'])[0]
        vixCor = vixComp.corrwith(vix['VIX'])[0]

        string = ""
        string += "MaxCorCCI:" + str(cciCor)
        string += os.linesep
        string += "MaxCorCCIWords:" + str(subset)
        string += os.linesep
        string += "MaxCorVix:" + str(vixCor)
        string += os.linesep + os.linesep
        with open("./log.txt", "a+") as f:
            f.write(string)

    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=4)(delayed(loop)(i) for i in range(99999999999))
    # while True:
    #     subset = list(sorted(random.sample(words, 10)))
        
    #     loop(subset)
    # with Pool(1) as p:
    #     p.map(loop, random.shuffle(list(itertools.combinations(words, 10))))

    # df = combineCsvs("./csvs", words)
    # coefficients = determineCoeff(df)
    # index = createIndex(poswords, negwords, coefficients)

def test2(subset, plot=False):
    subPos = []
    subNeg = []
    for word in subset:
        if word in negwords:
            subNeg += [word]
        else:
            subPos += [word]

    df = setupAllWords()

    print(subset)
    subsetDF = df[list(subset)]
    length = len(subsetDF)
    delay = 42*30
    trainingWords = subsetDF.iloc[0:delay+1]

    r = setupMarket()
    rf = setupShort()
    r = copyFormat(rf, r)

    

    # create strategy object
    rts = RealTimeStrategy(subPos, subNeg, startTrainDate=trainingWords.index[0], endTrainDate=trainingWords.index[-1])
    rts.initW(trainingWords)
    rts.initS()



    sd = subsetDF.index[delay+1]
    d = subsetDF.index[delay+1]
    endD = subsetDF.index[-1]

    valuesR = holdStrat(r.loc[d:])

    rDF = pd.DataFrame.from_dict(valuesR, orient='index').rename({0:"r"}, axis='columns')
    rDF = rDF.sort_index()


    rts.initStrat(d-rts.oneDay, 100000, r.loc[d-rts.oneDay][0], rf.loc[d-rts.oneDay][0])

    while d<=endD:
        rts.handleData(subsetDF.loc[d], r, rf, d)
        d+= rts.oneDay


    index = rts.index.rolling(30).mean()

    indexVIX = copyFormat(vix, index)
    indexCCI = copyFormat(cci, index)

    cciComp = pd.concat([indexCCI, cci], axis=1)
    vixComp = pd.concat([indexVIX, vix], axis=1)

    # cciComp.plot(secondary_y='CCI')
    # vixComp.plot(secondary_y='VIX')

    print("cciCor =", cciComp.corrwith(cciComp['CCI'])[0])
    print("vixCor =", vixComp.corrwith(vix['VIX'])[0])
    # # plt.show()
 
    # d = subsetDF.index[int(length/3*2)-30]
    # rts.updateStratDay = d + rts.oneDay
    # print (d)
    # print(r)
    # print(r.index.contains(d))
    # rts.initStrat(d, r.loc[d][0])
    # d += rts.oneDay
    # endD = subsetDF.index[-1]
    # while d<=endD:
    #     rts.handleData(subsetDF.loc[d], r, rf, d)
    #     d += timedelta(days=1)

    df = pd.DataFrame.from_dict(rts.portfolioVals, orient='index').rename({0:"Strategy"}, axis='columns')
    df = df.sort_index()
    # # rSharesDF = pd.DataFrame.from_dict(rSharesD, orient='index')
    # # rSharesDF = rSharesDF.sort_index().rename({0: "rShares"}, axis='columns')
    # # rfSharesDF = pd.DataFrame.from_dict(rfSharesD, orient='index')
    # # rfSharesDF = rfSharesDF.sort_index().rename({0: "rfShares"}, axis='columns')



    # ######### TOOTOTOODOOOOOO I think i only need 4 years training data since PCA only needs 4 years for weights... everything else is updated per day i beleive.. 
    # ####### ^^^^^^^^^^^^^^\
    # #### I could make a function that gives it 4 years word data and initialized the word weights with pca on this
    # ### then another function would just handle data, make the index, and run the strat using that index value!!!

    df = df/df.iloc[0] - 1

    market = rDF/rDF.iloc[0] - 1

    print("Return:",df.iloc[-1][0] - market.iloc[-1][0])
    print("\n")
    print("-"*10)

    if plot:
        cciComp.plot(secondary_y='CCI')
        vixComp.plot(secondary_y='VIX')
        market = copyFormat(df, market)
        success = pd.concat([df, market], axis=1)


    # # shares = pd.concat([wsDF, rfSharesDF, rSharesDF], axis=1)
    # # shares.plot(secondary_y="ws")
        success.plot()
    
        plt.ylabel("Return (100's %)")
        plt.title("Strategy vs. Market Asset")
    
        plt.show()
        print(df)
    # # print(rts.index.rolling(30).mean())


def test(subset):
    subPos = []
    subNeg = []
    for word in subset:
        if word in negwords:
            subNeg += [word]
        else:
            subPos += [word]

    df = setupAllWords()

    subsetDF = df[list(subset)]
    length = len(subsetDF)
    trainingDF = subsetDF.iloc[0:int(length / 3 * 2)]

    r = setupMarket()
    rf = setupShort()
    r = copyFormat(rf, r)
    trainingR = r.iloc[0:int(length / 3 * 2)-30]
    trainingRf = rf.iloc[0:int(length / 3 * 2)-30]

    valuesR = holdStrat(r)

    rDF = pd.DataFrame.from_dict(valuesR, orient='index').rename({0:"r"}, axis='columns')
    rDF = rDF.sort_index()

    # create strategy object
    rts = RealTimeStrategy(subPos, subNeg, startTrainDate=trainingDF.index[0], endTrainDate=trainingDF.index[-1])
    rts.trainIndex(trainingDF, trainingR, trainingRf)
    index = rts.index.rolling(30).mean()

    indexVIX = copyFormat(vix, index)
    indexCCI = copyFormat(cci, index)

    cciComp = pd.concat([indexCCI, cci], axis=1)
    vixComp = pd.concat([indexVIX, vix], axis=1)

    cciComp.plot(secondary_y='CCI')
    vixComp.plot(secondary_y='VIX')
    print("cciCor =", cciComp.corrwith(cciComp['CCI'])[0])
    print("vixCor =", vixComp.corrwith(vix['VIX'])[0])
    # plt.show()
 
    d = subsetDF.index[int(length/3*2)-30]
    rts.updateStratDay = d + rts.oneDay
    print (d)
    print(r)
    print(r.index.contains(d))
    rts.initStrat(d, r.loc[d][0])
    d += rts.oneDay
    endD = subsetDF.index[-1]
    while d<=endD:
        rts.handleData(subsetDF.loc[d], r, rf, d)
        d += timedelta(days=1)

    df = pd.DataFrame.from_dict(rts.portfolioVals, orient='index').rename({0:"Strategy"}, axis='columns')
    df = df.sort_index()

    # rSharesDF = pd.DataFrame.from_dict(rSharesD, orient='index')
    # rSharesDF = rSharesDF.sort_index().rename({0: "rShares"}, axis='columns')
    # rfSharesDF = pd.DataFrame.from_dict(rfSharesD, orient='index')
    # rfSharesDF = rfSharesDF.sort_index().rename({0: "rfShares"}, axis='columns')



    ######### TOOTOTOODOOOOOO I think i only need 4 years training data since PCA only needs 4 years for weights... everything else is updated per day i beleive.. 
    ####### ^^^^^^^^^^^^^^\
    #### I could make a function that gives it 4 years word data and initialized the word weights with pca on this
    ### then another function would just handle data, make the index, and run the strat using that index value!!!


    df = df/df.iloc[0] - 1
    print("df")
    print(df.iloc[0])
    df.plot()
    plt.show()
    market = rDF/rDF.iloc[0] - 1


    returnDiff = df.iloc[-1][0] - marketReturns.iloc[-1][0]

    # success = pd.concat([df, market], axis=1)
    # # shares = pd.concat([wsDF, rfSharesDF, rSharesDF], axis=1)
    # # shares.plot(secondary_y="ws")
    # success.plot(secondary_y="ws")
    # plt.ylabel("% Return")
    # plt.show()
    # print(rts.index.rolling(30).mean())

def doPlots(subset):
# setup dfs

    subPos = []
    subNeg = []
    for word in subset:
        if word in negwords:
            subNeg += [word]
        else:
            subPos += [word]

    df = setupAllWords()

    subsetDF = df[list(subset)]


    print(trainIndex(subsetDF, subsetDF.index[0], subsetDF.index[-1], subPos, subNeg))

    coefficients = determineCoeff(subsetDF)
    print(coefficients)
    # index = createIndex(subPos, subNeg, coefficients)
    index = setupIndex()
    index = (index - index.min())/(index.max() - index.min()) #normalize!

    
    indexVIX = copyFormat(vix, index)
    indexCCI = copyFormat(cci, index)

    cciComp = pd.concat([indexCCI, cci], axis=1)
    vixComp = pd.concat([indexVIX, vix], axis=1)

    cciComp.plot(secondary_y='CCI')
    vixComp.plot(secondary_y='VIX')
    print("cciCor =", cciComp.corrwith(cciComp['CCI'])[0])
    print("vixCor =", vixComp.corrwith(vix['VIX'])[0])
    negate = cciComp.corrwith(cciComp['CCI'])[0] > 0

    # plt.show()

    r = setupMarket()
    rf = setupShort()
    r = copyFormat(rf, r)

    valuesR = holdStrat(r)

    rDF = pd.DataFrame.from_dict(valuesR, orient='index').rename({0:"r"}, axis='columns')
    rDF = rDF.sort_index()

    index4Market = copyFormat(rf, index).rename({0:"Index"}, axis='columns')

    As = np.arange(1.000, 0.200, -0.01)
    bestA = 0.00
    bestReturn = 0.00
    for a in As:

    #.0002 is best so far 
    # RIGHT NOW THIS ISNT A LAG BASED MODEL, MAYBE IT SHOULD BE

    # PROBLEM SINCE MY INDEX IS NEGATIVE SO w GOES UP FOR A LOWER INDEX.... MEANING WE DO MORE OF RISKY ASSET IN TOUGHER TIMES...

        valuesStrat, ws, _, _ = tradingStrat(r, rf, index4Market, a, negate=negate)
        df = pd.DataFrame.from_dict(valuesStrat, orient='index').rename({0:"Strategy"}, axis='columns')
        df = df.sort_index()

        wsDF = pd.DataFrame.from_dict(ws, orient='index')
        wsDF = wsDF.sort_index().rename({0: "ws"}, axis='columns')

        stratReturns = df/df.iloc[0] - 1
        marketReturns = rDF/rDF.iloc[0] - 1

        returnDiff = stratReturns.iloc[-1][0] - marketReturns.iloc[-1][0]
        if returnDiff > bestReturn:
            bestA = a 
            bestReturn = returnDiff
    print("Best A:", bestA)
    print("Best Return:", bestReturn, "\n")
    valuesStrat, ws, rSharesD, rfSharesD = tradingStrat(r, rf, index4Market, bestA)

    df = pd.DataFrame.from_dict(valuesStrat, orient='index').rename({0:"Strategy"}, axis='columns')
    df = df.sort_index()

    wsDF = pd.DataFrame.from_dict(ws, orient='index')
    wsDF = wsDF.sort_index().rename({0: "ws"}, axis='columns')
    rSharesDF = pd.DataFrame.from_dict(rSharesD, orient='index')
    rSharesDF = rSharesDF.sort_index().rename({0: "rShares"}, axis='columns')
    rfSharesDF = pd.DataFrame.from_dict(rfSharesD, orient='index')
    rfSharesDF = rfSharesDF.sort_index().rename({0: "rfShares"}, axis='columns')


    df = df/df.iloc[0] - 1
    market = rDF/rDF.iloc[0] - 1

    success = pd.concat([df, market, wsDF], axis=1)
    shares = pd.concat([wsDF, rfSharesDF, rSharesDF], axis=1)
    shares.plot(secondary_y="ws")
    success.plot(secondary_y="ws")
    plt.ylabel("% Return")
    plt.show()
    
# unsure what this part is
    # lists = sorted(values.items())
    # x, y = zip(*lists)

    # plt.plot(x, y)

    # plt.show()
# end uncertainty

if __name__ == '__main__':
    # monteCarloIndices()
    # doPlots(['vagrant', 'bribe', 'expensive', 'pollution', 'affluent', 'debtor', 'aristocracy', 'patronage', 'equity', 'default'])
    # doPlots(['affluent', 'debtor', 'advantage', 'miser', 'backwardness', 'default', 'bribe', 'afloat', 'meritorious', 'capitalize'])
    subsets = [
['gain', 'donation', 'gamble', 'crisis', 'ghetto', 'inflation', 'inexpensive', 'default', 'blackmail', 'unemployed'],
['donate', 'donation', 'frugal', 'laid', 'poor', 'depression', 'expensive', 'aristocracy', 'charity', 'ghetto'],
['waste', 'donation', 'benefit', 'jobless', 'aristocrat', 'expensive', 'profit', 'buy', 'privileged', 'inflation'],
['default', 'gift', 'benefactor', 'pollution', 'priceless', 'unprofitable', 'benevolent', 'race', 'radical', 'productivity'],
['default', 'benevolence', 'patron', 'capitalize', 'endow', 'luxury', 'frugal', 'nobility', 'gamble', 'broke'],
['gain', 'charitable', 'allowance', 'destitute', 'costliness', 'betrothal', 'expense', 'recession', 'pollution', 'beggar'],
['default', 'benefit', 'unemployed', 'liquidate', 'capitalize', 'squander', 'extravagant', 'contribute', 'backwardness', 'crisis'],
['default', 'boom', 'debtor', 'expensive', 'blackmail', 'costliness', 'bonus', 'breadwinner', 'nobility', 'poor'],
['deficit', 'liquidation', 'privileged', 'hole', 'invaluable', 'allowance', 'recession', 'afloat', 'charitable', 'bankrupt'],
['ghetto', 'aristocrat', 'costly', 'profitable', 'bum', 'steal', 'depreciation', 'inflation', 'advantage', 'donation'],
['breadwinner', 'backward', 'depreciation', 'cooperative', 'ruin', 'recession', 'afloat', 'patronage', 'depression', 'default'],
['bankruptcy', 'fine', 'pollution', 'squander', 'tariff', 'domination', 'uneconomical', 'depreciation', 'fellowship', 'affluent'],
['jobless', 'inflation', 'ruin', 'backward', 'default', 'affluence', 'steal', 'cheap', 'laid', 'benefit'],
['destitute', 'default', 'allowance', 'beggar', 'aristocracy', 'productivity', 'inflation', 'depreciation', 'debtor', 'backward'],
['recession', 'crisis', 'inflation', 'invaluable', 'intervention', 'donation', 'poverty', 'contribution', 'benevolence', 'inherit'],
['nobility', 'associate', 'depreciation', 'benefactor', 'bargain', 'liquidate', 'cost', 'inflation', 'default', 'backer'],
['depression', 'radical', 'extravagant', 'gold', 'default', 'recession', 'commoner', 'bequeath', 'advantage', 'invaluable'],
['squander', 'liquidation', 'nobleman', 'advantage', 'bankruptcy', 'priceless', 'steal', 'waste', 'aristocrat', 'cooperative'],
['productive', 'lucrative', 'radical', 'deficit', 'ruin', 'default', 'benevolence', 'backwardness', 'bankrupt', 'gift']
    ]
    for subset in subsets:
        test2(['affluent', 'debtor', 'advantage', 'miser', 'backwardness', 'default', 'bribe', 'afloat', 'meritorious', 'capitalize'], plot=True)
        exit(1)
    
        


# part A: download data into csv files
# cci = pd.read_csv("./csvs/CCI.csv")
# df = pd.read_csv("./csvs/index.csv")
# df = df.set_index("Unnamed: 0")

# df = df.rolling(30).mean()
# df = df.iloc[::30,:]
# df = df.iloc[2:,:]
# print(len(df))
# print(len(cci))
# ax = df.plot()
# cci.plot(ax=ax, secondary_y=True)
# plt.show()

# negwords = readWordList("wordslist_neg_econ.txt")
# poswords = readWordList("wordslist_pos_econ.txt")
# words = negwords + poswords
# # print len(poswords), len(negwords)
# # print "RUNNING\n"
# # # df = getTrendsData(words)
# df = combineCsvs("./csvs", words)
# print "\nEND\n"
# print df
# print
# print
# coefficients = determineCoeff(df)
# # print coefficients.loc['coeff'].index
# m = -99999999
# mi = ""
# s = 99999999
# si = ""
# for index in coefficients.loc['coeff'].index:
#     val = float(coefficients.loc['coeff', index])
#     # print type(val)
#     if val > m:
#         m = val
#         mi = index
#     elif val < s:
#         s = val
#         si = index

# print "MAX:", mi, m
# print "MIN:", si, s


    # print index + ":", coefficients.loc['coeff', index]
# print max(coefficients.loc['coeff'])
# createIndex(poswords, negwords, coefficients)



# part B: normalize data into a single dataframe
# df1 = pd.read_csv("./csvs/2004-01-01 2004-09-26---['advantage', 'affluence', 'affluent', 'afloat', 'allowance'].csv")
# df1 = df1.set_index('date')
# df2 = pd.read_csv("./csvs/2004-09-11 2005-06-07---['advantage', 'affluence', 'affluent', 'afloat', 'allowance'].csv")
# df2 = df2.set_index('date')

# normalizedDf = normalizeData(df1, df2)
# normalizedDf.to_csv("./csvs/combo.csv")

