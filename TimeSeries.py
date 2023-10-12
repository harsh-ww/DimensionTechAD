import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os.path


# plots a graph with given parameters, x, y data
def plotData(graphType, x, y, width=-1):
    try:
        plot_func = getattr(plt, graphType)
        if width == -1:
            plot_func(x, y)
        else:
            plot_func(x, y, width=width)
    except AttributeError:
        if graphType == 'N/A':
            pass
        else:
            raise ValueError('Unsupported graph type ' + graphType + '.')


# adds labels to current graph and outputs the final result
def presentGraph(xLabel, yLabel, title):
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.show()


# gets dataframe and accounts for missing headings
def getDataframe(fileFormat, filepath):

    if not (os.path.isfile(filepath)):
        raise ValueError('File does not exist')

    # if headings do not exist, get headings and set names equal to those headings
    if checkHeadings(filepath):
        headings = getHeadings(fileFormat)
        df = pd.read_csv(filepath, names = headings)
    # if headings already exist, we can just read the csv file and pandas will assume top row is headings
    else:
        df = pd.read_csv(filepath)

    return df


# returns true if headings do not exist (i.e. need to add headings)
# and returns false if the csv file already has headings
def checkHeadings(filepath):
    # gets the top row of the CSV
    top = pd.read_csv(filepath, header = None).head(1).values.tolist()[0]

    # loop through the elements of the top row
    for element in top:
        # if each and every element is a string, then conclude that headings exist
        if isinstance(element, str):
            continue
        # if any of the cells is not a string, e.g. is a number, then conclude that headings do not exist
        else:
            return True

    return False


# returns a list of correct headings
def getHeadings(fileFormat):
    # currently the headings for CSI data is known,
    # to work with more types of data, we can simply add additional if statements here
    # headings must be known
    if fileFormat == 'CSI':
        return ['Date', 'Source', 'Source-ID', 'Open', 'High', 'Low', 'AdjustedClose', 'Close', 'Volume', 'Contract']
    else:
        raise ValueError('This is CSV data is currently unavailable, only CSI data is supported')


# extracts a title from a filepath
def getFilename(filepath):
    # turn the path into an array split at every '/'
    splitPath = filepath.split('/')
    # get just the filename from whole path
    fileName = splitPath[len(splitPath)-1]
    # remove file extension
    return fileName.split('.')[0]


# plots a graph of a given variable against date
def plotTimeSeries(fileFormat, filepath, dependentVar):

    df = getDataframe(fileFormat, filepath)

    if not dependentVar in df.columns:
        raise ValueError('This field (' + dependentVar + ') does not exist')

    # get dates and the dependent variable data from the data frame
    # and flatten arrays with itertools
    dates = list(itertools.chain.from_iterable(df[['Date']].values.tolist()))
    dependent = list(itertools.chain.from_iterable(df[[dependentVar]].values.tolist()))

    title = getFilename(filepath)
    plotData('plot', dates, dependent)
    plotData('plot', dates, calcSMA(dependent, int(len(dependent)*0.03)))
    plotData('plot', dates, calcEWMA(dependent, 32))
    # add some stats to the figure
    plt.text(30, .49, describe_helper(pd.Series(dependent))[0])
    plt.text(40, .49, describe_helper(pd.Series(dependent))[1])
    presentGraph("Date", dependentVar, title)


# plots a graph of returns of a given variable (the day to day difference) against date
def plotReturns(fileFormat, filepath, returnsVar):
    
    df = getDataframe(fileFormat, filepath)
    
    # get dates and the differences of the returns data from the data frame
    # and flatten array with itertools
    dates = list(itertools.chain.from_iterable(df[['Date']].values.tolist()))
    returns = list(itertools.chain.from_iterable(df[[returnsVar]].diff().values.tolist())) # uses pandas .diff() method to get differences between days

    title = getFilename(filepath)
    plotData('plot', dates, returns)
    # add some stats to the figure
    plt.text(30, .49, describe_helper(pd.Series(returns))[0])
    plt.text(40, .49, describe_helper(pd.Series(returns))[1])
    presentGraph("Date", (returnsVar + " returns"), title)


# plots a graph of arithmetic returns of a given variable against date
def plotAmetricReturns(fileFormat, filepath, returnsVar):

    df = getDataframe(fileFormat, filepath)

    # get dates and the differences of the returns data from the data frame
    # and flatten arrays
    dates = list(itertools.chain.from_iterable(df[['Date']].values.tolist()))
    dependent = list(itertools.chain.from_iterable(df[[returnsVar]].values.tolist()))

    ameticRet = []

    # arithmetic return calculation
    ameticRet.append(float('nan'))
    for i in range (1, len(dependent)):
        ameticRet.append((dependent[i]/dependent[i-1])-1)

    title = getFilename(filepath)
    plotData("scatter", dates, ameticRet)
    # add some stats to the figure
    plt.text(30, .49, describe_helper(pd.Series(ameticRet))[0])
    plt.text(40, .49, describe_helper(pd.Series(ameticRet))[1])
    presentGraph("Date", ("Arithmetic " + returnsVar + " returns"), title)
    

# plots a graph of logarithm returns of a given variable against date
def plotLogReturns(fileFormat, filepath, returnsVar):

    df = getDataframe(fileFormat, filepath)

    # get dates and the differences of the returns data from the data frame
    # and flatten arrays
    dates = list(itertools.chain.from_iterable(df[['Date']].values.tolist()))
    dependent = list(itertools.chain.from_iterable(df[[returnsVar]].values.tolist()))

    logRet = []

    # logarithmic return calculation
    logRet.append(float('nan'))
    for i in range (1, len(dependent)):
        logRet.append(np.log(dependent[i]/dependent[i-1]))

    title = getFilename(filepath)
    plotData("scatter", dates, logRet)
    # add some stats to the figure
    plt.text(30, .49, describe_helper(pd.Series(logRet))[0])
    plt.text(40, .49, describe_helper(pd.Series(logRet))[1])
    presentGraph("Date", ("Logarithmic " + returnsVar + " returns"), title)


# plot histogram for a given variable
def histReturns(fileFormat, filepath, returnsVar, startDate, endDate):

    # number of years within range
    years = int(endDate.split('-')[0]) - int(startDate.split('-')[0])
    barWidth = 0.4 + ((years-2) * 0.1)
    if barWidth > 1.0:
        barWidth = 1.0

    df = getDataframe(fileFormat, filepath) # whole dataframe
    dfBetween = df[(df['Date'] >= startDate) & (df['Date'] <= endDate)] # filter to include rows only within date range
    returns = list(itertools.chain.from_iterable(dfBetween[[returnsVar]].diff().values.tolist())) # get the column required and flatten array with itertools

    # create an array of frequencies, and an array of edges, the number of bins should be the number of data points divided by a constant
    freq, edges = np.histogram(returns[1:], bins=(len(returns)//30) )
    # find the mid points of each bin
    midWith = (edges[1:]+edges[:-1])/2

    # plot histogram
    title = ("Histogram: " + returnsVar + " returns for " + getFilename(filepath) + " between " + startDate + " and " + endDate)
    plotData("bar", midWith, freq, barWidth)
    # add stats to the figure
    plt.text(30, .49, describe_helper(pd.Series(midWith))[0])
    plt.text(40, .49, describe_helper(pd.Series(midWith))[1])
    presentGraph("Return", "Frequency", title)


def describe_helper(series):
    labels, values = "", ""

    splits = str(series.describe()).split()
    desc = series.describe()

    for i in range(0, len(splits), 2):
        labels += ((splits[i]) + ": \n")
    for j in range(0, len(desc)-1):
        values += (str(round(desc[j], 2)) + "\n")
    
    return labels, values

    

# calculates the exponentially weighted moving average for a dataset and given half life
def calcEWMA(data, halflife):
    weights = [(1/2)**(x/halflife) for x in range(len(data), 0, -1)]
    out = []
    for i in range(1, len(data)+1):
        m = wmean(data[:i], weights[:i])
        out.append(m)
    return out

# compute the weighted mean of data
def wmean(data, weights):
    # compute weighted mean of data.
    assert len(data) == len(weights)
    numer = [data[i] * weights[i] for i in range(len(data))]
    return (sum(numer)/sum(weights))


# calculates the simple moving average for a dataset and given window size
def calcSMA(data, winSize):
    # iteratively compute simple moving average over window of data.
    m = mean(data[:winSize])
    out = [float('nan')] * (winSize-1) + [m]
    for i in range(winSize, len(data)):
        m += (data[i] - data[i-winSize]) / winSize
        out.append(m)
    return out

# compute mean of data
def mean(data):
    return sum(data) / len(data)


def main():
    
    #plotTimeSeries('CSI', './data/CSI_HG_1.csv', 'Close')

    #plotReturns('CSI', './data/CSI_HG_1.csv', 'Close')
    #plotAmetricReturns('CSI', './data/CSI_HG_1.csv', 'Close')
    #plotLogReturns('CSI', './data/CSI_HG_1.csv', 'Close')

    #histReturns('CSI', './data/CSI_HG_1.csv', 'Close', '2000-01-01', '2005-01-01')
    #histReturns('CSI', './data/CSI_HG_1.csv', 'Close', '2000-01-01', '2010-01-01')
    #histReturns('CSI', './data/CSI_HG_1.csv', 'Close', '2000-01-01', '2015-01-01')
    #histReturns('CSI', './data/CSI_HG_1.csv', 'Close', '2000-01-01', '2020-01-01')

    testArr = [1.0, 2.0, 3.0, 5.0, 6.1, 9.21, 123.09, 09.1, 2.09277]
    print (new_describe_helper(pd.Series(testArr))[0])
    print (new_describe_helper(pd.Series(testArr))[1])


    return

main()


# Histogram: add stats like mean, variance etc, beneath the graph
# Moving averages, plot two lines comparing a smaller time period to a larger one, see short term trends vs long term trends and where they cross are often points of decision/interest
# ^^ read the book
# Create a dashboard, i.e. present everything nicely on jupyter notebooks
# with the dashboard you want to present a graph for each year all in one box for example, basically use common sense and analytical skills to create a dashboard which is informative
# Currently comparing returns, we could normalise it further by plotting returns divided volatility, which could be std deviation for example