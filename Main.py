"""
This program simplifies the calculations necessary to determine profitability of cryptocurrency mining
infrastructure.


Vega 64 statistics: https://wccftech.com/amd-rx-vega-64-mining-performance-blows-away-titan-v-xmr-monero/


With an ASIC you buy a dodgy piece of fire hazard equipment, that is only good for one or perhaps two algos,
that eats electricity, devalues before your eyes and might be ‘bricked’ by the time you get it.

Monero start date was 4-18-2014

"""
import copy
import datetime
import math
import statistics as s

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from dateutil import parser

"""
We will calculate the per card profit and plot the historical results, given
#1 Historical mining difficulty
#2 Historical fixed block reward + estimated transaction fees
"""


def main():
    # Define parameters
    kileen = ElectricitySource(650, 18.5)
    commercialEnergy = ElectricitySource(20, 120)
    subsetKileen = ElectricitySource(20, 18.5)
    # Estimating cost of mining equipment at scale. If you have a 13 GPU rig, the cost of all the other hardware is roughly the cost of a 14th GPU @ $500
    # Vega 64 Release Date = August 14, 2017
    releaseDate = datetime.datetime.strptime("2017-07-14", "%Y-%m-%d").date()
    # Watts, Hash/Second, Cost, Release Date, Details
    vega64 = MiningRig(203.00, 1965.00, float(
        500 + 500 / 13), releaseDate, "Vega 64")
    symbol = "XMR"  # Monero
    shouldPlot = True
    # estimateCumulativeIncome(symbol, vega64, commercialEnergy, shouldPlot)
    # Determine marginal addition to operating income since release date per added card
    plotRiskCurve(symbol, vega64, subsetKileen)


def plotRiskCurve(coin_symbol, mining_rig, electricity_source):
    maxEfficiency = 0.90
    maxRigs = np.math.floor((electricity_source.energyOutput * 1000.00 *
                             1000.00 * maxEfficiency) / mining_rig.energyConsumption)
    print("Max rigs with 90% efficiency and " + str(mining_rig.energyConsumption) +
          " Watts per rig and " + str(electricity_source.energyOutput) + " MW/h total capacity")
    print("Calculating risk curve for 10 to " +
          str(maxRigs) + " " + mining_rig.details + "s")
    cumulativeIncomes = []
    meanBreakEvenTimes = []
    medianBreakEvenTimes = []
    numRigs = 10
    xAxisData = []
    while numRigs <= maxRigs:
        tempMiningRig = copy.deepcopy(mining_rig)
        tempMiningRig.hashRate = mining_rig.hashRate * numRigs
        tempMiningRig.energyConsumption = mining_rig.energyConsumption * numRigs
        tempMiningRig.price = mining_rig.price * numRigs

        print("Simulating with " + str(numRigs) + " " + mining_rig.details + "s @ " +
              str(tempMiningRig.energyConsumption) + " W/h " + str(tempMiningRig.hashRate) + " H/s $" + str(tempMiningRig.price))

        cumulativeIncome, meanBreakEvenTime, medianBreakEvenTime = estimateCumulativeIncome(
            coin_symbol, tempMiningRig, electricity_source, False)
        cumulativeIncomes.append(cumulativeIncome)
        meanBreakEvenTimes.append(meanBreakEvenTime)
        medianBreakEvenTimes.append(medianBreakEvenTime)
        xAxisData.append(numRigs/10)
        numRigs += 10

    print(cumulativeIncomes)
    print(meanBreakEvenTimes)
    print(medianBreakEvenTimes)
    print(xAxisData)

    # Plot operating income vs # rigs
    # cumulativeIncomes = []
    # cumulativeIncomes = cumulativeIncomes[:len(xAxisData)]
    x = xAxisData
    y = cumulativeIncomes
    low = min(y)
    high = max(y)
    plt.ylim([0, math.ceil(high + 0.5 * (high - low))])
    plt.xlabel('# Rigs (multiples of 10)')
    plt.ylabel('Operating Income (Tens of millions USD)')
    plt.title('Operating Income vs # Rigs')
    label = "$" + str('{0:.2f}'.format(electricity_source.costPerMegawatt)
                      ) + " MW/h + " + str(mining_rig.details)
    plt.bar(x, y, label=label)
    plt.show()

    # Plot cost of rigs
    costOfRigs = []
    numRigs = 10
    while numRigs <= maxRigs:
        tempMiningRig = copy.deepcopy(mining_rig)
        tempMiningRig.hashRate = mining_rig.hashRate * numRigs
        tempMiningRig.energyConsumption = mining_rig.energyConsumption * numRigs
        tempMiningRig.price = mining_rig.price * numRigs
        costOfRigs.append(tempMiningRig.price)
        numRigs += 10
    y = costOfRigs[:len(xAxisData)]
    low = min(y)
    high = max(y)
    plt.ylim([0, math.ceil(high + 0.5 * (high - low))])
    plt.xlabel('# Rigs (multiples of 10 GPUs)')
    plt.ylabel('$ Tens of Millions')
    plt.title('Fixed Costs Estimated')
    label = "$" + str('{0:.2f}'.format(electricity_source.costPerMegawatt)
                      ) + " MW/h + " + str(mining_rig.details)
    plt.bar(x, y, label=label)
    plt.show()

    # Plot energy consumption
    energyConsumption = []
    numRigs = 10
    while numRigs <= maxRigs:
        tempMiningRig = copy.deepcopy(mining_rig)
        tempMiningRig.hashRate = mining_rig.hashRate * numRigs
        tempMiningRig.energyConsumption = mining_rig.energyConsumption * numRigs
        tempMiningRig.price = mining_rig.price * numRigs
        energyConsumption.append(tempMiningRig.energyConsumption)
        numRigs += 10
    y = energyConsumption[:len(xAxisData)]
    low = min(y)
    high = max(y)
    plt.ylim([0, math.ceil(high + 0.5 * (high - low))])
    plt.xlabel('# Rigs (multiples of 10 GPUs)')
    plt.ylabel('10 x MW/h')
    plt.title('Energy Consumption')
    label = "$" + str('{0:.2f}'.format(electricity_source.costPerMegawatt)
                      ) + " MW/h + " + str(mining_rig.details)
    plt.bar(x, y, label=label)
    plt.show()

    # Plot mean break-even time
    y = meanBreakEvenTimes[:len(xAxisData)]
    low = min(y)
    high = max(y)
    plt.ylim([0, math.ceil(high + 0.5 * (high - low))])
    plt.xlabel('# Rigs (multiples of 10 GPUs)')
    plt.ylabel('Days')
    plt.title('Mean Break-Even Time')
    label = "$" + str('{0:.2f}'.format(electricity_source.costPerMegawatt)
                      ) + " MW/h + " + str(mining_rig.details)
    plt.bar(x, y, label=label)
    plt.show()

    #  Plot median break-even time
    y = medianBreakEvenTimes[:len(xAxisData)]
    low = min(y)
    high = max(y)
    plt.ylim([0, math.ceil(high + 0.5 * (high - low))])
    plt.xlabel('# Rigs (multiples of 10 GPUs)')
    plt.ylabel('Days')
    plt.title('Median Break-Even Time')
    label = "$" + str('{0:.2f}'.format(electricity_source.costPerMegawatt)
                      ) + " MW/h + " + str(mining_rig.details)
    plt.bar(x, y, label=label)
    plt.show()


def estimateCumulativeIncome(coin_symbol, mining_rig, electricity_source, shouldPlot):
    """
    This function estimates daily profit for mining given parameters.
    It calculates gross revenue and then subtracts electricity costs.
    Mining infrastructure costs are not taken into consideration.
    """
    if coin_symbol == "XMR":
        if shouldPlot:
            print("Calculating daily Monero mining profits...")
        moneroHashRates, moneroPrices, blockRewards = readData(coin_symbol)
        index = 0
        dailyRevenue = []
        dailyOperatingIncome = []
        totalProfitFigures = []
        while index < len(moneroHashRates):
            totalNetworkHashRate = moneroHashRates[index].rate
            price = moneroPrices[index].rate
            reward = blockRewards[index].rate
            # Monero rewards miners with variable fixed blocks + transaction fees for that block
            estimatedPercentTransactionFees = 0.06479 / 4.51
            blockReward = reward + reward * estimatedPercentTransactionFees
            # Monero rewards blocks every 2 minutes
            rewardsPerDay = (24 * 60 / 2)
            totalProfit = rewardsPerDay * blockReward
            coinRevenue = calculateRevenuePerHash(totalProfit, totalNetworkHashRate,
                                                  mining_rig.hashRate) * mining_rig.hashRate
            revenue = coinRevenue * price  # Convert XMR to USD
            dailyRevenue.append(revenue)
            # Calculate profit
            profit = revenue - mining_rig.energyConsumption * 24 * \
                electricity_source.costPerMegawatt / 1000.00 / 1000.00
            dailyOperatingIncome.append(profit)
            totalProfitFigures.append(totalProfit * price)
            index += 1

        # Plot Operating Income
        dailyProfits = []
        dates = []
        index = 0
        for reward in moneroHashRates:
            dates.append(reward.date)
            dailyProfits.append(dailyOperatingIncome[index])
            index += 1
        # Annotate graph
        releaseDate = mining_rig.releaseDate
        index = 0
        xIndex = 0
        for date in dates:
            if date == releaseDate:
                xIndex = index
            index += 1
        if shouldPlot:
            fig, ax = plt.subplots()
            plt.xlabel('Date')
            years = mdates.YearLocator()  # every year
            months = mdates.MonthLocator()  # every month
            yearsFmt = mdates.DateFormatter('%Y')
            ax.xaxis.set_major_locator(years)
            ax.xaxis.set_major_formatter(yearsFmt)
            ax.xaxis.set_minor_locator(months)
            # ax.set_ylim(0, 10)
            plt.ylabel('Operating Income (USD)')
            plt.title('Operating Income vs Date')
            label = "$" + str('{0:.2f}'.format(electricity_source.costPerMegawatt)
                              ) + " MW/h + " + str(mining_rig.details)
            plt.bar(dates, dailyOperatingIncome, label=label)
            # Annotate graph
            ax.annotate('Release Date', xy=(releaseDate, dailyOperatingIncome[xIndex]),
                        xytext=(releaseDate,
                                dailyOperatingIncome[xIndex] * 0.6),
                        arrowprops=dict(facecolor='white', shrink=0.05))
            hardForkDate = datetime.datetime.strptime(
                "2018-04-06", "%Y-%m-%d").date()
            index = 0
            hardForkIndex = 0
            for date in dates:
                if date == hardForkDate:
                    hardForkIndex = index
                index += 1
            hardForkDateText = datetime.datetime.strptime(
                "2018-04-30", "%Y-%m-%d").date()
            ax.annotate('      ASIC hard \n      fork', xy=(hardForkDate, dailyOperatingIncome[hardForkIndex]),
                        xytext=(hardForkDateText,
                                dailyOperatingIncome[hardForkIndex] * 0.7),
                        arrowprops=dict(facecolor='white', shrink=0.05))
            ax.legend(loc='best')
            plt.show()
        # ax.plot(dates, dailyRevenueFigures, label='Revenue')
        # Plot EMA
        # dataFrame = pd.DataFrame(np.array(dailyProfits))
        # ema_short = dataFrame.ewm(span=10, adjust=False).mean()
        # ax.plot(dates, ema_short, label='20 day EMA')

        # Determine time to turn a profit on mining hardware
        cumulativeOperatingIncome = []
        index = 0
        for operatingIncome in dailyOperatingIncome:
            cumulativeOperatingIncome.append(
                sumRange(dailyOperatingIncome, 0, index))
            index += 1
        xAxisData = []
        for reward in moneroHashRates:
            xAxisData.append(reward.date)
        if shouldPlot:
            fig, ax = plt.subplots()
            ax.plot(xAxisData, cumulativeOperatingIncome, label=label)
            ax.legend(loc='best')
            plt.xlabel('Date')
            years = mdates.YearLocator()  # every year
            months = mdates.MonthLocator()  # every month
            yearsFmt = mdates.DateFormatter('%Y')
            ax.xaxis.set_major_locator(years)
            ax.xaxis.set_major_formatter(yearsFmt)
            ax.xaxis.set_minor_locator(months)
            plt.ylabel('Operating Income (USD)')
            plt.title('Cumulative Operating Income')
            # Annotate Graph
            ax.annotate('ASIC hard fork', xy=(hardForkDate, cumulativeOperatingIncome[hardForkIndex]),
                        xytext=(
                            hardForkDate, cumulativeOperatingIncome[hardForkIndex] * 0.7),
                        arrowprops=dict(facecolor='white', shrink=0.05))
            ax.annotate('Release Date August 14, 2017', xy=(releaseDate, cumulativeOperatingIncome[xIndex]),
                        xytext=(releaseDate,
                                cumulativeOperatingIncome[xIndex] * 0.6),
                        arrowprops=dict(facecolor='white', shrink=0.05))
            # Draw Start & Break even points
            startLine = []
            breakEvenLine = []
            for x in xAxisData:
                startLine.append(cumulativeOperatingIncome[xIndex])
                breakEvenLine.append(
                    cumulativeOperatingIncome[xIndex] + mining_rig.price)
            ax.plot(xAxisData, breakEvenLine, label='Break Even')
            ax.plot(xAxisData, startLine, label='Earliest Start Date')
            ax.legend(loc='best')
            plt.show()

        # Calculate mean & median return since release date
        meanOperatingIncome = s.mean(dailyOperatingIncome[xIndex:])
        medianOperatingIncome = s.median(dailyOperatingIncome[xIndex:])

        print("Mean daily operating income since " +
              str(releaseDate) + ": $" + str(meanOperatingIncome))
        print("Median daily operating income since " +
              str(releaseDate) + ": $" + str(medianOperatingIncome))
        # Calculate break even time
        breakEvenDaysMean = mining_rig.price / meanOperatingIncome
        breakEvenDaysMedian = mining_rig.price / medianOperatingIncome

        print("Mean break even days: " + str(breakEvenDaysMean))
        print("Median break even days: " + str(breakEvenDaysMedian))

        # Plot hash rate vs date
        xAxisData = []
        yAxisData = []
        for reward in moneroHashRates:
            xAxisData.append(reward.date)
            yAxisData.append(reward.rate)
        if shouldPlot:
            fig, ax = plt.subplots()
            ax.plot(xAxisData, yAxisData)
            plt.xlabel('Date')
            years = mdates.YearLocator()  # every year
            months = mdates.MonthLocator()  # every month
            yearsFmt = mdates.DateFormatter('%Y')
            ax.xaxis.set_major_locator(years)
            ax.xaxis.set_major_formatter(yearsFmt)
            ax.xaxis.set_minor_locator(months)
            plt.ylabel('Hash / Second')
            plt.title('Monero Network Hash Rate vs Date')
            plt.show()

            # Plot price vs date
            xAxisData = []
            yAxisData = []
            for reward in moneroPrices:
                xAxisData.append(reward.date)
                yAxisData.append(reward.rate)
            fig, ax = plt.subplots()
            ax.plot(xAxisData, yAxisData)
            plt.xlabel('Date')
            years = mdates.YearLocator()  # every year
            months = mdates.MonthLocator()  # every month
            yearsFmt = mdates.DateFormatter('%Y')
            ax.xaxis.set_major_locator(years)
            ax.xaxis.set_major_formatter(yearsFmt)
            ax.xaxis.set_minor_locator(months)
            plt.ylabel('Price (USD)')
            plt.title('Monero Price vs Date')
            plt.show()

            # Plot reward vs date
            xAxisData = []
            yAxisData = []
            for reward in blockRewards:
                date = parser.parse(reward.date)
                xAxisData.append(date)
                yAxisData.append(reward.rate)
            fig, ax = plt.subplots()
            ax.plot(xAxisData, yAxisData)
            plt.xlabel('Date')
            years = mdates.YearLocator()  # every year
            months = mdates.MonthLocator()  # every month
            yearsFmt = mdates.DateFormatter('%Y')
            ax.xaxis.set_major_locator(years)
            ax.xaxis.set_major_formatter(yearsFmt)
            ax.xaxis.set_minor_locator(months)
            plt.ylabel('Block Reward (XMR)')
            plt.title('Monero Block Reward vs Date (Adjusted)')
            plt.show()
    elif coin_symbol == "BTC":
        print("Calculating daily Bitcoin mining profits...")
        # TODO Do bitcoin calculation
    else:
        print("Coin symbol not supported.")

    # Return cumulative return from release date, onward
    totalOperatingIncome = sum(dailyOperatingIncome[xIndex:])
    return totalOperatingIncome, breakEvenDaysMean, breakEvenDaysMedian


def readData(coin_symbol):
    debug = False
    if coin_symbol == "XMR":
        # Read in historical network hash rate into array
        fixedBlockRewardFileName = "./data/monerohashratemodified.txt"
        file1 = open(fixedBlockRewardFileName, "r")
        hashRateLine = file1.readlines()
        hashRateLine = "".join(str(x) for x in hashRateLine)
        file1.close()
        hashRates = hashRateLine.split(",")
        hashRates = list(map(','.join, zip(hashRates[::2], hashRates[1::2])))
        pricingDate = datetime.datetime.strptime("2014/06/04",
                                                 "%Y/%m/%d").date()  # Earliest date we have for Monero prices
        moneroHashRates = []
        for rate in hashRates:
            temp = rate[rate.find("\"") + 1:]
            dateString = temp[:temp.find("\"")]
            hashRate = temp[temp.find(",") + 1: len(temp) - 1]
            date = datetime.datetime.strptime(dateString, "%Y/%m/%d").date()
            if date >= pricingDate:
                if hashRate == 'null':
                    hashRate = 0
                moneroHashRates.append(HashRate(date, float(hashRate)))
                latestHashDate = date
        if debug:
            print(str(len(moneroHashRates)) + " days of hash rates")

        # Read in historical pricing data
        fixedBlockRewardFileName = "./data/moneroprice.txt"
        file1 = open(fixedBlockRewardFileName, "r")
        hashRateLine = file1.readlines()
        hashRateLine = "".join(str(x) for x in hashRateLine)
        file1.close()
        hashRates = hashRateLine.split(",")
        hashRates = list(map(','.join, zip(hashRates[::2], hashRates[1::2])))
        moneroPrices = []
        for rate in hashRates:
            temp = rate[rate.find("\"") + 1:]
            dateString = temp[:temp.find("\"")]
            date = datetime.datetime.strptime(dateString, "%Y/%m/%d").date()
            hashRate = temp[temp.find(",") + 1: len(temp) - 1]
            if date <= latestHashDate:
                moneroPrices.append(HashRate(date, float(hashRate)))
        # print(str(len(moneroPrices)) + " days of monero prices")

        # Read in fixed block reward into array
        fixedBlockRewardFileName = "./data/monerofixedblockreward.txt"
        file1 = open(fixedBlockRewardFileName, "r")
        fixedBlockRewards = file1.readlines()
        file1.close()
        # print(str(len(fixedBlockRewards)) + " days of block rewards")

        # Find earliest pricing date index
        # Find latest hash rate date index
        fixedBlockRewardDateFileName = "./data/moneroblockrewarddate.txt"
        file1 = open(fixedBlockRewardDateFileName, "r")
        fixedBlockRewardDates = file1.readlines()
        file1.close()
        if debug:
            print("Pruning reward dates...")
        earliestIndex = 0
        latestIndex = 0
        index = 0
        if debug:
            print(str(pricingDate) + " to " + str(latestHashDate))
        for line in fixedBlockRewardDates:
            line = line.strip()
            date = datetime.datetime.strptime(line, "%Y-%m-%d").date()
            if date == latestHashDate:
                latestIndex = index
            elif date == pricingDate:
                earliestIndex = index
            index += 1
        fixedBlockRewards = fixedBlockRewards[earliestIndex:latestIndex + 1]
        fixedBlockRewardDates = fixedBlockRewardDates[earliestIndex:latestIndex + 1]
        blockRewards = []
        index = 0
        while index < len(fixedBlockRewards):
            # Account for mining time split - anytime before 2016-03-23 must be doubled
            reward = float(fixedBlockRewards[index])
            if fixedBlockRewardDates[index] < "2016-03-23":
                reward = reward * 2
            blockRewards.append(HashRate(fixedBlockRewardDates[index], reward))
            index += 1
        if debug:
            print(str(len(fixedBlockRewards)) + " days of block rewards")
        return moneroHashRates, moneroPrices, blockRewards
    else:
        print("Symbol not supported!")


def sumRange(L, a, b):
    sum = 0
    for i in range(a, b + 1, 1):
        sum += L[i]
    return sum


def calculateRevenuePerHash(totalProfit, totalNetworkHashRate, marginalHashRate):
    return totalProfit / (totalNetworkHashRate + marginalHashRate)


class MiningRig:
    energyConsumption = 0  # Watts
    hashRate = 0  # Hash / Second
    price = 0
    releaseDate = ""
    details = ""

    def __init__(self, energyConsumption, hashRate, price, releaseDate, details):
        self.energyConsumption = energyConsumption
        self.hashRate = hashRate
        self.price = price
        self.releaseDate = releaseDate
        self.details = details


class ElectricitySource:
    energyOutput = 0  # MW MegaWatts/h
    costPerMegawatt = 0.0  # $ / MegaWatt/h

    def __init__(self, energyOutput, costPerMegawatt):
        self.energyOutput = energyOutput
        self.costPerMegawatt = costPerMegawatt


class HashRate:
    date = ""  # YYYY/MM/DD
    rate = 0  # H/s

    def __init__(self, date, rate):
        self.date = date
        self.rate = rate


if __name__ == "__main__":
    main()
