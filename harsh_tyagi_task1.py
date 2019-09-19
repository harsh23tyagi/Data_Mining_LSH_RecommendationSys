#
# ----------------------------------------------------Imports-----------------------------------------------------------------
from pyspark.context import SparkContext, SparkConf
import sys
from collections import OrderedDict
import csv
from itertools import combinations
from itertools import product
import copy
import time
import random
import csv

#
# --------------------------------------------------Global Variables----------------------------------------------------------
spark = None
items = None
inputfile = sys.argv[1]  # "yelp_train.csv"
gidDict = {}
n = 120
b = 40
r = 3
candidateTuple = None
bids = []
mainCount = 0
t = None
m = 0
hashedList = []
outputFile = sys.argv[2]  # "output1.csv"
listvala = []
listvalb = []
sc = None
#
# --------------------------------------------------Initialize----------------------------------------------------------


def initialize():
    global sc, spark, items, inputfile, t, m, gidDict, bids, hashedList, n, b, r, candidateTuple, listvala, listvalb
    print("Initializing...")

    t = time.time()
    candidateList = []
    frequentList = []
    sc_conf = SparkConf()
    sc_conf.setAppName("Task1")
    sc_conf.setMaster('local[*]')
    # sc_conf.set("spark.driver.bindAddress", "127.0.0.1")
    sc = SparkContext(conf=sc_conf)
    sc.setLogLevel("ERROR")
    csvread = sc.textFile(inputfile)
    columnName = csvread.first().split(',')
    items = csvread.map(lambda line: line.split(",")).filter(
        lambda line: (line) != columnName)
    #  column name is userid, businessid, starts

    userids = list(set(sorted(items.keys().collect())))
    k = 0
    for user in userids:
        if(user not in gidDict):
            gidDict[user] = k
            k = k+1
    # print(k)

    bids = list(set(sorted(items.values().collect())))
    # bids = copy.copy(sorted(bids))
    # print(len(bids))
    m = len(userids)
    listvala = random.sample(range(1, m), n)
    listvalb = random.sample(range(1, m), n)
    bid_uid = items.map(lambda x: (
        (x[1], x[0]), 1)).reduceByKey(lambda x, y: x+y).map(lambda x: (x[0])).groupByKey().mapValues(list)

    bid_uid_hashed = bid_uid.map(lambda x: initialHash(x))

    dict_uniques = {}
    for each in bid_uid.collect():
        dict_uniques[each[0]] = set(each[1])

    bid_uid_hashed2 = bid_uid_hashed.map(
        lambda x: hashing(x))
    # print(bid_uid_hashed2.first())

    # creating signature matrix column per business IDs
    start = 0
    end = r
    tempSim = []
    finalList = []
    hashedListSet = bid_uid_hashed2.collect()
    length = len(hashedListSet)
    c = 1
    print("Finding similar pairs...")

    dictionEvery = {}

    while(end <= n):
        tempDict = []
        for each in hashedListSet:
            templist = sorted(each[1][start:end])
            tempDict.append((tuple(templist), each[0]))
            # tempDict.append((tuple(each[1][start:end]), each[0]))
        dictionEvery[c] = tempDict
        c = c+1
        start = end
        end = end+r
    dictionaryCheck = {}

    # for i in range(1, b+1):
    #     dictionaryCheck = {}
    #     for i in range(0, )

    length = len(dictionEvery[1])
    candidateset = []
    candidateTuple = []
    print("Working on Bands 1 to 40 ")
    for i in range(1, b+1):
        justBid = []
        dictionBand = dictionEvery[i]
        # print("Working on Band: "+str(i))
        mapper = sc.parallelize(dictionBand).groupByKey().mapValues(
            list).filter(lambda x: (len(x[1]) > 1))

        justBid = mapper.map(lambda c: c[1]).collect()
        candidateTuple.append(justBid)
        # print(justBid)

    # print(len(candidateTuple[0]))
    # print(len(candidateTuple[1]))
    # print((candidateTuple[1]))

    candidateset = (candidateTuple)  # it was list(set(candidateTuple))

    candidatepairs = []
    count = 0
    for each in candidateset:
        for e in each:
            l1 = list(combinations(sorted(e), 2))
            candidatepairs.extend(l1)

    candPairSet = []  # set()

    candPairSet = (candidatepairs)  # it was list(set(candidatepairs))
    lines = []
    print("Finding final Jaccard Simmilarity")
    finalPairs = []
    for each in candPairSet:
        set1 = dict_uniques[each[0]]
        set2 = dict_uniques[each[1]]
        inter = set1 & set2
        # print(len(inter), len(set1), len(set2))
        jaccard = (float(len(inter)))/(float(len(set1.union(set2))))
        # print(jaccard)
        if(jaccard >= 0.5):
            # print(jaccard)
            lines.append([each[0], each[1], jaccard])
            finalPairs.append(each)

    # print(len(list(set(finalPairs))))
    # print(len((finalPairs)))
    answer = writeToFile(lines)
    # calculatingPreRec(lines)
    print("Total Items Printed: "+str(answer))
    print("Duration: "+str(time.time()-t))
    # print("Finished")


def writeToFile(lines):
    global outputFile
    print("Writing to File...")
    l2 = []
    with open(outputFile, 'w') as writeFile:
        # writer = csv.writer(writeFile)
        writeFile.write("business_id_1, business_id_2, simmilarity\n")
        # l2.append(['business_id_1', 'business_id_2', 'simmilarity'])
        # writer.writerows(l2)
        # lines.sort(key=lambda x: (x[0], x[1]))

        setFinal = set()
        for each in lines:
            setFinal.add(tuple(each))
        mainList = list(setFinal)
        mainList.sort(key=lambda x: (x[0], x[1]))
        l2 = []
        for each in mainList:
            # l2.append(list(each))
            writeFile.write(str(each[0])+"," +
                            str(each[1])+","+str(each[2])+"\n")

        # writer.writerows(l2)
        return len(setFinal)


def hashing(x):
    global gidDict, n, listvala, listvalb, m
    finalList = []
    for a in range(1, n+1):
        tempList = []
        for each in x[1]:
            rem = (((listvala[a-1]*each)+listvalb[a-1])) % m
            tempList.append(rem)
        finalList.append(min(tempList))
    return (x[0], (list(finalList)))


def initialHash(x):
    global gidDict
    finalList = []
    for each in x[1]:
        finalList.append(gidDict.get(each))
    return (x[0], sorted(finalList))


def calculatingPreRec(lines):
    # false postive- mylist - original list
    # false negative- original - mylist
    # true positives = total - false positives

    global sc
    mapper = sc.parallelize(lines).map(lambda x: (x[0], x[1]))
    # print(mapper.first())

    # reading ground truth values
    csvread = sc.textFile("pure_jaccard_similarity.csv")
    columnName = csvread.first().split(',')
    its = csvread.map(lambda line: line.split(",")).filter(
        lambda line: (line) != columnName).map(lambda x: (x[0], x[1]))
    true_positives = len(its.collect())
    false_negatives = len(its.subtract(mapper).collect())
    false_positives = len(mapper.subtract(its).collect())
    print("False-negatives: "+str(false_negatives))
    print("False-Positives: "+str(false_positives))
    precision = float(true_positives)/float(true_positives+false_positives)
    recall = float(true_positives)/float(true_positives+false_negatives)
    print("Precision: "+str(precision))
    print("Recall: "+str(recall))

# -------------------------------------------------- Main Function Call --------------------------------------------------


def main():
    initialize()


if __name__ == "__main__":
    print("Started....")
    main()
    # print("Completed")
    pass
