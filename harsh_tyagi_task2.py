from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.context import SparkContext, SparkConf
import csv
import sys
import time
import math

spark = None
sc = None
# "yelp_train.csv"  # sys.argv[3] "test_trainRating.csv"
inputfile = sys.argv[1]  # "yelp_train.csv"
# "yelp_val.csv"  # sys.argv[3] "test_testRating.csv"
validationfile = sys.argv[2]  # "yelp_val.csv"
items = None
dictUid = {}
dictBid = {}
list_unaccounted = []
outputFile = sys.argv[4]  # "output_task2.csv"
dict_code_uid = {}
dict_code_bid = {}
t = None
case = float(sys.argv[3])


def initialize():
    global sc, spark, inputfile, t, items, validationfile, dictUid, dictBid, list_unaccounted, dict_code_uid, dict_code_bid, t, case
    t = time.time()
    sc_conf = SparkConf()
    sc_conf.setAppName("Task2")
    sc_conf.setMaster('local[*]')
    # sc_conf.set("spark.driver.bindAddress", "127.0.0.1")
    sc = SparkContext(conf=sc_conf)
    sc.setLogLevel("ERROR")
    csvread = sc.textFile(inputfile)
    columnName = csvread.first().split(',')
    items = csvread.map(lambda line: line.split(",")).filter(
        lambda line: (line) != columnName)

    # ------------Reading evaluation data-----------
    csvread2 = sc.textFile(validationfile)
    columnName2 = csvread2.first().split(',')
    validationData = csvread2.map(lambda line: line.split(",")).filter(
        lambda line: (line) != columnName2)

    # calling case 3:
    if case == 3:
        implement_case3(items, validationData)
        print("Duration: "+str(time.time()-t))
        return
    # calling case 2:
    if case == 2:
        implement_case2(items, validationData)
        print("Duration: "+str(time.time()-t))
        return
    # Ending case 2

    # ------------PreProcessing data for training the mode-----------
    if case == 1:
        bid_uid = items.map(lambda u: (u[0], u[1]))

        keys = list(set(bid_uid.keys().collect()))
        values = list(set(bid_uid.values().collect()))

        dictUid = dict(zip(keys, range(0, len(keys))))
        dictBid = dict(zip(values, range(0, len(values))))
        for k, v in dictUid.items():
            dict_code_uid[v] = k

        for k, v in dictBid.items():
            dict_code_bid[v] = k

        ratings = items.map(lambda l: Rating(
            int(dictUid[l[0]]), int(dictBid[l[1]]), float(l[2])))

        # Training the model on train data
        rank = 2
        lambd = 0.5
        numIterations = 10
        model = ALS.train(ratings, rank, numIterations, lambd)

        print("Total entries in validation data: " +
              str(len(validationData.collect())))
        # ----------------------Creating a map with integer values for users and business on validation test set-----------------

        test_on_validation = validationData.map(lambda p: mapData(p))
        #
        validationRating = test_on_validation.filter(
            lambda p: (p[0] == 1)).map(lambda r: (r[1][0], r[1][1], r[1][2]))

        accountedPairs = test_on_validation.filter(
            lambda p: (p[0] == 1)).map(lambda r: (r[1][0], r[1][1]))

        UnaccountedPairs = test_on_validation.filter(
            lambda p: p[0] == 0).map(lambda r: ((r[1][0], r[1][1]), 2.75))

        # print("Accounted Pairs: "+str(len(accountedPairs.collect())))

        # print("Unaccounted Pairs: "+str(len(UnaccountedPairs.collect())))
        # print(test_on_validation.count())
        # print("Unaccounted Pairs: "+str(len(list_unaccounted)))

        # ----------------------Evaluate the model on training data----------------------
        # testdata = ratings.map(lambda p: (p[0], p[1]))
        # predictions = model.predictAll(testdata).map(8
        #     lambda r: ((r[0], r[1]), r[2]))
        # ratesAndPreds = ratings.map(lambda r: (
        #     (r[0], r[1]), r[2])).join(predictions)
        # MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()

        # # import validation data

        # print("Mean Squared Error = " + str(MSE))

        # ----------------------Evaluate the model on testing data----------------------
        predictions = model.predictAll(accountedPairs).map(
            lambda r: ((r[0], r[1]), r[2]))
        # print(len(predictions.collect()))
        finalpred = predictions.union(UnaccountedPairs)
        # print(len(finalpred.collect()))
        # return
        # ratesAndPreds = validationRating.map(lambda r: (
        #     (r[0], r[1]), r[2])).join(predictions)
        ratesAndPreds = validationRating.map(lambda r: (
            (r[0], r[1]), r[2])).join(finalpred)
        MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        writeToFile(finalpred)
        rmse = math.sqrt(MSE)
        print("Root Mean Squared Error = " + str(rmse))
    print("Duration: "+str(time.time()-t))
# -------------------Mapping Data For Case 1--------------------------


def mapcheck(l):
    global dict_code_uid, dict_code_bid
    try:
        return list((dict_code_uid[l[0][0]], dict_code_bid[l[0][1]], l[1]))
    except:
        return list((l[0][0], l[0][1], l[1]))


def writeToFile(rdd):
    global outputFile, dict_code_uid, dict_code_bid
    mapper = rdd.map(lambda l: mapcheck(l))

    lines = mapper.collect()
    # print(lines[0:2])
    # return
    print("Writing to File...")
    l2 = []
    with open(outputFile, 'w') as writeFile:
        writeFile.write("user_id, business_id, prediction\n")
        lines.sort(key=lambda x: (x[0], x[1]))
        print(len(lines))
        for each in lines:
            writeFile.write(str(each[0])+"," +
                            str(each[1])+","+str(each[2])+"\n")
    return len(lines)


def mapData(p):
    global dictUid, dictBid, list_unaccounted
    if((p[0] in dictUid) and (p[1] in dictBid)):
        return (1, (dictUid[p[0]], dictBid[p[1]], float(p[2])))
    else:
        return(0, (p[0], p[1], float(p[2])))
    # return p

# -------------------Case 2--------------------------


def implement_case2(items, validationData):
    global sc
    bui_user = items.map(lambda x: (x[1], x[0])).groupByKey(
    ).mapValues(list).collectAsMap()
    user_bui = items.map(lambda x: (
        x[0], x[1])).groupByKey().mapValues(set).collectAsMap()

    users = user_bui.keys()
    list_lampard = []
    dict_ratings = {}
    for each in items.collect():
        dict_ratings[(each[0], each[1])] = each[2]
    # get_item_sum = items.map(lambda x: (x[1], float(x[2]))).reduceByKey(
    #     lambda x, y: x+y).collectAsMap()
    # dict_ratingSum = {}
    # for k, v in user_bui.items():
    #     dict_ratingSum[k] = 0
    #     for each in list(v):
    #         dict_ratingSum[k] += float(dict_ratings[k, each])
    prediction_lines = []
    for validate in validationData.collect():
        user = str(validate[0])
        business = str(validate[1])
        list_pearson = []
        if(not user_bui.get(user)):
            # ratings = float(
            #     float(get_item_sum[user])/float(len(user_bui[user])))
            prediction_lines.append((user, business, 2.75))
            continue
        if(not bui_user.get(business)):
            # since 3.8 is almost the average
            prediction_lines.append((user, business, 2.75))
            continue
        lexical = user_bui[user]
        flag = False
        var_flag = 0.0
        if(business in lexical):
            flag = True
            var_flag = float(dict_ratings[(user, business)])
            lexical.remove(business)
        sum_all = 0.0
        for b in lexical:
            sum_all += float(dict_ratings[(user, b)])
            # try:
            #     sum_all += float(dict_ratings[(user, b)])
            # except:
            #     sum_all += 2.5
            #     pass
        avg_u = float(sum_all/len(lexical))
        for eachuser in bui_user[business]:
            inter = user_bui[eachuser] & lexical
            # if(business in inter):
            #     inter.remove(business)
            sum1 = 0.0
            sum2 = 0.0
            length = len(inter)
            if(length == 0):
                continue
            for i in inter:
                sum1 = sum1+float(dict_ratings[(eachuser, i)])
                sum2 = sum2+float(dict_ratings[(user, i)])
            average_v = float(float(sum1)/float(length))

            average_u = float(float(sum2)/float(length))
            # average_u = avg_u
            numerator = 0.0
            denominator = 0.0
            for i in inter:
                numerator += (float(dict_ratings[(user, i)])-float(average_u)) * \
                    (float(dict_ratings[(eachuser, i)])-float(average_v))
            denom1 = 0.0
            denom2 = 0.0
            for j in inter:
                denom1 += ((float(dict_ratings[(user, j)]
                                  )-float(average_u))**2)
                denom2 += ((float(dict_ratings[(eachuser, j)]
                                  )-float(average_v))**2)
            denominator = 0.0
            denominator = float((math.sqrt(float(denom1)))
                                * (math.sqrt(float(denom2))))
            w_u_v = 0.0
            if denominator == 0.0:
                w_u_v = 0.15
            else:
                w_u_v = float(float(numerator)/float(denominator))
            list_pearson.append(
                (float(dict_ratings[(eachuser, business)]), average_v, w_u_v))
        numerator = 0.0
        denominator = 0.0
        business_for_u = user_bui[user]
        sum_u = 0.0
        predict = 0.0
        if(business in business_for_u):
            business_for_u.remove(business)
        for b in business_for_u:
            sum_u += float(dict_ratings[(user, b)])
        avg_u = float(float(sum_u)/float(len(business_for_u)))
        for entry in list_pearson:
            numerator += float((float(entry[0]) -
                                float(entry[1])) * float(entry[2]))
            denominator += float(abs(float(entry[2])))
        predict = float(avg_u)
        if denominator != 0.0:
            predict += (float(numerator/denominator))
        if flag:
            predict = var_flag
        # # Skewing results to clip between 0 and 5
        if predict < 1.0:
            predict = 1.0
        elif predict > 5.0:
            predict = 5.0
        # predict = round(predict, 3)
        prediction_lines.append((user, business, predict))
    writingToFileCase2(prediction_lines)
    calculatingRmse(prediction_lines, validationData)
    print("Finished")
    # print(dict_ratings)
    # print(bui_user.first())
    # print(user_bui.first())
    return


def calculatingRmse(prediction_lines, validationfile):
    global sc
    map_pred = sc.parallelize(prediction_lines).map(
        lambda l: ((l[0], l[1]), float(l[2])))
    # print(map_pred.first())

    ratesAndPreds = validationfile.map(
        lambda r: ((r[0], r[1]), float(r[2]))).join(map_pred)
    # print(ratesAndPreds.first())
    MSE = ratesAndPreds.map(lambda r: (
        float(r[1][0]) - float(r[1][1]))**2).mean()
    rmse = math.sqrt(MSE)
    print("Root Mean Squared Error = " + str(rmse))


def writingToFileCase2(lines):
    # print(lines[0:2])
    # return
    print("Writing to File...")
    with open(outputFile, 'w') as writeFile:
        writeFile.write("user_id, business_id, prediction\n")
        lines.sort(key=lambda x: (x[0], x[1]))
        print(len(lines))
        for each in lines:
            writeFile.write(str(each[0])+"," +
                            str(each[1])+","+str(each[2])+"\n")
    return len(lines)


# ------------------------------------------------------------Case 3--------------------------------------------------------------

def implement_case3(items, validationData):
    global sc
    bui_user = items.map(lambda x: (x[1], x[0])).groupByKey(
    ).mapValues(set).collectAsMap()
    user_bui = items.map(lambda x: (
        x[0], x[1])).groupByKey().mapValues(list).collectAsMap()

    # get_item_sum = items.map(lambda x: (x[1], float(x[2]))).reduceByKey(
    #     lambda x, y: x+y).collectAsMap()  # ).mapValues(set).collectAsMap()

    # creating average business size
    # sum_size = items.map(lambda x: (x[1], x[0])).groupByKey().mapValues(list).map(
    #     lambda x: (1, len(x[1])))
    # count = sum_size.count()
    # sum_for_b = sum_size.reduceByKey(lambda x, y: x+y).first()[1]
    # average = float(sum_for_b/count)
    # print(average)
    # return
    # Creating dictionary for (user,business)->rating

    users = user_bui.keys()
    list_lampard = []
    dict_ratings = {}
    for each in items.collect():
        dict_ratings[(each[0], each[1])] = each[2]
    prediction_lines = []

    # Calculating Pearson Correlation
    for validate in validationData.collect():
        user = str(validate[0])
        business = str(validate[1])
        list_pearson = []
        if(not user_bui.get(user)):
            # ratings = float(
            #     float(get_item_sum[business])/float(len(bui_user[business])))
            prediction_lines.append((user, business, 2.75))
            continue
        if(not bui_user.get(business)):
            # since 3.8 is almost the average
            prediction_lines.append((user, business, 2.75))
            continue
        lexical = bui_user[business]
        # Removing the user if it exists in the user_list for that business, because that is what we have to predict
        if(user in lexical):
            lexical.remove(user)

        neighbourhood_size = 3
        for bus in user_bui[user]:
            inter = bui_user[bus] & lexical
            # if(user in inter):
            #     inter.remove(user)
            # inter = bui_user[bus] & bui_user[business]
            # Calculating average of the items being compared on corated
            sum1 = 0
            sum2 = 0
            length = len(inter)
            if(length == 0):
                list_pearson.append(
                    (float(dict_ratings[(user, bus)]), 0.1))
                continue
            for i in inter:
                sum1 = sum1+float(dict_ratings[(i, bus)])
                sum2 = sum2+float(dict_ratings[(i, business)])
            average_j = float(float(sum1)/float(length))

            average_i = float(float(sum2)/float(length))

            # Calculating Pearson Correlation
            numerator = 0.0
            denominator = 0.0
            for i in inter:
                numerator += float((float(dict_ratings[(i, business)])-float(average_i)) *
                                   (float(dict_ratings[(i, bus)])-float(average_j)))
            denom1 = 0.0
            denom2 = 0.0
            w_i_j = 0.0
            for j in inter:
                denom1 += float(
                    (float(dict_ratings[(j, business)])-float(average_i))**2)
                denom2 += float((float(dict_ratings[(j, bus)]
                                       )-float(average_j))**2)
            denominator = 0.0
            denominator = float((math.sqrt(float(denom1)))
                                * (math.sqrt(float(denom2))))
            if denominator == 0.0:
                w_i_j = 1
            else:
                w_i_j = float(float(numerator)/float(denominator))

            list_pearson.append(
                (float(dict_ratings[(user, bus)]), float(w_i_j)))
            # Appending pearson values to a list
            # if(w_i_j > 0):
            #     list_pearson.append(
            #         (float(dict_ratings[(user, bus)]), w_i_j))

            # End of Pearson Correlation
        # Prediction for the user, business pair
        numerator = 0.0
        denominator = 0.0
        predict = 0.0
        list_pearson.sort(key=lambda x: (-x[1]))

        # list_pear = copy.copy(list_pearsob)
        len_total = len(list_pearson)
        # neighbourhood_size = 2
        neighbourhood_size = int(len_total/2)
        for entry in list_pearson[0:neighbourhood_size]:
            numerator += float(float(entry[0]) * float(entry[1]))
            denominator += float(abs(float(entry[1])))

        if denominator != 0:
            predict += (float(numerator/denominator))
            if(predict < 1):
                predict = 1.0
            elif(predict > 5):
                predict = 5
        else:
            predict += 2.75  # float(float(entry[0]))

        prediction_lines.append((user, business, predict))

    writingToFileCase2(prediction_lines)
    calculatingRmse(prediction_lines, validationData)
    print("Finished")


def main():
    initialize()


if __name__ == "__main__":
    print("Started....")
    main()

    # print("Completed")
    pass
