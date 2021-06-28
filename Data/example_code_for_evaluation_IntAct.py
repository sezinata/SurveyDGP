sum_disease_AUC = 0
sum_disease_AUCList = []
sum_disease_AUPRC = 0
sum_disease_AUPRCList = []
sum_cancerdisease_AUPRC = 0
sum_cancerdisease_AUC = 0
out_folder = dataset_choice + 'scores' + str(repeat) + 'x5folds' + method_choice[met] + "" + operation_choice[choice]

new_file = open("ndcg" + str(repeat) + "X5" + dataset_choice + ".txt", mode="a+")
new_file.write("Method " + "Operation " + "Disease " + "p@" + str(top_k) + " " + "r@" + str(top_k) + " " + "ap@" + str(
    top_k) + " " + "map " + "ndcg@" + str(
    top_k) + " " + "balanced_acc. " + "AUC " + "scikit_AreaPR " + "AUPRC " + "stdeAUC " + "stdeAUPRC" + " stdAUC " + " stdAUPRC" + '\n')
sum_disease_areaPR = 0
sum_disease_b_acc = 0

sum_disease_ndcg = 0
sum_disease_pk = 0
sum_disease_rk = 0
sum_disease_apk = 0
sum_disease_mapk = 0
for dis in range(len(diseases)):
    disease = diseases[dis]
    sumAUCList = []
    sumAUCPRList = []
    sumaverages = 0
    sumaveragesAUCPR = 0
    sumaverages_ndcg = 0
    sumaverages_balanced_accuracy = 0
    for rep in range(repeat):
        rep += 1
        sumaverages_ndcg = 0
        sumaverages_pk = 0
        sumaverages_rk = 0
        sumaverages_apk = 0
        sumaverages_mapk = 0
        sumaverages_b_acc = 0

        sumaverages_areaPR = 0
        sum_ndcg = 0
        sum_pk = 0
        sum_rk = 0
        sum_apk = 0
        sum_mapk = 0

        sumAUC = 0
        sumAUCPR = 0
        sum_areaPR = 0
        sum_b_acc = 0
        for j in range(5):
            j += 1
            trainIndices = []
            train_IndiceFile = pathTest + "/" + dataset_choice + "/" + disease + "" + dataset + "" + str(
                rep) + "" + str(j) + "indicesofTrainSet.txt"
            fR = open(train_IndiceFile, 'r')
            for line in fR:
                trainIndices.append(int(line.strip()))
            xtrain = []  # torch.Tensor(torch.ones(len(trainIndices),resultOfModel1.size(1)))

            train_label_file = pathTest + "/" + dataset_choice + "/" + disease + dataset + "" + str(
                rep) + "" + str(j) + "trainLabel.txt"
            ytrain = []
            fR = open(train_label_file, 'r')

            index = 0
            trainindicesused = []
            for line in fR:  # as reading train labels
                i = trainIndices[index]  # i is node name in ppi, since in ppi nodes reprsented by indices
                i = str(i)
                if i in node_embed_dict:  # for nodes which do not have any neighbors to be trained in none of views/networks, to check singleton nodes
                    xtrain.append(node_embed_dict[i])
                    ytrain.append(int(line.strip()[0]))  # if it is in xtrain then it must be also in ytrain
                    trainindicesused.append(int(i))
                index += 1

            xtrain = np.array(xtrain)
            ytrain = np.array(ytrain)
            trainindicesused = np.array(trainindicesused)

            testIndices = []
            test_IndiceFile = pathTest + "/" + dataset_choice + "/" + disease + "" + dataset + "" + str(
                rep) + "" + str(j) + "indicesofTestSet.txt"
            fR = open(test_IndiceFile, 'r')
            for line in fR:
                testIndices.append(int(line.strip()))
            xtest = []  # torch.Tensor(torch.ones(len(testIndices),resultOfModel1.size(1)))

            test_label_file = pathTest + "/" + dataset_choice + "/" + disease + dataset + "" + str(
                rep) + "" + str(j) + "testLabel.txt"
            ytest = []
            fR = open(test_label_file, 'r')

            index = 0
            testindicesused = []

            for line in fR:
                i = testIndices[index]
                i = str(i)
                if i in node_embed_dict:  # to exclude nodes which do not have any neighbors in any networks
                    xtest.append(node_embed_dict[i])
                    ytest.append(int(line.strip()[0]))
                    testindicesused.append(int(i))
                index += 1
            xtest = np.array(xtest)
            ytest = np.int_(np.array(ytest))
            testindicesused = np.array(testindicesused)
            log_reg = LogisticRegression()
            log_reg.fit(xtrain, ytrain)
            predictions = log_reg.predict_proba(xtest)[:, 1]

            AUC, AUCPR, areaPR, b_acc = survey_methods_eval(ytest, predictions)
            sumAUC += AUC
            sumAUCPR += AUCPR

            indices = np.argsort(-predictions)  # minus decreasing
            predictions_init = np.sort(predictions)[::-1]  # [::-1] converts to reverse order (decreasing)
            ytest_init = ytest[indices]
            actual = [i for i in range(len(ytest_init)) if ytest_init[i] == 1]
            sum_ndcg += ndcg_at_k(ytest_init, top_k)
            sum_pk += pk(actual, top_k)
            sum_rk += rk(actual, top_k)
            sum_apk += apk(actual, top_k)
            sum_mapk += mapk(actual, len(ytest))
            sum_b_acc += b_acc
            sum_areaPR += areaPR

        average = sumAUC / n_splits
        averageAUPRC = sumAUCPR / n_splits
        sumaverages += average
        sumaveragesAUCPR += averageAUPRC

        average_ndcg = sum_ndcg / n_splits
        sumaverages_ndcg += average_ndcg
        average_pk = sum_pk / n_splits
        sumaverages_pk += average_pk
        average_rk = sum_rk / n_splits
        sumaverages_rk += average_rk
        average_apk = sum_apk / n_splits
        sumaverages_apk += average_apk
        average_mapk = sum_mapk / n_splits
        sumaverages_mapk += average_mapk

        average_areaPR = sum_areaPR / n_splits
        sumaverages_areaPR += average_areaPR
        average_b_acc = sum_b_acc / n_splits
        sumaverages_b_acc += average_b_acc

    AUCresult = sumaverages / repeat
    AUCPRresult = sumaveragesAUCPR / repeat
    sumaverages_ndcg_result = sumaverages_ndcg / repeat
    sumaverages_pk_result = sumaverages_pk / repeat
    sumaverages_rk_result = sumaverages_rk / repeat
    sumaverages_apk_result = sumaverages_apk / repeat
    sumaverages_mapk_result = sumaverages_mapk / repeat

    sumaverages_areaPR_result = sumaverages_areaPR / repeat
    sumaverages_b_acc_result = sumaverages_b_acc / repeat

    ALL_RESULT = method_choice[met] + " " + operation_choice[choice] + " " + disease + ' ' + str(
        sumaverages_pk_result) + ' ' + str(sumaverages_rk_result) + ' ' + str(sumaverages_apk_result) + ' ' + str(
        sumaverages_mapk_result) + ' ' + str(sumaverages_ndcg_result) + ' ' + str(sumaverages_b_acc_result) + ' ' + str(
        AUCresult) + ' ' + str(sumaverages_areaPR_result) + ' ' + str(AUCPRresult) + ' ' + str(
        np.std(sumAUCList, ddof=1) / np.sqrt(prev_splits * repeat)) + ' ' + str(
        np.std(sumAUCPRList, ddof=1) / np.sqrt(prev_splits * repeat)) + ' ' + str(
        np.std(sumAUCList, ddof=1)) + ' ' + str(np.std(sumAUCPRList, ddof=1)) + '\n'
    print(ALL_RESULT)
    new_file.write(ALL_RESULT)

    sum_disease_AUPRC += AUCPRresult
    sum_disease_AUC += AUCresult
    sum_disease_ndcg += sumaverages_ndcg_result

    sum_disease_pk += sumaverages_pk_result
    sum_disease_rk += sumaverages_rk_result
    sum_disease_apk += sumaverages_apk_result
    sum_disease_mapk += sumaverages_mapk_result

    sum_disease_areaPR += sumaverages_areaPR_result
    sum_disease_b_acc += sumaverages_b_acc_result


ALL_RESULT = method_choice[met] + " " + operation_choice[choice] + " " + 'average_diseases' + ' ' + str(
    sum_disease_pk / len(diseases)) + ' ' + str(sum_disease_rk / len(diseases)) + ' ' + str(
    sum_disease_apk / len(diseases)) + ' ' + str(sum_disease_mapk / len(diseases)) + ' ' + str(
    sum_disease_ndcg / len(diseases)) + ' ' + str(sum_disease_b_acc / len(diseases)) + ' ' + str(
    sum_disease_AUC / len(diseases)) + ' ' + str(sum_disease_areaPR / len(diseases)) + ' ' + str(
    sum_disease_AUPRC / len(diseases)) + ' ' + str(np.std(sum_disease_AUCList, ddof=1) / np.sqrt(175)) + ' ' + str(
    np.std(sum_disease_AUPRCList, ddof=1) / np.sqrt(175)) + ' ' + str(np.std(sum_disease_AUCList, ddof=1)) + ' ' + str(
    np.std(sum_disease_AUPRCList, ddof=1)) + '\n'
print(ALL_RESULT)
new_file.write(ALL_RESULT)

new_file.close()