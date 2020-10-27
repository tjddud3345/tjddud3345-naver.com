import numpy as np
import csv
from scipy.stats import beta
import matplotlib.pyplot as plt

# 0~1 로 normalization 하는 함수
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)


# 그냥 예시를 들기위한 y_hat 값을 불러오는 함수입니다.
def read_yhat() :
    f = open('anger_te_y_hat.txt', 'r')
    y_hat = []
    csvReader = csv.reader(f, delimiter='\n')
    for i in csvReader:
        temp = []
        temp.append(i)
        y_hat.append(float(temp[0][0]))
    f.close()
    return y_hat

# array 를 list로 바꿔주는 함수
def ArrayToList(array) :
    list = []
    for i in range(array.shape[0]) :
        list.append(array[i])
    return list

# 1차원 리스트 평균
def Mean_1d(list) :
    temp = 0
    for i in range(len(list)) :
        temp += list[i]
    average = temp / len(list)
    return average

# 1차원 리스트 분산
def Variance_1d(list, mean) :
    temp = 0
    for i in range(len(list)) :
        temp += np.square(list[i]-mean)
    return temp/len(list)


#  A, B 값
def AandB(mean, var) :
    a = mean * ( ((mean*(1-mean))/var) - 1)
    b = (1-mean) * ( ((mean*(1-mean))/var) - 1)
    return a, b

# CDF값을 그리기 위한 함수
def PyNy(y_hat, PD_LEN) :
    PD_y_hat = y_hat[:PD_LEN]
    PD_y_hat_mean = Mean_1d(PD_y_hat)
    PD_y_hat_var = Variance_1d(PD_y_hat, PD_y_hat_mean)
    PD_a, PD_b = AandB(PD_y_hat_mean, PD_y_hat_var)

    nonPD_y_hat = y_hat[PD_LEN:]
    nonPD_y_hat_mean = Mean_1d(nonPD_y_hat)
    nonPD_y_hat_var = Variance_1d(nonPD_y_hat, nonPD_y_hat_mean)
    N_a, N_b = AandB(nonPD_y_hat_mean, nonPD_y_hat_var)

    cdf_x = np.linspace(0, 1, 1001)

    P_y = beta.cdf(cdf_x, PD_a, PD_b)
    N_y = beta.cdf(cdf_x, N_a, N_b)

    # plot 을 안쓰실때는 # 으로 막아두시길
    for i in range(len(N_y)) :
        N_y[i] = 1-N_y[i]

    # 측정된 cdf 값을 그리는 plt
    plt.plot(cdf_x, P_y, 'red'); #main emotion
    plt.plot(cdf_x, N_y, 'blue'); #non_emotion

    plt.ylim(0, 2)
    plt.show()

    # y_hat 값에 대한 histogram 을 그리는 plt
    if PD_a > 0:
        plt.hist([y_hat[:PD_LEN], y_hat[PD_LEN:]])
        plt.show()

    return P_y, N_y


# 측정된 cdf 값을 이용하여 y_hat 값을 보고 그 값이 -1 클래스인지 1 클래스인지 예측하여 반환
# 저는 PD가 1, Normal 이 -1으로 계산하였습니다.
def GetCDF(N_y_hat, PD_y_hat) :
    y_hat = PD_y_hat + N_y_hat
    P_y, N_y = PyNy(y_hat, len(PD_y_hat))

    cdf_min = 1
    for i in range(1, 999, 1):
        temp = abs(P_y[i] - N_y[i])
        if cdf_min >= temp:
            cdf_min = temp
            cdf_idx = i

    pred_N = []
    pred_PD = []

    for i in range(len(N_y_hat)):
        if N_y_hat[i] >= cdf_idx * 0.001:
            pred_N.append(1)
        else:
            pred_N.append(-1)

    for i in range(len(PD_y_hat)):
        if PD_y_hat[i] >= cdf_idx * 0.001:
            pred_PD.append(1)
        else:
            pred_PD.append(-1)

    return pred_N, pred_PD