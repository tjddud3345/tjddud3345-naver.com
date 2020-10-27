import numpy as np
import ft
import csv

# 사용방법
# 1. SVM에서 나온 y_hat값을
temp_y_hat = ft.read_yhat()

# normalization을 시키고 다시 list type으로 바꿔줍니다.
# 참고로 train 및 test data를 한번에 normalization 해야합니다. 저는 편의상 train set 만 갖고 예시를 보여드리겠습니다.
y_hat = ft.ArrayToList(ft.MinMaxScaler(temp_y_hat))

# normalization 된 y_hat을 target function(실제 y값)이 -1인 data 와 1인 데이터로 나눠줍니다. (저는 Normal class는 -1, parkinson class 는 1로 작업했습니다)
# 제가 parkinson(PD) 관련 데이터로 작업을 했어서 변수명이 저렇습니다.
PD_LEN = 147
N_y_hat = y_hat[PD_LEN:]
PD_y_hat = y_hat[:PD_LEN]

# GetCDF 함수에 y_hat 값을 넣으면 각 데이터에 대한 예측값(-1 또는 1)을 반환합니다.
pred_N, pred_PD = ft.GetCDF(N_y_hat, PD_y_hat)



