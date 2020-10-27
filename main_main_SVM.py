import csv
import ft
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#ex) anger이 엑셀 기준 1~127행 -> 리스트 [0:127]
# anger_non_anger : anger(1~127) , non_anger(128~535) / A열은 파일명 써있음 (공통)
# boredom_non_boredom : boredom(1~81), non_boredom(82~535)
# disgust_non_disgust : disgust(1~46), non_disgust(47~535)
# fear_non_fear : fear(1~69), non_fear(70~535)
# happy_non_happy : happy(1~71), non_happy(72~535)
# neutral_non_neutral : neutral(1~79), non_neutral(80~535)
# sad_non_sad : sad(1~62). non_sad(63~535)
emotion = 'sad'
print(emotion)
divide_num = 62
gamma_num = 8 # 1/8, 1/4, 1/2, 1, 2, 4, 8
print('gamma num :', gamma_num)

########## 데이터 읽기
total_data = []
f = open(emotion + '_non_' + emotion +'.csv', 'rt')
rdr = csv.reader(f)
for line in rdr :
    total_data.append(line)
f.close()
print('total data num :', len(total_data))

########## emotion data 와 non emotion data로 나누기
emotion_data = total_data[0:divide_num]
non_emotion_data = total_data[divide_num:535]


print('emotion num', len(emotion_data))
print('non_emotion num', len(non_emotion_data))

########### emotion data와 non emotion data를 x, y data로 나누기
emotion_x_data = []
emotion_y_data = []
non_emotion_x_data = []
non_emotion_y_data = []

for i in range(0, len(emotion_data)) :
    emotion_x_data.append(emotion_data[i][1:])
    emotion_y_data.append(int(emotion_data[i][0]))

for j in range(0, len(non_emotion_data)) :
    non_emotion_x_data.append(non_emotion_data[j][1:])
    non_emotion_y_data.append(int(non_emotion_data[j][0]))

########## emotion data의 x, y data와 non_emotion data의 x, y data를 train / test data로 나누기 (70% : 30%)
# emotion data tr/te 나누기
tr_emotion_X, te_emotion_X, tr_emotion_Y, te_emotion_Y = train_test_split(emotion_x_data, emotion_y_data, test_size = 0.3, random_state = 777)

# non emotion data tr/te 나누기
tr_non_emotion_X, te_non_emotion_X, tr_non_emotion_Y, te_non_emotion_Y = train_test_split(non_emotion_x_data, non_emotion_y_data, test_size = 0.3, random_state = 777)

########## 실험에 사용할 train data와 test data 만들기 : train_data = tr_emotion + tr_non_emotion / test_data = te_emotion + te_non_emotion
Real_train_X_data = tr_emotion_X + tr_non_emotion_X
Real_train_Y = tr_emotion_Y + tr_non_emotion_Y
Real_Test_X_data = te_emotion_X + te_non_emotion_X
Real_Test_Y = te_emotion_Y + te_non_emotion_Y
print('train_emotion 데이터 개수 :', len(tr_emotion_X))
print('train_non_emotion 데이터 개수 :', len(tr_non_emotion_X))
print('test_emotion 데이터 개수 :', len(te_emotion_X))
print('test_non_emotion 데이터 개수 :', len(te_non_emotion_X))

########## train data와 test data를 minmaxscaler 하기
scaler = MinMaxScaler()
X_train_scale = ft.ArrayToList(scaler.fit_transform(Real_train_X_data))
X_test_scale = ft.ArrayToList(scaler.fit_transform(Real_Test_X_data))

########## 감마값 정하기 : KLD 이용(svm output을 이용해 결정하기)
# 최종적으로 우리가 사용해야하는 데이터 : X_train_scale, X_test_scale, Real_train_Y, Real_Test_Y
# 1) SVM 실행
svc = svm.SVC(kernel='rbf', gamma=gamma_num)
svc.fit(X_train_scale, Real_train_Y)
print("SVM Train Accuracy : %.5f" % (svc.score(X_train_scale, Real_train_Y)))
print("SVM Test accuracy : %.5f" % (svc.score(X_test_scale, Real_Test_Y)))

# 2) y_hat값 저장하기
tr_y_hat = svc.decision_function(X_train_scale)
te_y_hat = svc.decision_function(X_test_scale)

########## 감마값 정하기 : KLD 이용(svm output을 이용해 결정하기)
##################################################################### KLD용 데이터 만들기
# # 2) KLD 사용할 data 만들기
# # tr_y_hat 데이터에서 emotion data / non_emotion data 분리하기
# KLD_emotion_data = []
# KLD_non_emotion_data = []
#
# for k in range(0, len(Real_train_Y)) :
#     if Real_train_Y[k] == 1 :
#         KLD_emotion_data.append(tr_y_hat[k])
#
#     else :
#         KLD_non_emotion_data.append(tr_y_hat[k])

# 3) KLD 실행
#### 어떻게 해야할지 모르겠음
#### 클래스 별 데이터 갯수가 다른데 어떻게 계산해야 하는지 모르겠음
# KLD_result = ft_2.KLD()
# print(KLD_result)
#
# print(KLD_non_emotion_data)
# print(KLD_emotion_data)

#그렇다면, 그냥 svm 성능으로 결정할까?####################################################
############################################################################################################

########## CPON
# 1) train_y_hat과 test_y_hat 더하고 통채로 normalization 하기
total_y_hat = ft.ArrayToList(tr_y_hat) + ft.ArrayToList(te_y_hat)
norm_y_hat = ft.ArrayToList(ft.MinMaxScaler(total_y_hat))

# 2) norm_y_hat 값을 train emotion / train non_emotion / test emotion / test non_emotion으로 나누기
tr_emotion_norm_y_hat = norm_y_hat[0:len(tr_emotion_X)]
tr_non_emotion_norm_y_hat = norm_y_hat[len(tr_emotion_X) : len(tr_emotion_X)+len(tr_non_emotion_X)]
te_emotion_norm_y_hat = norm_y_hat[len(tr_emotion_X)+len(tr_non_emotion_X) : len(tr_emotion_X) + len(tr_non_emotion_X) + len(te_emotion_X)]
te_non_emotion_norm_y_hat = norm_y_hat[len(tr_emotion_X) + len(tr_non_emotion_X) + len(te_emotion_X) : ]


# emotion class = 1 / non_emotion class = -1
Tr_pred_non_emotion, Tr_pred_emotion = ft.GetCDF(tr_non_emotion_norm_y_hat, tr_emotion_norm_y_hat)
Te_pred_non_emotion, Te_pred_emotion = ft.GetCDF(te_non_emotion_norm_y_hat, te_emotion_norm_y_hat)


# 최종 성능
tr_count_emotion = 0 # 1이어야하는 것을 1이라고 맞게 예측한 데이터 갯수
tr_count_non_emotion = 0 # -1이어야 하는 것을 -1이라고 맞게 예측한 데이터 개수

for tr_count_non_emotion_num in range(0, len(Tr_pred_non_emotion)) :
    if Tr_pred_non_emotion[tr_count_non_emotion_num] == -1 :
        tr_count_non_emotion = tr_count_non_emotion + 1

for tr_count_emotion_num in range(0, len(Tr_pred_emotion)) :
    if Tr_pred_emotion[tr_count_emotion_num] == 1 :
        tr_count_emotion = tr_count_emotion + 1

te_count_emotion = 0
te_count_non_emotion = 0
for te_count_non_emotion_num in range(0, len(Te_pred_non_emotion)) :
    if Te_pred_non_emotion[te_count_non_emotion_num] == -1 :
        te_count_non_emotion = te_count_non_emotion + 1

for te_count_emotion_num in range(0, len(Te_pred_emotion)) :
    if Te_pred_emotion[te_count_emotion_num] == 1 :
        te_count_emotion = te_count_emotion + 1



CPON_tr_result = (tr_count_emotion + tr_count_non_emotion) / (len(Tr_pred_emotion) + len(Tr_pred_non_emotion))
CPON_te_result = (te_count_emotion + te_count_non_emotion) / (len(Te_pred_emotion) + len(Te_pred_non_emotion))

print('CPON tr_ressult : ', CPON_tr_result)
print('CPON_te_result :', CPON_te_result)