import csv
from scipy import stats

emotion = 'fear'
divide_num = 69
feature_num = 11

total_data = []
f = open(emotion + '_non_' + emotion +'.csv', 'rt')
rdr = csv.reader(f)
for line in rdr :
    total_data.append(line)
f.close()

total_data_one_feature = []
for i in range(0, len(total_data)):
    total_data_one_feature.append(float(total_data[i][feature_num]))

emotion_data_one_feature = total_data_one_feature[0:divide_num]
non_emotion_data_one_feature = total_data_one_feature[divide_num:535]


print(emotion_data_one_feature)
# pvalue
result = stats.ttest_ind(emotion_data_one_feature, non_emotion_data_one_feature)
print(result)