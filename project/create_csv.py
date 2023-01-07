import glob
import pandas as pd
import os

# RAVDESS dataset emotions
# shift emotions left to be 0 indexed for PyTorch
emotions_dict = {
    0: 'surprised',
    1: 'neutral',
    2: 'calm',
    3: 'happy',
    4: 'sad',
    5: 'angry',
    6: 'fearful',
    7: 'disgust'
}
emotions_dict_3 = {
    0: 'positive',
    1: 'neutral',
    2: 'negative',
}

"""
SAVEE:
'a' = 'anger'
'd' = 'disgust'
'f' = 'fear'
'h' = 'happiness'
'n' = 'neutral'
'sa' = 'sadness'
'su' = 'surprise'
"""


#
# def write_data_to_csv(train_name="Train_test_.csv"):
#     """
#     Reads speech TESS & RAVDESS datasets from directory and write it to a metadata CSV file.
#     params:
#         emotions (list): list of emotions to read from the folder, default is ['sad', 'neutral', 'happy']
#         train_name (str): the output csv filename for training data, default is 'Train_tess_ravdess.csv'
#         test_name (str): the output csv filename for testing data, default is 'Test_tess_ravdess.csv'
#         verbose (int/bool): verbositiy level, 0 for silence, 1 for info, default is 1
#
#
#         Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
#     """
#     train_target = {"path": [], "emotion": [], "label": [], "gender": []}
#     test_target = {"path": [], "emotion": [], "label": []}
#     counter = 0
#     count_negative = 0
#     count_positive = 0
#     count_neutral = 0
#
#     data_path = '../project/data/RAVDESS/Actor_*/*.wav'
#
#     for file in glob.glob(data_path):
#
#         file_path = os.path.basename(file)
#         train_target["path"].append(file)
#
#         # get emotion label from the sample's file
#         emotion = int(file_path.split("-")[2])
#
#         #  move surprise to 0 for cleaner behaviour with PyTorch/0-indexing
#         if emotion == 8:
#             emotion = 0  # surprise is now at 0 index; other emotion indeces unchanged
#
#         if emotions_dict.get(emotion) == 'neutral':
#             train_target["emotion"].append('neutral')
#             train_target["label"].append(1)
#             count_neutral += 1
#
#         elif emotions_dict.get(emotion) == 'happy' or emotions_dict.get(emotion) == 'surprised' or emotions_dict.get(
#                 emotion) == 'calm':
#             train_target["emotion"].append('positive')
#             train_target["label"].append(0)
#             count_positive += 1
#         elif emotions_dict.get(emotion) == 'angry' or emotions_dict.get(emotion) == 'sad' \
#                 or emotions_dict.get(emotion) == 'disgust' or emotions_dict.get(emotion) == 'fearful':
#             train_target["emotion"].append('negative')
#             train_target["label"].append(2)
#             count_negative += 1
#
#         # train_target["emotion"].append(emotions_dict.get(emotion))
#         # train_target["label"].append(emotion)
#
#         # get other labels we might want
#         # even actors are female, odd are male
#         gender = ""
#         if (int((file_path.split("-")[6]).split(".")[0])) % 2 == 0:
#             gender = 'female'
#
#         else:
#             gender = 'male'
#         train_target["gender"].append(gender)
#
#     # print(f' RAVDESS:\n negative {count_negative} , postitve {count_positive} , neutral {count_neutral}')
#
#     total = count_negative + count_positive + count_neutral
#     print(total)
#     # data_path = '../project/data/SAVEE/DC_*/*.wav'
#     count_negative = 0
#     count_positive = 0
#     count_neutral = 0
#     # SAVEE dataset
#
#     sad = 0
#     dis = 0
#     fear = 0
#     angry = 0
#
#     data_path = '../project/data/SAVEE/*'
#     counter = 0
#     emotion = []
#
#     for file in glob.glob(data_path):
#
#         file_path = os.path.basename(file)
#
#         if file[-8:-6] == '_a' and angry < 30:
#             # emotion.append('male_angry')
#             # train_target["emotion"].append("angry")
#             # train_target["label"].append(5)
#             train_target["emotion"].append('negative')
#             train_target["label"].append(2)
#             count_negative += 1
#             angry += 1
#             train_target["gender"].append("male")
#             train_target["path"].append(file)
#
#         elif file[-8:-6] == '_d' and dis < 30:
#             # emotion.append('male_disgust')
#             # train_target["emotion"].append("disgust")
#             # train_target["label"].append(7)
#             train_target["emotion"].append('negative')
#             train_target["label"].append(2)
#             count_negative += 1
#             dis += 1
#             train_target["gender"].append("male")
#             train_target["path"].append(file)
#
#
#         elif file[-8:-6] == '_f' and fear < 30:
#             # emotion.append('male_fear')
#             # train_target["emotion"].append("fearful")
#             # train_target["label"].append(6)
#             train_target["emotion"].append('negative')
#             train_target["label"].append(2)
#             count_negative += 1
#             fear += 1
#             train_target["gender"].append("male")
#             train_target["path"].append(file)
#
#         elif file[-8:-6] == '_h':
#             # emotion.append('male_happy')
#             # train_target["emotion"].append("happy")
#             # train_target["label"].append(3)
#             train_target["emotion"].append('positive')
#             train_target["label"].append(0)
#             count_positive += 1
#             train_target["gender"].append("male")
#             train_target["path"].append(file)
#
#         elif file[-8:-6] == '_n':
#             # emotion.append('male_neutral')
#             # train_target["emotion"].append("neutral")
#             # train_target["label"].append(1)
#             train_target["emotion"].append('neutral')
#             train_target["label"].append(1)
#             count_neutral += 1
#             train_target["gender"].append("male")
#             train_target["path"].append(file)
#
#
#         elif file[-8:-6] == 'sa' and sad < 30:
#             # emotion.append('male_sad')
#             # train_target["emotion"].append("sad")
#             # train_target["label"].append(4)
#             train_target["emotion"].append('negative')
#             train_target["label"].append(2)
#             count_negative += 1
#             sad += 1
#             train_target["gender"].append("male")
#             train_target["path"].append(file)
#
#
#         elif file[-8:-6] == 'su':
#             # emotion.append('male_surprise')
#             # train_target["emotion"].append("surprised")
#             # train_target["label"].append(0)
#             train_target["emotion"].append('positive')
#             train_target["label"].append(0)
#             count_positive += 1
#             train_target["gender"].append("male")
#             train_target["path"].append(file)
#         else:
#             continue
#
#     total += count_negative + count_positive + count_neutral
#     print("savv ", count_negative + count_positive + count_neutral)
#     print(f' SAVEE:\n negative {count_negative} , postitve {count_positive} , neutral {count_neutral}')
#     # print(f' SAVEE:\n angry {angry} , disgusts {dis} , sad {sad}, fear {fear}')
#
#     count_negative = 0
#     count_positive = 0
#     count_neutral = 0
#
#     sad_m, fear_m, dis_m, angry_m, = 0, 0, 0, 0
#     sad_w, fear_w, dis_w, angry_w, = 0, 0, 0, 0
#     #
#     # CREMA-D dataset
#     data_path = '../project/data/CREMA-D/*'
#     female = [1002, 1003, 1004, 1006, 1007, 1008, 1009, 1010, 1012, 1013, 1018, 1020, 1021, 1024, 1025, 1028, 1029,
#               1030, 1037, 1043, 1046, 1047, 1049,
#               1052, 1053, 1054, 1055, 1056, 1058, 1060, 1061, 1063, 1072, 1073, 1074, 1075, 1076, 1078, 1079, 1082,
#               1084, 1089, 1091]
#
#     for i in glob.glob(data_path):
#         # print(i)
#
#         file_path = os.path.basename(i)
#         # print(file_path)
#
#         part = file_path.split('_')
#         if int(part[0]) in female:
#             temp = 'female'
#
#         else:
#             temp = 'male'
#
#         # gender.append(temp)
#         if part[2] == 'SAD' and temp == 'male' and sad_m < 134:
#             # train_target["emotion"].append("sad")
#             # train_target["label"].append(4)
#             train_target["emotion"].append('negative')
#             train_target["label"].append(2)
#             count_negative += 1
#             sad_m += 1
#             train_target["path"].append(i)
#
#             train_target["gender"].append("male")
#         elif part[2] == 'ANG' and temp == 'male' and angry_m < 134:
#             # emotion.append('male_angry')
#             # train_target["emotion"].append("angry")
#             # train_target["label"].append(5)
#             train_target["emotion"].append('negative')
#             train_target["label"].append(2)
#             train_target["gender"].append("male")
#             count_negative += 1
#             angry_m += 1
#             train_target["path"].append(i)
#
#         elif part[2] == 'DIS' and temp == 'male' and dis_m < 134:
#             # train_target["emotion"].append("disgust")
#             # train_target["label"].append(7)
#             train_target["emotion"].append('negative')
#             train_target["label"].append(2)
#             train_target["gender"].append("male")
#             count_negative += 1
#             dis_m += 1
#             train_target["path"].append(i)
#
#         elif part[2] == 'FEA' and temp == 'male' and fear_m < 134:
#             # train_target["emotion"].append("fearful")
#             # train_target["label"].append(6)
#             train_target["emotion"].append('negative')
#             train_target["label"].append(2)
#             train_target["gender"].append("male")
#             count_negative += 1
#             fear_m += 1
#             train_target["path"].append(i)
#
#         elif part[2] == 'HAP' and temp == 'male':
#             # train_target["emotion"].append("happy")
#             # train_target["label"].append(3)
#             train_target["gender"].append("male")
#             train_target["emotion"].append('positive')
#             train_target["label"].append(0)
#             count_positive += 1
#             train_target["path"].append(i)
#
#         elif part[2] == 'NEU' and temp == 'male':
#             # train_target["emotion"].append("neutral")
#             # train_target["label"].append(1)
#             train_target["gender"].append("male")
#             train_target["emotion"].append('neutral')
#             train_target["label"].append(1)
#             count_neutral += 1
#             train_target["path"].append(i)
#
#         elif part[2] == 'SAD' and temp == 'female' and sad_w < 120:
#             # train_target["emotion"].append("sad")
#             # train_target["label"].append(4)
#             train_target["emotion"].append('negative')
#             train_target["label"].append(2)
#             train_target["gender"].append("female")
#             count_negative += 1
#             sad_w += 1
#             train_target["path"].append(i)
#
#         elif part[2] == 'ANG' and temp == 'female' and angry_w < 120:
#             # train_target["emotion"].append("angry")
#             # train_target["label"].append(5)
#             train_target["emotion"].append('negative')
#             train_target["label"].append(2)
#             train_target["gender"].append("female")
#             count_negative += 1
#             angry_w += 1
#             train_target["path"].append(i)
#
#         elif part[2] == 'DIS' and temp == 'female' and dis_w < 120:
#             # train_target["emotion"].append("disgust")
#             # train_target["label"].append(7)
#             train_target["emotion"].append('negative')
#             train_target["label"].append(2)
#             train_target["gender"].append("female")
#             count_negative += 1
#             dis_w += 1
#             train_target["path"].append(i)
#
#         elif part[2] == 'FEA' and temp == 'female' and fear_w < 120:
#             # train_target["emotion"].append("fearful")
#             # train_target["label"].append(6)
#             train_target["emotion"].append('negative')
#             train_target["label"].append(2)
#             train_target["gender"].append("female")
#             count_negative += 1
#             fear_w += 1
#             train_target["path"].append(i)
#
#
#         elif part[2] == 'HAP' and temp == 'female':
#             # train_target["emotion"].append("happy")
#             # train_target["label"].append(3)
#             train_target["emotion"].append('positive')
#             train_target["label"].append(0)
#             train_target["gender"].append("female")
#             count_positive += 1
#             train_target["path"].append(i)
#
#         elif part[2] == 'NEU' and temp == 'female':
#             # train_target["emotion"].append("neutral")
#             # train_target["label"].append(1)
#             train_target["gender"].append("female")
#             train_target["emotion"].append('neutral')
#             train_target["label"].append(1)
#             count_neutral += 1
#             train_target["path"].append(i)
#
#     total_cma = count_negative + count_positive + count_neutral
#     total += count_negative + count_positive + count_neutral
#     print(f' CAMA_D:\n negative {count_negative} , postitve {count_positive} , neutral {count_neutral}')
#     print(total_cma)
#     print(total)
#     print(
#         f' CAMA_D:\n fear {fear_m + fear_w} , sad {sad_m + sad_w} , angry {angry_m + angry_w}, disgust {dis_m + dis_w}')
# #TESS dataset
# data_path = '../project/data/TESS/*'
# for i in glob.glob(data_path):
#     file_name = os.path.basename(i)
#
#
#
#     for f in glob.glob(data_path+"/*"):
#         file_path = os.path.basename(f)
#         # print(i+"\\"+file_path)
#
#
#         #print(file_path)
#         if file_name == 'OAF_angry' or file_name == 'YAF_angry':
#             train_target["path"].append(i + "\\" + file_path)
#             train_target["emotion"].append("angry")
#             train_target["label"].append(5)
#             train_target["gender"].append("female")
#
#         elif file_name == 'OAF_disgust' or file_name == 'YAF_disgust':
#             train_target["path"].append(i + "\\" + file_path)
#             train_target["emotion"].append("disgust")
#             train_target["label"].append(7)
#             train_target["gender"].append("female")
#
#         elif file_name == 'OAF_Fear' or file_name == 'YAF_fear':
#             train_target["path"].append(i + "\\" + file_path)
#             train_target["emotion"].append("fearful")
#             train_target["label"].append(6)
#             train_target["gender"].append("female")
#
#         elif file_name == 'OAF_happy' or file_name == 'YAF_happy':
#             train_target["path"].append(i + "\\" + file_path)
#             train_target["emotion"].append("happy")
#             train_target["label"].append(3)
#             train_target["gender"].append("female")
#
#         elif file_name == 'OAF_neutral' or file_name == 'YAF_neutral':
#             train_target["path"].append(i + "\\" + file_path)
#             train_target["emotion"].append("neutral")
#             train_target["label"].append(1)
#             train_target["gender"].append("female")
#
#         elif file_name == 'OAF_Pleasant_surprise' or file_name == 'YAF_pleasant_surprised':
#             train_target["path"].append(i + "\\" + file_path)
#             train_target["emotion"].append("surprised")
#             train_target["label"].append(0)
#             train_target["gender"].append("female")
#
#         elif file_name == 'OAF_Sad' or file_name == 'YAF_sad':
#             train_target["path"].append(i + "\\" + file_path)
#             train_target["emotion"].append("sad")
#             train_target["label"].append(4)
#             train_target["gender"].append("female")

# print(len(train_target["path"]),len(train_target["emotion"]), len(train_target["label"]), len(train_target["gender"]))
# Creating a DataFrame
# pd.DataFrame(train_target).to_csv(train_name)






#########################
def write_data_to_csv(train_name="Train_test_.csv"):
    """
    Reads speech TESS & RAVDESS datasets from directory and write it to a metadata CSV file.
    params:
@ -98,11 +43,9 @@ def write_tess_ravdess_csv(emotions=emotions_dict, train_name="Train_test_ravdes
    test_target = {"path": [], "emotion": [], "label": []}
    counter = 0



"""
    data_path = '../project/data/Actor_*/*.wav'
    data_path = '../project/data/RAVDESS/Actor_*/*.wav'
    data_path1 = '../project/data/Actor_'
    for file in glob.glob(data_path):

        file_path = os.path.basename(file)
        write_tess_ravdess_csv(emotions=emotions_dict, train_name="Train_test_ravdes")
        # get emotion label from the sample's file
        emotion = int(file_path.split("-")[2])


        #  move surprise to 0 for cleaner behaviour with PyTorch/0-indexing
        if emotion == 8:
            emotion = 0  # surprise is now at 0 index; other emotion indeces unchanged
@ -129,80 +71,58 @@ def write_tess_ravdess_csv(emotions=emotions_dict, train_name="Train_test_ravdes
            gender = 'male'
        train_target["gender"].append(gender)

    pd.DataFrame(train_target).to_csv(train_name)


#######################3

# write_data_to_csv()

import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('Train_test_.csv')
# Count the number of occurrences of each emotion
counts = df['emotion'].value_counts()

# Extract the names of the emotions
emotion_names = counts.index

# Plot the horizontal bars
plt.barh(range(len(emotion_names)), counts, color=(222 / 255, 71 / 255, 142 / 255))

# Set the tick labels to the names of the emotions
plt.yticks(range(len(emotion_names)), emotion_names)

# Show the plot
# plt.show()
fraction = f'{3374}/{5174} = {3374 / 5174:.2f}'
print(fraction)
