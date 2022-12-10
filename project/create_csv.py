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


def write_data_to_csv(train_name="Train_test_.csv"):
    """
    Reads speech TESS & RAVDESS datasets from directory and write it to a metadata CSV file.
    params:
        emotions (list): list of emotions to read from the folder, default is ['sad', 'neutral', 'happy']
        train_name (str): the output csv filename for training data, default is 'Train_tess_ravdess.csv'
        test_name (str): the output csv filename for testing data, default is 'Test_tess_ravdess.csv'
        verbose (int/bool): verbositiy level, 0 for silence, 1 for info, default is 1


        Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    """
    train_target = {"path": [], "emotion": [], "label": [], "gender": []}
    test_target = {"path": [], "emotion": [], "label": []}
    counter = 0

    data_path = '../project/data/RAVDESS/Actor_*/*.wav'
    data_path1 = '../project/data/Actor_'

    for file in glob.glob(data_path):

        file_path = os.path.basename(file)
        train_target["path"].append(file)

        # get emotion label from the sample's file
        emotion = int(file_path.split("-")[2])

        #  move surprise to 0 for cleaner behaviour with PyTorch/0-indexing
        if emotion == 8:
            emotion = 0  # surprise is now at 0 index; other emotion indeces unchanged

        train_target["emotion"].append(emotions_dict.get(emotion))
        train_target["label"].append(emotion)

        # get other labels we might want
        # even actors are female, odd are male
        gender = ""
        if (int((file_path.split("-")[6]).split(".")[0])) % 2 == 0:
            gender = 'female'

        else:
            gender = 'male'
        train_target["gender"].append(gender)

        # data_path = '../project/data/SAVEE/DC_*/*.wav'

    # SAVEE dataset

    data_path = '../project/data/SAVEE/*'
    counter = 0
    emotion = []

    for file in glob.glob(data_path):

        file_path = os.path.basename(file)
        train_target["path"].append(file)

        if file[-8:-6] == '_a':
            # emotion.append('male_angry')
            train_target["emotion"].append("angry")
            train_target["label"].append(5)

        elif file[-8:-6] == '_d':
            # emotion.append('male_disgust')
            train_target["emotion"].append("disgust")
            train_target["label"].append(7)

        elif file[-8:-6] == '_f':
            # emotion.append('male_fear')
            train_target["emotion"].append("disgust")
            train_target["label"].append(7)

        elif file[-8:-6] == '_h':
            # emotion.append('male_happy')
            train_target["emotion"].append("happy")
            train_target["label"].append(3)

        elif file[-8:-6] == '_n':
            # emotion.append('male_neutral')
            train_target["emotion"].append("neutral")
            train_target["label"].append(1)

        elif file[-8:-6] == 'sa':
            # emotion.append('male_sad')
            train_target["emotion"].append("sad")
            train_target["label"].append(4)

        elif file[-8:-6] == 'su':
            # emotion.append('male_surprise')
            train_target["emotion"].append("surprised")
            train_target["label"].append(0)

        train_target["gender"].append("male")

    pd.DataFrame(train_target).to_csv(train_name)


if __name__ == '__main__':
    write_data_to_csv()
