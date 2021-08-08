import numpy as np
from csv import DictWriter
import csv
import json


# Function to convert a CSV to JSON: Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath, txtFilePath):
    data = {}
    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
        # Convert each row into a dictionary and add it to data
        for rows in csvReader:
            key = rows['Comment Id']
            data[key] = rows
    # Open a json writer, and use the json.dumps() function to dump data
    with open(txtFilePath, 'w', encoding='utf-8') as txtf:
        txtf.write(json.dumps(data, indent=4))
    # Open a json writer, and use the json.dumps() function to dump data
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))


def split_data(csvFilePath, jsonFilePath, txtFilePath):
    fields = ['Video Name', 'Channel Name', 'Comment Id', 'User Name',  'Date', 'Likes']

    with open("train.csv", 'a') as f1, open("validation.csv", 'a') as f2, open("test.csv", 'a') as f3:
        f1_dict = DictWriter(f1, fieldnames=fields)
        f2_dict = DictWriter(f2, fieldnames=fields)
        f3_dict = DictWriter(f3, fieldnames=fields)

        f1_dict.writerow(fields)
        f2_dict.writerow(fields)
        f3_dict.writerow(fields)

        for line in jsonFilePath.keys():
            file = np.random.choice([f1_dict, f2_dict, f3_dict], p=[0.7, 0.1, 0.2])
            file.writerow(line)
    # with open(txtFilePath, 'w') as source_file:
    #     source_file = json.loads(source_file.read())
    #     with open("train.csv", 'a') as f1, open("validation.csv", 'a') as f2, open("test.csv", 'a') as f3:
    #         f1_dict = DictWriter(f1, fieldnames=fields)
    #         f2_dict = DictWriter(f2, fieldnames=fields)
    #         f3_dict = DictWriter(f3, fieldnames=fields)
    #
    #         f1_dict.writerow(fields)
    #         f2_dict.writerow(fields)
    #         f3_dict.writerow(fields)
    #
    #         for line in source_file['Comment Id']:
    #             file = np.random.choice([f1_dict, f2_dict, f3_dict], p=[0.7, 0.1, 0.2])
    #             file.writerow(line)


def main():
    csvFilePath = r'data/youtube_dataset.csv'
    txtFilePath = r'data/youtube_dataset.txt'
    jsonFilePath = r'data/youtube_dataset.json'

    make_json(csvFilePath, jsonFilePath, txtFilePath)
    split_data(csvFilePath, jsonFilePath, txtFilePath)


if __name__ == "__main__":
    main()


# with open("data/youtube_dataset.csv", encoding='utf8') as source_file:
#     with open("train.csv", 'a') as f1, open("validation.csv", 'a') as f2, open("test.csv", 'a') as f3:
#         f1_dict = DictWriter(f1, fieldnames=fields)
#         f2_dict = DictWriter(f2, fieldnames=fields)
#         f3_dict = DictWriter(f3, fieldnames=fields)
#
#         for i, line in enumerate(source_file):
#             if i == 0:
#                 f1_dict.writerow(line)
#                 f2_dict.writerow(line)
#                 f3_dict.writerow(line)
#                 continue
#             file = np.random.choice([f1_dict, f2_dict, f3_dict], p=[0.7, 0.1, 0.2])
#             file.writerow(line)
