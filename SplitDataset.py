import glob, os

# Current directory
currentDir = os.path.dirname(os.path.abspath(__file__))

print(currentDir)

# change directory to folder containing entire dataset
currentDir = 'data/obj'

# Percentage of images to be used for validating model (test split)
testSplit = 10; # 10% test, 90% train

# using the open() functions write ('w') feature to create the train and test files
trainFile = open('data/train.txt', 'w')
testFile = open('data/test.txt', 'w')

# Populating train.txt and test.txt files
counter = 1
# round the test split up to nearest full percentage
indexTest = round(100 / testSplit)
# glob used to retrive filenames based on a specific pattern, (ending with .jpg)
for pathAndFilename in glob.iglob(os.path.join(currentDir, "*.jpg")):
    # get the name of every jpg file in the folder (i.e. split the path by the basename)
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    # if counter in loop equals the test split percentage
    if counter == indexTest:
        counter = 1
        # write the jpg files to test.txt
        testFile.write("data/obj" + "/" + title + '.jpg' + "\n")
    else:
        #else write the remaining percentage to train.txt
        trainFile.write("data/obj" + "/" + title + '.jpg' + "\n")
        counter = counter + 1