import os

def batch_rename():
    count = 1
    for filename in os.listdir("."):
        os.rename(filename, str(count)+"."+str(filename.split(".")[1]))
        count+=1

