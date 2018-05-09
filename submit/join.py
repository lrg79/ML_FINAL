from os import listdir
from os.path import isfile, join

total_files = [f for f in listdir("Datasets/test/images-val-pub") if isfile(join("Datasets/test/images-val-pub/", f))]

classified = {}




file = open("output.csv", "r")
file.readline()
for l in file.readlines():
    classified[l.split(",")[0].split(".")[0]] = l.split(",")[1]

file.close()


file = open("newoutput.csv", "w+")
file.write("image_label,celebrity_name\n")

for i in range(len(total_files)):
    try:
        name = classified[total_files[i].split(".")[0]]
        file.write("%s,%s\n" %(total_files[i], name.strip()))
    except:
        print("no label")
        file.write("%s,%s" % (total_files[i], "killian_weinberger\n"))


file.close()

print(len(total_files))
print(len(classified))