
import os

directory = '/efs/hang/telecombrain/globecom23/Paragraphs/NEW/SA2'
 
# iterate over files in
# that directory
num_sentences = 0
count = 0

for filename in os.listdir(directory):

    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        with open(f, 'r') as f:
            for line in f:
                num_sentences += line.count(".")


print(num_sentences)


        




