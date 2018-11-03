from stemming.porter2 import stem
from stop_words import get_stop_words

with open('28.txt', 'r') as input_file:
    input = input_file.read()
    input = input.replace(",","")
    input = input.replace(".","")
    input = input.lower().split()

stop_words = get_stop_words('english')

output = []
for word in input:
    word = stem(word)
    if word not in stop_words:
        output.append(word)

output = "\n".join(output)

with open('result.txt', 'w') as output_file:
    output_file.write(output)
