# This is used to tokenize dataset and save 

from transformers import AutoTokenizer
import datasets


tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
# can be uncommented if data is loaded once and saved at some location
ds=datasets.load_from_disk("openwebtext")
# ds=datasets.load_dataset("Skylion007/openwebtext")


print(ds)

input_texts= ds['test']['text']

encodings = tokenizer("\n\n".join(input_texts), return_tensors="pt")
print(encodings.input_ids.shape)

with open("./cpp/encodings.txt","w") as file:
    file.write(str({"input_ids":encodings.input_ids.tolist()}))

