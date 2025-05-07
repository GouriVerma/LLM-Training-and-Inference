# This code trains the Llama model and save the model in folder trained-llama, specified in output directory of TrainingArguments


import datasets
from transformers import Trainer,TrainingArguments,AutoModelForCausalLM,AutoTokenizer,DataCollatorForLanguageModeling
from tqdm import tqdm
import torch
import os
import wandb #Library to track training and validation metrics with huggingface Trainer API which is used to train model


#Setting up the environment variables
# ----------------------------
# Environment Setup
# ----------------------------
os.environ["WANDB_PROJECT"]="openwebtext_analysis"
os.environ["WANDB_LOG_MODEL"] = "false"

# GPU Check
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("Fallback to CPU")


# ----------------------------
# Dataset Preparation
# ----------------------------

# Load Dataset from Hub for First Time and Saving it to disk, to be commented after first run
# ds=datasets.load_dataset("Skylion007/openwebtext")
# #Split is done because it is not done in dataset already
# train_test=ds['train'].train_test_split(test_size=0.001,seed=42)
# train_test_val=train_test['train'].train_test_split(test_size=0.0005,seed=42)
# new_ds = datasets.DatasetDict({
#     'train': train_test_val['train'],
#     'test': train_test['test'],
#     'valid': train_test_val['test']})
# new_ds.save_to_disk("openwebtext")

#Load Dataset from local after 1st run as we are saving the dataset in the above line
ds = datasets.load_from_disk("openwebtext")
print(ds)


# For now, for running the code in server
# ds = datasets.load_from_disk("/mnt/combined/home/parveen/gouri/openwebtext")
# mini_ds = datasets.DatasetDict({
#     'train': ds['train'].select(range(1000000,2000000)),
#     'test': ds['test'],
#     'val': ds['val']})
# print(mini_ds)
# print(mini_ds['train'][0])

###########################################################################################################################################################




# ----------------------------
# Model Initialization
# ----------------------------

model_id = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)




max_length=1024

# ----------------------------
# Tokenization
# ----------------------------

# Take the whole dataset dictionary, join it, tokenize it and split it into samples of 1024 length each 
def preprocess_function(examples):
    txt="\n".join(examples["text"])
    if len(txt)<max_length:
        print(f"len {len(txt)}")
    inputs=tokenizer(txt,max_length=max_length,truncation=True,return_overflowing_tokens=True,return_length=True)
    input_batch = []
    for length, input_ids in zip(inputs["length"], inputs["input_ids"]):
        if length == max_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}



# Get tokenized dataset
tokenized_ds = ds.map(
    preprocess_function,
    batched=True,
    remove_columns=ds["train"].column_names,
)



print(tokenized_ds)
tokenized_ds.save_to_disk("tokenized_ds") #Save to disk to load easily because tokenization each time takes time


# Directly load after one run, as saved in previous commands- it takes time to tokenize, hence saved after one run
# tokenized_ds=datasets.load_from_disk("tokenized_ds")
# print(tokenized_ds)




# Function to computer perplexity which is a metric in text generation => finds how much model's predicted words differ from actual, how much model is producing different next words than expected one
def compute_perp(input_texts):
    encodings = tokenizer("\n\n".join(input_texts), return_tensors="pt")

    max_length =1024
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    print(ppl)
    print(n_tokens)


# ----------------------------
# Training Setup
# ----------------------------


# ----------------------------
# Training & Evaluation
# ----------------------------

# TrainingArguments API
args = TrainingArguments(
    output_dir="trained-llama",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=50,
    gradient_accumulation_steps=128,
    eval_accumulation_steps=100,
    num_train_epochs=1,
    # max_steps=2056,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    learning_rate=2e-5,
    # save_steps=5000,
    fp16=True,
    report_to="wandb",
    save_total_limit=3
)


# Trainer API
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["val"],
)


input_texts = ds['train']['text'][:100]
input_texts2 = ds['test']['text'][:100]
input_texts3 = ds['val']['text'][:100]
compute_perp(input_texts)
compute_perp(input_texts2)
compute_perp(input_texts3)
trainer.train()
wandb.finish()

compute_perp(input_texts)
compute_perp(input_texts2)
compute_perp(input_texts3)











