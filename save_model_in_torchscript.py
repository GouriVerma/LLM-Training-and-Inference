import datasets
from transformers import Trainer,TrainingArguments,AutoModelForCausalLM,AutoTokenizer,DataCollatorForLanguageModeling,LlamaConfig,LlamaForCausalLM
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import wandb

ds = datasets.load_from_disk("/mnt/combined/home/parveen/gouri/openwebtext")
print(ds)

class ScriptableLlamaForCausalLM(LlamaForCausalLM):
    def forward(self, input_ids, labels, attention_mask=None, token_type_ids=None, cache_position=None):
        # Call the parent class's forward method with only supported arguments
        return super().forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,cache_position=cache_position,labels=labels )


model_id = "meta-llama/Llama-3.2-1B"
model_dir="trained-llama/checkpoint-4064"
# # base_model = AutoModelForCausalLM.from_pretrained(model_id,torchscript=True,device_map="auto")
# # base_model.tie_weights()
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# configuration = LlamaConfig(torchscript=True,tie_word_embeddings=True)
# base_model=LlamaForCausalLM(config=configuration)
# base_model = base_model.from_pretrained(model_id,config=configuration,device_map="auto")
# print(base_model)

configuration = LlamaConfig.from_pretrained(model_dir)

# Update the configuration if needed
configuration.torchscript = True
configuration.tie_word_embeddings = True

# Instantiate the model with the updated configuration and load the weights
# base_model = LlamaForCausalLM.from_pretrained(model_id, config=configuration)
base_model = AutoModelForCausalLM.from_pretrained(model_dir)
base_model.config.torchscript=True
base_model.config.tie_word_embeddings=True

# base_model=LlamaForCausalLM(config=configuration)
# base_model = base_model.from_pretrained(model_id,config=configuration)
base_model.tie_weights()
print(base_model.config)

from transformers import LlamaForCausalLM

# class CustomLlamaForCausalLM(LlamaForCausalLM):
#     def forward(self, input_ids=None, attention_mask=None, position_ids=None, 
#                 past_key_values=None, inputs_embeds=None, labels=None, 
#                 use_cache=None, output_attentions=None, output_hidden_states=None, 
#                 return_dict=None, cache_position=None, num_logits_to_keep=0):
#         # Call the original forward method without **kwargs
#         return super().forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             labels=labels,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             cache_position=cache_position,
#             num_logits_to_keep=num_logits_to_keep
#         )

# # Then use the custom class for tracing
# base_model = CustomLlamaForCausalLM.from_pretrained(model_dir)

# traced_model = torch.jit.script(model)


#base_model2=AutoModelForCausalLM.from_pretrained(model_id)


print("------------------------------------------------------------------------------------------------------------------")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# inputs="I like travelling because"
# base_model.to(device)
# tokenized_inputs=tokenizer(inputs,return_tensors="pt").to(device)
# output_ids=base_model.generate(tokenized_inputs['input_ids'])
# print(tokenizer.batch_decode(output_ids,skip_special_tokens=True))



# tokenized_ds=datasets.load_from_disk("my_dataset")
# print(tokenized_ds)



def compute_perp(input_texts,model):
    encodings = tokenizer("\n\n".join(input_texts), return_tensors="pt")

    max_length =1024
    print(max_length)
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        # print(f"begin_loc: {begin_loc}, type: {type(begin_loc)}")
        # print(f"end_loc: {end_loc}, type: {type(end_loc)}")

        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            loss,logits,past_key_values = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood =loss

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


input_texts = ds['val']['text'][0]
tokenized_text=tokenizer(input_texts,return_tensors="pt")
# print(tokenized_text)
base_model.eval()
# with torch.no_grad():

#     output_ids1=base_model(input_ids=tokenized_text['input_ids'],labels=tokenized_text['input_ids'])
#     # #output_ids2=base_model2(input_ids=tokenized_text['input_ids'],labels=tokenized_text['input_ids'])
#     print(output_ids1)
    # print()
    # print(output_ids2)


# compute_perp(input_texts)
# for handle in base_model._forward_hooks.values():
#     handle.remove()

# torch.save(base_model.state_dict(),"llama_new.pt")

class TorchScriptWrapper(nn.Module):
    def __init__(self, base_model):
        super(TorchScriptWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, input_ids,target_ids):
        outputs = self.base_model(input_ids=input_ids,labels=target_ids)
        # print(outputs.logits)
        # print(outputs.loss)
        loss, logits, past_key_values = outputs[:3]
        print(past_key_values)
        return loss, logits  # Return only logits for TorchScript compatibility
    

# Instantiate the wrapper
model = TorchScriptWrapper(base_model)

# model.eval()  # Switch to evaluation mode

print("yoooooooooooooooooooooooooooo")

traced_model = torch.jit.trace(model,[tokenized_text['input_ids'],tokenized_text['input_ids']])
print(traced_model)
torch.jit.save(traced_model, "traced_llama_4064_3.pt")

# traced_model = torch.jit.script(model)
# print(traced_model)
# torch.jit.save(traced_model, "traced_llama_4064_s.pt")

# inputs="I like travelling because"

# print()
# print()
# # tokenized_inputs=tokenizer(inputs,return_tensors="pt")
# # input_ids = tokenized_inputs['input_ids']
# traced_model.eval()
# with torch.no_grad():

#     loss, something, logits=traced_model(tokenized_text['input_ids'],tokenized_text['input_ids'])
#     print(loss)
#     print(logits)

# scripted_model=torch.jit.script(model)
# print(scripted_model)

# print(traced_model)

# loaded_model = torch.jit.load("traced_llama.pt")
# print(loaded_model)

# loaded_model.eval()
# with torch.no_grad():

#    # output_ids1=traced_model(input_ids=tokenized_text['input_ids'],labels=tokenized_text['input_ids'])
#     output_ids2=loaded_model(input_ids=tokenized_text['input_ids'],labels=tokenized_text['input_ids'])
#     #print(output_ids1)
#     print()
#     print(output_ids2)


# input_texts = ds['train']['text'][:10]
# input_texts2 = ds['test']['text'][:10]
# input_texts3 = ds['val']['text'][:10]
# compute_perp(input_texts,loaded_model)
# compute_perp(input_texts2,loaded_model)
# compute_perp(input_texts3,loaded_model)

