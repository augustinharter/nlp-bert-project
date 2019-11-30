# %%

import numpy as np
import itertools
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from keras.preprocessing.sequence import pad_sequences

import transformers
from transformers import BertTokenizer, BertForTokenClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

labels = open("labels.txt").read().split(sep="\n")
label_map = dict((labels[i],i) for i in range(len(labels)))
print("LabelMap:\n", label_map)

#%%

def flat_accuracy(preds, labels, masks):
  masks_flat = masks.flatten()
  preds = torch.argmax(preds, dim=2)
  print("predicted:", preds[0,:20])
  print("actual:", labels[0,:20])
  pred_flat = preds.flatten()
  labels_flat = labels.flatten()
  print("compared:", pred_flat[:20] == labels_flat[:20])
  print("masks:", masks_flat)
  hits = [e for (i, e) in enumerate(pred_flat == labels_flat) if masks_flat[i]] #and labels_flat[i]!=label_map["O"]
  print(len(hits))
  return float(sum(hits))/len(hits)

def load_data(filename):
  tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-german-cased', do_lower_case=False)
  keys = label_map.keys()
  stream = open(filename)
  all_sentences = []
  all_labels = []
  all_tokens = []
  all_ids = []
  sentence = "[CLS]"
  labels = [25]
  tokens = ["[CLS]"]
  lines = stream.readlines()
  for i in trange(len(lines)):
    line = lines[i]
    line =line.rstrip()
    if line == "":
      sentence = sentence + " [SEP]"
      all_sentences.append(sentence)
      sentence = "[CLS]"

      tokens.append("[SEP]")
      all_tokens.append(tokens)
      all_ids.append([tokenizer.convert_tokens_to_ids(x) for x in tokens])
      tokens = ["[CLS]"]

      labels.append(26)
      all_labels.append(labels)
      labels = [25]

    else:
      word = line.split()[0]
      label = line.split()[1]
      tokenized_word = tokenizer.tokenize(word)
      labels.append(label_map[label])
      for _ in range(len(tokenized_word)-1):
        labels.append(label_map[label])
      sentence = sentence + " "+ word
      tokens = tokens + tokenized_word

  return all_sentences, all_labels, all_tokens, all_ids

sentences, labels, tokens, ids = load_data("train.txt")
eval_sent, eval_labels, eval_tok, eval_ids = load_data("dev.txt")
print("Data Preview:\n", sentences[0:4], "\n", tokens[0:4], "\n", labels[0:4], "\n", ids[0:4])
print("Eval Data Preview:\n", eval_sent[0:4], "\n", eval_tok[0:4], "\n", eval_labels[0:4], "\n", eval_ids[0:4])
size = len(ids)

# %%

model = transformers.BertForTokenClassification.from_pretrained('bert-base-german-cased', cache_dir="bert_pretrained_config", num_labels=27).to(device)
optimizer = transformers.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
# %%

def batch_loader(ids, labels, batchsize):
  ids = pad_sequences(ids, maxlen=150, dtype="long", truncating="post", padding="post")
  labels = pad_sequences(labels, maxlen=150, dtype="int", truncating="post", padding="post")
  ids = torch.tensor(ids)
  labels = torch.tensor(labels)
  size = ids.shape[0]
  step = 0

  # Create attention masks
  attention_masks = []
  # Create a mask of 1s for each token followed by 0s for padding
  for seq in ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
  masks = torch.tensor(attention_masks)
  print(masks[0:4,:40])

  while step*batchsize<size:
    start = step*batchsize
    end = start + batchsize
    step +=1
    yield ids[start:end], labels[start:end], masks[start:end]

# %%
epochs = 8
num_batches = 100
num_eval_batches = 5
batchsize = 30

batch_gen = batch_loader(ids, labels, batchsize)
eval_batches_gen = batch_loader(eval_ids, eval_labels, batchsize)
# BERT training loop
for _ in trange(epochs, desc="Epoch"):  
  
  ## TRAINING
  
  # Set our model to training mode
  model.train()  
  # Tracking variables
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  # Train the data for one epoch
  for batch in tqdm(itertools.islice(batch_gen, num_batches), total=num_batches, desc="Batch"):
    optimizer.zero_grad()
    batch = tuple(t.to(device) for t in batch)
    ids, labels, masks = batch
    # Forward pass
    loss = model(ids, attention_mask=masks, labels=labels)[0]
    tqdm.write(str(loss.item()))  
    # Backward pass
    loss.backward()
    # Update parameters and take a step using the computed gradient
    optimizer.step()
    # Update tracking variables
    tr_loss += loss.item()
    #nb_tr_examples += b_input_ids.size(0)
    #nb_tr_steps += 1
  #print("Train loss: {}".format(tr_loss/nb_tr_steps))
       
  ## VALIDATION

  # Put model in evaluation mode
  model.eval()
  # Tracking variables 
  eval_loss, eval_accuracy = 0.0, 0.0
  nb_eval_steps, nb_eval_examples = 0, 0
  # Evaluate data for one epoch
  for batch in itertools.islice(eval_batches_gen, num_eval_batches):
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    batch = tuple(t.to(device) for t in batch)
    ids, labels, masks = batch
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits = model(ids, attention_mask=masks)[0]
    # Move logits and labels to CPU
    tmp_eval_accuracy = flat_accuracy(logits, labels, masks)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1
  tqdm.write("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
  print("lr",optimizer.param_groups[0]["lr"])
  scheduler.step()
  print("lr", optimizer.param_groups[0]["lr"])

# plot training performance
#plt.figure(figsize=(15,8))
#plt.title("Training loss")
#plt.xlabel("Batch")
#plt.ylabel("Loss")
#plt.plot(train_loss_set)
#plt.show()


# %%
