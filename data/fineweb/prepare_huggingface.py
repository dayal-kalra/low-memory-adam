import os
from tqdm import tqdm
import numpy as np
from transformers import GPT2Tokenizer  # Import HuggingFace tokenizer
from datasets import load_from_disk

num_proc = 8
num_proc_load_dataset = num_proc

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

print("Loading dataset from disk...")
dataset = load_from_disk("/home/dayal/scratch.mzms/datasets/fineweb-edu-10B/")

# Create train/val split
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')
print(split_dataset)

def process(example):
    # Tokenize using HuggingFace tokenizer
    # ids = tokenizer.encode(example['text'], add_special_tokens=False)  # Similar to encode_ordinary
    ids = tokenizer.encode(
        example['text'], 
        add_special_tokens=False,
        truncation=True,
        max_length=1024
    )

    ids.append(tokenizer.eos_token_id)  # Add EOS token (equivalent to EOT in tiktoken)
    out = {'ids': ids, 'len': len(ids)}
    return out


# Tokenize the dataset
tokenized = split_dataset.map(process, remove_columns=['text'], desc="tokenizing the splits", num_proc=num_proc,)

# Save tokenized data to binary files
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16  # Still valid as GPT-2's vocabulary is < 2**16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

