import datasets

ds=datasets.load_dataset("Skylion007/openwebtext")
#Split is done because it is not done in dataset already
train_test=ds['train'].train_test_split(test_size=0.001,seed=42)
train_test_val=train_test['train'].train_test_split(test_size=0.0005,seed=42)
new_ds = datasets.DatasetDict({
    'train': train_test_val['train'],
    'test': train_test['test'],
    'valid': train_test_val['test']})
new_ds.save_to_disk("openwebtext")