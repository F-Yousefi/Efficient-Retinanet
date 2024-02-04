from data.charlotte_dataset import generate_charlotte_dataloader
from data.voc2007_dataset import generate_voc_dataloader
train, val, test = generate_charlotte_dataloader(batch_size= 4, num_workers=8, train_frac=0.90, val_frac=0.05, test_frac=0.05)
print(f"len(train):{len(train)},len(val):{len(val)},len(test):{len(test)}")
train, val = generate_voc_dataloader(batch_size=4, num_workers=8, fraction=1)
print(f"len(train):{len(train)},len(val):{len(val)}")

