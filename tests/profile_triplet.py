import time, random
from PIL import Image
from core.models import identification as idm
from core.config import (POKEMON_CARD_DATABASE_FOLDER_PATH, IDENTIFICATION_DATA_FOLDER_PATH,
                         IDENTIFICATION_IMAGE_SIZE, POSITIVE_DATA_NUMBER)
from core.transforms import get_valid_transform, get_identification_train_transform

t0 = time.time()
train_ds, _ = idm.get_datasets()
print(f"get_datasets() build (loads negative-selector JSONs): {time.time()-t0:.2f}s, len={len(train_ds)}")

# Time full __getitem__ over random indices
N = 30
idxs = [random.randrange(len(train_ds)) for _ in range(N)]
t = time.time()
for i in idxs:
    _ = train_ds[i]
print(f"\nfull __getitem__: {(time.time()-t)/N*1000:.1f} ms/sample over {N} samples")

# Break down the components manually
anchor_tf = get_valid_transform(IDENTIFICATION_IMAGE_SIZE)
data_tf = get_identification_train_transform(IDENTIFICATION_IMAGE_SIZE)
sel = train_ds.negative_selector
al = train_ds.anchor_list

def timeit(label, fn, n=30):
    t = time.time()
    for _ in range(n):
        fn()
    print(f"  {label:28s}: {(time.time()-t)/n*1000:7.1f} ms")

print("\ncomponent breakdown:")
sample_anchor = al[0].replace(".jpg","")
timeit("open+tf anchor (full jpg)", lambda: anchor_tf(Image.open(POKEMON_CARD_DATABASE_FOLDER_PATH + al[random.randrange(len(al))])))
timeit("open+tf positive png", lambda: data_tf(Image.open(IDENTIFICATION_DATA_FOLDER_PATH + sample_anchor + "-0.png")))
timeit("negative_selector.select", lambda: sel.select_negative(al[random.randrange(len(al))].replace(".jpg","")))
