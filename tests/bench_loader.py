import time, torch
from torch.utils.data import DataLoader
from core.models import identification as idm
from core.config import IDENTIFICATION_BATCH_SIZE

def bench(name, **kw):
    train_ds, _ = idm.get_datasets()
    loader = DataLoader(train_ds, batch_size=IDENTIFICATION_BATCH_SIZE, shuffle=True, **kw)
    it = iter(loader)
    next(it)  # warmup (spawn workers)
    t = time.time(); n = 15
    for _ in range(n):
        b = next(it)
    dt = (time.time()-t)/n
    print(f"{name:42s}: {dt*1000:7.0f} ms/batch  ({IDENTIFICATION_BATCH_SIZE/dt:6.1f} samples/s)")
    del loader, it

if __name__ == "__main__":
    bench("num_workers=0", num_workers=0)
    bench("num_workers=4", num_workers=4)
    bench("num_workers=8 (current)", num_workers=8, persistent_workers=True)
    bench("num_workers=4 prefetch=4", num_workers=4, persistent_workers=True, prefetch_factor=4)
    bench("num_workers=12 pin_memory", num_workers=12, persistent_workers=True, pin_memory=True)
