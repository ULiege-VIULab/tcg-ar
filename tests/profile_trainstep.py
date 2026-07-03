import time, torch, torch.nn as nn, torch.optim as optim
from core.models import identification as idm
from core.config import IDENTIFICATION_LR

def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device, "| cuda:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    train_ds, valid_ds = idm.get_datasets()
    train_loader, valid_loader = idm.get_data_loaders(train_ds, valid_ds)
    model = idm.build_triplet_model(device)
    optimizer = optim.Adam(model.parameters(), lr=IDENTIFICATION_LR)
    criterion = nn.TripletMarginLoss(1.0, 2.0, reduction="mean")
    model.train()

    it = iter(train_loader)
    tdata=tfwd=tbwd=0.0; n=20
    torch.cuda.synchronize() if device=='cuda' else None
    for i in range(n+3):
        t=time.time(); a,p,ne,_ = next(it); 
        if device=='cuda': torch.cuda.synchronize()
        d_data=time.time()-t
        t=time.time(); a=a.to(device); p=p.to(device); ne=ne.to(device)
        ao=model(a); po=model(p); no=model(ne); loss=criterion(ao,po,no)
        if device=='cuda': torch.cuda.synchronize()
        d_fwd=time.time()-t
        t=time.time(); optimizer.zero_grad(); loss.backward(); optimizer.step()
        if device=='cuda': torch.cuda.synchronize()
        d_bwd=time.time()-t
        if i>=3:  # skip warmup
            tdata+=d_data; tfwd+=d_fwd; tbwd+=d_bwd
        if i<6 or i==n+2:
            print(f"  iter {i}: data={d_data*1000:6.0f}ms fwd={d_fwd*1000:6.0f}ms bwd={d_bwd*1000:6.0f}ms")
    print(f"\nAVG over {n}: data={tdata/n*1000:.0f}ms  fwd(3x resnet50)={tfwd/n*1000:.0f}ms  bwd+step={tbwd/n*1000:.0f}ms  total={(tdata+tfwd+tbwd)/n*1000:.0f}ms/it")

if __name__ == "__main__":
    run()
