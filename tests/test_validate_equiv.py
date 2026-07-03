"""Vectorized validate_triplet accuracy == original O(N^2) double-loop."""
import torch, torch.nn as nn, numpy as np

def original_acc(A, P, criterion):
    n = A.shape[0]; correct = 0
    for i in range(n):
        loss = criterion(A[i:i+1].repeat(1,1), P[i:i+1], A[i:i+1])  # d(A_i,P_i)+m
        wrong = False
        for j in range(n):
            crit = criterion(A[j:j+1], P[i:i+1], A[j:j+1])
            if loss.item() > crit.item():
                wrong = True; break
        correct += (0 if wrong else 1)
    return 100*correct/n

def vectorized_acc(A, P, chunk=64):
    n = A.shape[0]; correct = 0
    for s in range(0, n, chunk):
        e = min(s+chunk, n)
        dists = torch.cdist(P[s:e], A)
        own = dists[torch.arange(e-s), torch.arange(s,e)]
        best,_ = dists.min(dim=1)
        correct += int((own <= best + 1e-6).sum())
    return 100*correct/n

def run():
    torch.manual_seed(0)
    N, D = 200, 16
    A = torch.randn(N, D)
    # positives near their own anchor most of the time, sometimes near a wrong one
    P = A.clone()
    for i in range(N):
        if i % 5 == 0:                      # ~20% hard/wrong
            P[i] = A[(i+1) % N] + 0.05*torch.randn(D)
        else:
            P[i] = A[i] + 0.05*torch.randn(D)
    crit = nn.TripletMarginLoss(1.0, 2.0, reduction='none')
    o = original_acc(A, P, crit)
    v = vectorized_acc(A, P)
    print(f"original={o:.2f}%  vectorized={v:.2f}%  match={abs(o-v)<1e-9}")
    assert abs(o - v) < 1e-9, (o, v)
    print("OK: vectorized validate_triplet accuracy matches original")

if __name__ == "__main__":
    run()
