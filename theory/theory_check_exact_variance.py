import numpy as np
from math import exp, log, ceil, sqrt
rng=np.random.default_rng(1)
def pq(eps,k): return exp(eps)/(exp(eps)+k-1), 1/(exp(eps)+k-1)
def exact_var(n,eA,eX,k,pi,a,x):
    pA,qA=pq(eA,2); pX,qX=pq(eX,k); D=(pA-qA)*(pX-qX)
    pia=pi[a,:].sum(); pix=pi[:,x].sum(); pax=pi[a,x]
    P11=qA*qX+qA*(pX-qX)*pix+qX*(pA-qA)*pia+D*pax
    P10=qA+(pA-qA)*pia-P11; P01=qX+(pX-qX)*pix-P11; P00=1-P11-P10-P01
    EU2V2=(1-qA)**2*(1-qX)**2*P11+qX**2*(1-qA)**2*P10+qA**2*(1-qX)**2*P01+qA**2*qX**2*P00
    return (EU2V2/D**2-pax**2)/n, D, (P11,P10,P01,P00)
def sim(n,eA,eX,k,pi,a,x,R=60000):
    pA,qA=pq(eA,2); pX,qX=pq(eX,k); D=(pA-qA)*(pX-qX)
    My=np.full((2,2),qA); np.fill_diagonal(My,pA); Mx=np.full((k,k),qX); np.fill_diagonal(Mx,pX)
    Pn=My@pi@Mx.T; cnt=rng.multinomial(n,Pn.ravel(),size=R).reshape(R,2,k)
    T=(cnt-qA*cnt.sum(1,keepdims=True)-qX*cnt.sum(2,keepdims=True)+qA*qX*n)/D
    return T[:,a,x]/n, Pn
pi=np.array([[0.10,0.15,0.05],[0.20,0.30,0.20]])
for (n,eA,eX,k,a,x) in [(500,1.0,1.5,3,0,1),(4500,0.3,0.3,3,0,1),(900,2.0,0.7,3,1,2),(500,3.0,3.0,3,1,1)]:
    v,D,P=exact_var(n,eA,eX,k,pi,a,x); est,Pn=sim(n,eA,eX,k,pi,a,x)
    print(f"n={n} eA={eA} eX={eX}: var_sim={est.var():.4e} exact={v:.4e} mean={est.mean():.4f} true={pi[a,x]}  P11 formula {P[0]:.5f} vs matrix {Pn[a,x]:.5f}")
# Hoeffding sample size, binary attribute, eps_A=eps_X=eps, t=0.05, delta=0.05
print("\nrequired n (t=0.05, delta=0.05):")
res={}
for eps in (1,2,3,5):
    pA,qA=pq(eps,2); D=(pA-qA)**2; n=ceil(2*log(2/0.05)/(0.05**2*D**2)); res[eps]=n; print(eps,"D=%.4f"%D,"n_req=",n)
# empirical failure prob at n_req (joint cell of binary attr)
pi2=np.array([[0.2,0.1],[0.3,0.4]])
for eps,n in res.items():
    est,_=sim(n,eps,eps,2,pi2,1,1,R=20000); print(f"eps={eps}: n={n}  P(|err|>=0.05) simulated = {(np.abs(est-pi2[1,1])>=0.05).mean():.5f}  (Hoeffding guarantee <= 0.05)")
