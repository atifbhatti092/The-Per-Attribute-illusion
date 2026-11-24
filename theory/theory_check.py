import numpy as np, itertools
from math import exp, log, sinh, asinh, tanh, sqrt
rng=np.random.default_rng(0)
def pq(eps,k): 
    p=exp(eps)/(exp(eps)+k-1); q=1/(exp(eps)+k-1); return p,q

# ---- Prop A: variance of debiased frequency (k-ary RR) ----
def var_formula(n,eps,k,pi):
    e=exp(eps); return (1/n)*(pi*(1-pi)+(e+k-2)/(e-1)**2+pi*(k-2)/(e-1))
def sim_freq(n,eps,k,pi,R=200000):
    p,q=pq(eps,k); theta=q+(p-q)*pi
    # number of reports equal to v ~ Binomial(n, theta)
    c=rng.binomial(n,theta,size=R); est=(c/n-q)/(p-q); return est.mean(), est.var()
print("== Prop A: variance of debiased frequency ==")
for (n,eps,k,pi) in [(500,1.0,2,0.3),(500,2.0,4,0.3),(4500,3.0,8,0.15),(553,0.5,3,0.5)]:
    m,v=sim_freq(n,eps,k,pi); print(f"n={n} eps={eps} k={k} pi={pi}: mean={m:.4f} (true {pi}) var_sim={v:.3e} var_formula={var_formula(n,eps,k,pi):.3e}")

# ---- Prop B: joint counts closed form, unbiasedness and variance ----
def joint_check(n,ex,kx,ey,pi,R=40000):
    px,qx=pq(ex,kx); py,qy=pq(ey,2); D=(py-qy)*(px-qx)
    ny=2
    cells=[(a,b) for a in range(ny) for b in range(kx)]
    pi=np.array(pi).reshape(ny,kx)
    # noisy cell probs
    My=np.full((2,2),qy); np.fill_diagonal(My,py); Mx=np.full((kx,kx),qx); np.fill_diagonal(Mx,px)
    Pn=My@pi@Mx.T
    ests=np.zeros((R,ny,kx))
    cnt=rng.multinomial(n,Pn.ravel(),size=R).reshape(R,ny,kx)
    Np=cnt.sum(1,keepdims=True)  # N_{+b}
    Na=cnt.sum(2,keepdims=True)  # N_{a+}
    T=(cnt-qy*Np-qx*Na+qx*qy*n)/D
    # compare with matrix inversion form
    Tm=np.einsum('ij,rjk,kl->ril',np.linalg.inv(My),cnt,np.linalg.inv(Mx).T)
    # exact variance of one cell via g
    a,b=0,1
    P11=Pn[a,b]; P10=Pn[a,:].sum()-P11; P01=Pn[:,b].sum()-P11; P00=1-P11-P10-P01
    Eg2=((1-qx)**2*(1-qy)**2*P11+qx**2*(1-qy)**2*P10+qy**2*(1-qx)**2*P01+qx**2*qy**2*P00)/D**2
    var_th=(Eg2-pi[a,b]**2)/n
    return np.abs(T-Tm).max(), (T/n).mean(0), pi, (T[:,a,b]/n).var(), var_th, 1/(n*D**2), ((1-qx)**2*(1-qy)**2*qx*qy/D**2-1)/n
print("\n== Prop B: joint counts ==")
pi=[0.1,0.2,0.3,0.4/1 if False else 0.0]  # placeholder
pi_ex=[[0.10,0.15,0.05],[0.20,0.30,0.20]]
d,mean,pi_t,vs,vt,ub,lb=joint_check(500,1.5,3,1.0,np.array(pi_ex))
print("max |closed form - matrix inverse| =",d)
print("mean of estimator vs true:\n",np.round(mean,4),"\n",pi_t)
print(f"cell var simulated={vs:.3e} exact formula={vt:.3e} upper bound 1/(nD^2)={ub:.3e} lower bound={lb:.3e}")
d,mean,pi_t,vs,vt,ub,lb=joint_check(4500,0.3,3,0.3,np.array(pi_ex))
print(f"small eps: cell var simulated={vs:.3e} exact={vt:.3e} upper={ub:.3e} lower={lb:.3e}")

# ---- Prop C: local vs central variance for a proportion ----
print("\n== Prop C ==")
def ratio(n,eps): x=eps/2; return n/2*(x/sinh(x))**2
for n,eps in [(500,1),(500,5),(500,10)]:
    loc=exp(eps)/(n*(exp(eps)-1)**2); cen=2/(n**2*eps**2); print(f"n={n} eps={eps}: local/central priv-variance = {loc/cen:.3f}  formula {ratio(n,eps):.3f}")
def eq_eps(n,ec): return 2*asinh(ec*sqrt(n/8))
for n in [553,4500,13500]:
    print(n,[round(eq_eps(n,ec),2) for ec in (0.5,1,5,10)])
# verify equal variance
n=4500; ec=1; el=eq_eps(n,ec); print("check equal var:",exp(el)/(n*(exp(el)-1)**2), 2/(n**2*ec**2))

# ---- Prop D: linkage bound, exact Bayes success for small example ----
print("\n== Prop D: Bayes-optimal linkage vs bound ==")
def bayes_succ(cands,eps_attr):
    p,q=pq(eps_attr,2); d=len(cands[0]); best=0
    zs=list(itertools.product([0,1],repeat=d)); tot=0
    for z in zs:
        lik=[np.prod([p if z[i]==c[i] else q for i in range(d)]) for c in cands]
        tot+=max(lik)
    return tot/len(cands)
cands=[(0,0,0),(1,0,0),(0,1,1),(1,1,0),(1,1,1)]
for ea in [0.1,0.5,1.0,2.0]:
    et=3*ea; m=len(cands); print(f"eps_attr={ea}: eps_tot={et:.1f} exact Bayes success={bayes_succ(cands,ea):.4f} bound={exp(et)/(exp(et)+m-1):.4f}")
for m in (100,500,2000):
    print("m",m,"threshold ln(m-1)=%.2f"%log(m-1), [round(exp(e)/(exp(e)+m-1),4) for e in (5,10,25,50)])
print("\n== Prop E: membership advantage bound tanh(eps/2) ==")
print([ (e,round(tanh(e/2),4)) for e in (1,2,5,10,25)])
# ---- Prop F: Holder allocation ----
c=np.array([1.,8.,27.,0.5]); E=10.
opt=E*c**(1/3)/np.sum(c**(1/3)); val=np.sum(c/opt**2); print("\n== Prop F ==  opt",opt.round(3),"value",val,"formula",np.sum(c**(1/3))**3/E**2,"equal split",np.sum(c/(E/4)**2))
rs=[np.sum(c/(w)**2) for w in (np.random.default_rng(1).dirichlet(np.ones(4),5000)*E)]; print("min over random splits",min(rs),">=",val)
