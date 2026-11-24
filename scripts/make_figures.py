"""Regenerates Fig1-Fig7 of the manuscript from the CSV files in ../results (PNG 300 dpi, PDF)."""
import glob, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from math import exp
HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,"..","results"); OUT=os.path.join(HERE,"..","figures")
os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,"legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,
    "axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":300,"font.family":"DejaVu Sans"})
EPS=r"$\varepsilon_{\mathrm{tot}}$"
NAME={"adult":"Adult","bank_marketing":"Bank marketing","credit_approval":"Credit approval","diabetes":"Diabetes","german":"German","gmc":"Give Me Some Credit",
      "lending_club":"Lending Club","loan":"Loan","taiwan":"Taiwan"}
ORDER=["loan","german","taiwan","credit_approval","gmc","lending_club","adult","bank_marketing","diabetes"]
U=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES,"util_*.csv")))])
RE=pd.read_csv(os.path.join(RES,"table_reid_all.csv")); MI=pd.read_csv(os.path.join(RES,"table_mia_all.csv"))
AL=pd.read_csv(os.path.join(RES,"alloc_summary_rounded.csv"))
npv=U[U.method=="LR_nonprivate"].set_index("dataset").auc
COL={"central_DPLR":"#0072B2","central_DPGNB":"#56B4E9","local_LR":"#009E73","local_NB":"#D55E00"}
LAB={"central_DPLR":"Central DP-LR","central_DPGNB":"Central DP-GNB","local_LR":"Local LR","local_NB":"Local NB"}
def save(fig,n): fig.savefig(os.path.join(OUT,n+".png"),bbox_inches="tight"); fig.savefig(os.path.join(OUT,n+".pdf"),bbox_inches="tight"); plt.close(fig)
def tab(t): return U[U.total_eps==t].pivot_table(index="dataset",columns="method",values="auc")
# Fig 1: loan case study (numbers from the loan experiments reported in Section 5.1)
rows=[("Majority class",68.7,"#7f7f7f"),("Credit_History rule only",80.9,"#009E73"),("RF, no privacy",80.9,"#009E73"),("RF without Credit_History",64.2,"#7f7f7f"),
("Partial noise: numeric only (ε=1, not ε-DP)",79.6,"#D55E00"),("Noise on Credit_History, weight 0.2 (ε=1)",77.6,"#D55E00"),("Noise on Credit_History, weight 1.0 (ε=1)",66.6,"#D55E00"),
("Full local DP, ε=1 per attr. (ε_tot=11)",66.8,"#0072B2"),("Full local DP, ε=5 per attr. (ε_tot=55)",79.0,"#0072B2")]
fig,ax=plt.subplots(figsize=(7.0,3.1)); y=np.arange(len(rows))[::-1]
for yi,(l,v,c) in zip(y,rows): ax.barh(yi,v,color=c,height=.65); ax.text(v+.8,yi,f"{v:.1f}",va="center",fontsize=7)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0,100); ax.set_xlabel("Accuracy (%), 10-fold CV, Loan dataset (n = 614)"); ax.axvline(68.7,ls=":",c="grey",lw=.8)
fig.tight_layout(); save(fig,"Fig1")
# Fig 2
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3),sharey=True); x=np.arange(9); w=.2
for ax,t,pl in zip(axs,[25,50],["(a)","(b)"]):
    tb=tab(t)
    for k,m in enumerate(["local_NB","local_LR","central_DPLR","central_DPGNB"]):
        ax.bar(x+(k-1.5)*w,[tb.loc[d,m] for d in ORDER],w,color=COL[m],label=LAB[m])
    ax.plot(x,[npv[d] for d in ORDER],"k_",ms=11,mew=1.2,label="No privacy (LR)")
    ax.set_xticks(x); ax.set_xticklabels([NAME[d] for d in ORDER],rotation=40,ha="right"); ax.set_title(f"{pl} Total ε = {t}",pad=4); ax.set_ylim(0.5,1.0)
axs[0].set_ylabel("AUC (10-fold CV; axis starts at 0.5)"); h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc="upper center",ncol=5,frameon=False,bbox_to_anchor=(.5,1.05)); fig.tight_layout(rect=(0,0,1,.93)); save(fig,"Fig2")
# Fig 3
fig,axs=plt.subplots(3,3,figsize=(7.2,6.2)); axs=axs.ravel()
for ax,d in zip(axs,ORDER):
    g=U[U.dataset==d]
    for m in ["central_DPLR","central_DPGNB","local_LR","local_NB"]:
        gg=g[g.method==m].sort_values("total_eps"); ax.errorbar(gg.total_eps,gg.auc,yerr=gg.sd,marker="o",ms=3,lw=1.1,capsize=1.5,color=COL[m])
    ax.axhline(npv[d],ls="--",c="k",lw=.8); ax.set_xscale("log"); ax.set_xticks([10,25,50,100]); ax.set_xticklabels([10,25,50,100]); ax.minorticks_off(); ax.set_title(NAME[d],pad=3)
h=[plt.Line2D([],[],color=COL[m],marker="o",ms=3,label=LAB[m]) for m in COL]+[plt.Line2D([],[],color="k",ls="--",label="No privacy (LR)")]
fig.legend(handles=h,loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(.5,-0.01))
fig.supxlabel("Total privacy budget "+EPS,fontsize=8,y=0.045); fig.supylabel("AUC (10-fold CV)",fontsize=8); fig.tight_layout(rect=(0,0.07,1,1)); save(fig,"Fig3")
# Fig 4: truthfulness of randomized response
e=np.linspace(0,8,400); fig,ax=plt.subplots(figsize=(3.6,3.0))
for k,ls in [(2,"-"),(4,"--"),(8,":")]: ax.plot(e,np.exp(e)/(np.exp(e)+k-1),ls,c="k",lw=1.2,label=f"k = {k} categories")
dd=U.drop_duplicates("dataset").set_index("dataset").d; ea=[50/(v+1) for v in dd.values]
ax.plot(ea,[exp(x)/(exp(x)+1) for x in ea],"o",c="#D55E00",ms=4,label=f"{EPS} = 50, binary attribute")
ax.set_xlabel("Per-attribute ε"); ax.set_ylabel("P(true category reported)"); ax.set_ylim(0,1.02); ax.legend(frameon=False,loc="lower right"); fig.tight_layout(); save(fig,"Fig4")
# Fig 5
pal=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#000000","#999999","#8c564b"]
fig,axs=plt.subplots(1,2,figsize=(7.2,3.2))
for d,c in zip(ORDER,pal):
    g=RE[(RE.dataset==d)&(RE.pool==500)].sort_values("total_eps"); axs[0].plot(g.total_eps,g.top1,marker="o",ms=3,lw=1.1,color=c,label=NAME[d])
    g=RE[(RE.dataset==d)&(RE.total_eps==50)].sort_values("pool"); axs[1].plot(g.pool,g.top1,marker="o",ms=3,lw=1.1,color=c)
axs[0].axhline(1/500,ls=":",c="grey"); axs[0].text(10.5,1/500*1.3,"random guess",fontsize=6.5,color="grey")
axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xticks([10,25,50,100]); axs[0].set_xticklabels([10,25,50,100]); axs[0].minorticks_off()
axs[0].set_xlabel("Total privacy budget "+EPS); axs[0].set_ylabel("Top-1 re-identification rate"); axs[0].set_title("(a) Pool of 500 candidates",pad=4); axs[0].legend(frameon=False,fontsize=6,loc="upper left")
pools=[100,500,2000]; axs[1].plot(pools,[1/p for p in pools],ls=":",c="grey"); axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xticks(pools); axs[1].set_xticklabels(pools); axs[1].minorticks_off()
axs[1].set_xlabel("Candidate pool size"); axs[1].set_ylabel("Top-1 rate at "+EPS+" = 50"); axs[1].set_title("(b) Effect of pool size",pad=4); fig.tight_layout(); save(fig,"Fig5")
# Fig 6
M=MI.pivot_table(index="dataset",columns="model",values="mia_auc"); E=MI.pivot_table(index="dataset",columns="model",values="ci95")
cols=["LR_nonprivate","DPLR_eps10","DPLR_eps25","DPLR_eps50","DPLR_eps100","RF_overfit_control"]
LB={"LR_nonprivate":"LR, no privacy","DPLR_eps10":"DP-LR, "+EPS+" = 10","DPLR_eps25":"DP-LR, "+EPS+" = 25","DPLR_eps50":"DP-LR, "+EPS+" = 50","DPLR_eps100":"DP-LR, "+EPS+" = 100","RF_overfit_control":"Over-fitted RF (control)"}
pp=["#7f7f7f","#0072B2","#56B4E9","#009E73","#E69F00","#CC79A7"]; fig,ax=plt.subplots(figsize=(7.2,3.2)); w=.8/6
for k,c in enumerate(cols):
    ax.bar(np.arange(9)+(k-2.5)*w,[M.loc[d,c] for d in ORDER],w,yerr=[E.loc[d,c] for d in ORDER],capsize=1,color=pp[k],label=LB[c],error_kw={"lw":.6})
ax.axhline(.5,ls=":",c="k",lw=.8); ax.set_ylim(.45,.9); ax.set_xticks(np.arange(9)); ax.set_xticklabels([NAME[d] for d in ORDER],rotation=30,ha="right")
ax.set_ylabel("Membership-inference AUC (axis starts at 0.45)"); ax.legend(frameon=False,ncol=3,fontsize=6.5,loc="upper left"); fig.tight_layout(); save(fig,"Fig6")
# Fig 7 (from alloc_summary_rounded.csv: values rounded to 3 decimals as printed by the Step-11 notebook)
spec=[("NB proportional","prop_NB","uniform_NB","#D55E00"),("NB top-k","topk_NB","uniform_NB","#E69F00"),("LR proportional","prop_LR","uniform_LR","#009E73"),("LR top-k","topk_LR","uniform_LR","#56B4E9")]
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3),sharey=True); H=[]
for ax,t,pl in zip(axs,[25,50],["(a)","(b)"]):
    g=AL[AL.total_eps==t].set_index("dataset")
    for k,(lab,a,b,c) in enumerate(spec):
        h=ax.bar(np.arange(9)+(k-1.5)*.2,[g.loc[d,a]-g.loc[d,b] for d in ORDER],.2,color=c,label=lab)
        if t==25: H.append(h)
    ax.axhline(0,c="k",lw=.8); ax.set_xticks(np.arange(9)); ax.set_xticklabels([NAME[d] for d in ORDER],rotation=40,ha="right"); ax.set_title(f"{pl} Total ε = {t}",pad=4)
axs[0].set_ylabel("AUC gain over uniform allocation"); fig.legend(handles=H,labels=[s[0] for s in spec],loc="upper center",ncol=4,frameon=False,bbox_to_anchor=(.5,1.05))
fig.tight_layout(rect=(0,0,1,.93)); save(fig,"Fig7"); print("figures written to",OUT)
