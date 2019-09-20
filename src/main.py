import numpy as np
import matplotlib.pyplot as  plt
import argparse
from LogisticRegression import LogisticRegression
from ast import literal_eval
from collections import namedtuple

ModelEntry=namedtuple("ModelEntry",["model","errs","accrs"])

def interval(s):
	global GRAPH_TITLE
	ans=literal_eval(s)
	if isinstance(ans,tuple) and len(ans)==3 and all(isinstance(x, (float,int)) for x in ans):
		a,b,c=ans
		GRAPH_TITLE=rf'Gráfico de $J(\theta)$ para diferentes alphas que variam ${a}$ e ${b}$, passo {c}'
		return np.arange(a,b+c,c)
	if isinstance(ans,list) and all(isinstance(x, (float,int)) for x in ans):
		GRAPH_TITLE=rf'Gráfico de $J(\theta)$ para os alphas {ans}'
		return ans
	raise TypeError()
GRAPH_TITLE=""
parser=argparse.ArgumentParser()
parser.add_argument("--no_it","-n",type=int,required=True)
parser.add_argument("--interval","-i",type=interval,required=True)
parser.add_argument("--normalize",action="store_true")
args=parser.parse_args()

entradas=np.genfromtxt('../samples/entradas_x.txt', delimiter=',').T
saidas=np.genfromtxt('../samples/saidas_y.txt', delimiter=',')

plt.rc('text', usetex=True)
entries=[]
for alpha in args.interval:
	model=LogisticRegression(*entradas,saidas,alpha=alpha,normalize=True)
	errs=np.zeros((args.no_it))
	acrs=errs.copy()
	for i in range(args.no_it):
		acrs[i]=model.accuracy()
		thetao,j=next(model)
		errs[i]=j
	entries.append(ModelEntry(model,errs,acrs))

xs=np.array(range(1,args.no_it+1))
for entry in entries:
	plt.plot(xs,entry.errs,label=rf"$\alpha={entry.model.alpha}$")
plt.legend(fontsize=8)
plt.xlabel('Números de iteração')
plt.ylabel(r'$J(\theta)$')
plt.title(GRAPH_TITLE)

plt.figure()
for entry in entries:
	plt.plot(xs,entry.accrs,label=rf"$\alpha={entry.model.alpha}$")
plt.legend(fontsize=8)
plt.xlabel('Números de iteração')
plt.ylabel(r'acurácia')
plt.title(GRAPH_TITLE.replace(r"$J(\theta)$","acurácia"))
plt.show()