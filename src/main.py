import numpy as np
import matplotlib.pyplot as  plt
import argparse
from LogisticRegression import LogisticRegression

parser=argparse.ArgumentParser()
parser.add_argument("--no_it","-n",type=int,required=True)
parser.add_argument("--interval","-i",type=str,required=True)
args=parser.parse_args()

a,b,c=[float(o) for o in args.interval.split(",")]

entradas=np.genfromtxt('../samples/entradas_x.csv', delimiter=',').T
saidas=np.genfromtxt('../samples/saidas_y.csv', delimiter=',')

plt.rc('text', usetex=True)
xs=np.array(range(1,args.no_it+1))
ans=[]
for alpha in np.arange(a,b+c,c):
	model=LogisticRegression(*entradas,saidas,alpha=alpha,normalize=True)
	errs=np.zeros((args.no_it))
	for i in range(args.no_it):
		thetao,j=next(model)
		errs[i]=j
	ans.append([alpha,j,*thetao])
	plt.plot(xs,errs,label=rf"$\alpha={alpha}$")
header=";".join(["alpha","j",*(f"t{i}" for i in range(12))])
np.savetxt("results.csv", ans, delimiter=";",header=header)
plt.legend(fontsize=8)
plt.xlabel('Números de iteração')
plt.ylabel(r'$J(\theta)$')
plt.title(rf'Gráfico de $J(\theta)$ para diferentes alphas que variam ${a}$ e ${b}$, passo {c}')
plt.show()