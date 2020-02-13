import scipy.linalg
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import csv


# Document CSV ouvert
##################################################################################
with open("data.csv", newline='') as file:
	reader = csv.reader(file)
	data = list(map(tuple,reader))
##################################################################################


# Cette boucle fait la moyenne des données mensuelles
# Et changer les jours de l'année pour des mois
##################################################################################
pointX = []
pointY = []

j = 1
vetor = []

for i in range(2,len(data)-1,1):
	vetor.append(data[i][0])
	if(i%27 == 0):
		pointX.append(j)
		j += 1
		vetor = list(map(float, vetor))
		pointY.append(sum(vetor) / (float)(len(vetor)))
		vetor = []
##################################################################################


# Transformez les données de chaîne en flottant
##################################################################################
pointX = list(map(float, pointX))
pointY = list(map(float, pointY))
##################################################################################
  
p = 10000  # P défini par l'exercice 1

# Chiffres après une virgule dans les données,
# Lorsqu'elle est supérieure au nombre de chiffres décimaux, la précision d'interpolation augmente
decimalPlaces = 10**(2) 



n = len(pointX) - 1; # Nombre de splines à réaliser

a = []; b = []; c = []; d = []; # Coefficients de la fonction d'interpolation
h = []; g = []; # Tableaux auxiliaires utilisées pour calculer a, b, c, d 
T = []; Q = []; Qt = []; # Matrices auxiliaires utilisées pour calculer a, b, c, d 


# Calcul des tableaux auxiliaires g et h
##################################################################################
for i in range (n):
    h.append(pointX[i+1] - pointX[i])
    g.append(1/(pointX[i+1] - pointX[i]))

#Transformé en np.array pour faciliter les futurs calculs
h = np.array(h)
g = np.array(g)
##################################################################################


# Calcul des matrices auxiliaires Q , Qt et T
##################################################################################
for i in range(n-1):
    aux = []
    for j in range(n-1):
        if i == j:
            aux.append(2*(h[i]+h[i+1])/3)
        elif i == j+1:
            aux.append(h[i+1]/3)
        elif i == j-1:
            aux.append(h[i+1]/3)
        else:
            aux.append(0)
    T.append(aux)
T = np.array(T)

for i in range(n+1):
    aux = []
    for j in range(n-1):
        if i == j:
            aux.append(g[i])
        elif i == j+1:
            aux.append(-g[j]-g[j+1])
        elif i == j+2:
            aux.append(g[j+1])
        else:
            aux.append(0)
    Q.append(aux)
Q = np.array(Q)
Qt = Q.transpose()
##################################################################################


# Calcul des Coefficients c et a 
# auxMat1 et auxMat2 ont été utilisés uniquement pour des calculs intermédiaires
##################################################################################
auxMat1 = np.matmul(Qt,Q) + T*p
auxMat1 = np.linalg.inv(auxMat1)
auxMat2 = np.matmul(Qt,pointY)*p

c = np.matmul(auxMat1,auxMat2)
a = pointY - np.matmul(Q,c)/p
a = np.array(a)
##################################################################################


# Zéro a été ajouté au début et à la fin du tableau c
##################################################################################
auxC = np.array([0.0])
c = np.concatenate((auxC , c))
c = np.concatenate((c , auxC))
##################################################################################


# Calcul des Coefficients d et b 
# auxB a été utilisés uniquement pour des calculs intermédiaires
##################################################################################
for i in range(n):
    d.append((c[i+1] - c[i])/(3*h[i]))
d = np.array(d)

for i in range(n):
    auxB = (a[i+1] - a[i])/(h[i]) 
    auxB = auxB - (c[i]*h[i])
    auxB = auxB - (d[i]*h[i]*h[i])
    b.append(auxB)
b = np.array(b)
##################################################################################


# La première boucle for sert à itérer entre les splines
# La deuxième fonction consiste à placer les points d'intervalle de spline dans la fonction
# La fonction round a été utilisée car la boucle for n'accepte que des entiers
# Acquisition des points F(y)
##################################################################################
Fy = []
for j in range(n):
    for i in range(round(pointX[j]*decimalPlaces),round(pointX[j+1]*decimalPlaces),1):
        Fy.append(a[j]+b[j]*(i/decimalPlaces - pointX[j])+c[j]*(i/decimalPlaces - pointX[j])**2+d[j]*(i/decimalPlaces- pointX[j])**3)
##################################################################################


# Plot avec notre fonction
##################################################################################
Fy = np.array(Fy)
Fx = np.linspace(min(pointX),max(pointX), num=Fy.size, endpoint=True)

fig = plt.gcf()
plt.plot(pointX, pointY, 'o',Fx, Fy, '-')
plt.legend(['Points', 'Notre Splines'], loc='best')
plt.show()
fig.savefig('notreSplinesBiggerData.png', format='png')
##################################################################################

# Plot du Python3
##################################################################################
F = interp1d(pointX, pointY, kind='cubic')
X = np.linspace(1, 12, num=10000, endpoint=True)

fig = plt.gcf()
plt.plot(pointX, pointY, 'o',X, F(X), '-')
plt.legend(['Points', 'Python splines cubic'], loc='best')
plt.show()
fig.savefig('pythonSplinesBiggerData.png', format='png')
##################################################################################