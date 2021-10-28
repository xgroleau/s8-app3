import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm

# Import from text files
from src.classification.bayesian_classifier import BayesianClassifier


def calc_erreur_classification(original_data, classified_data):
    # génère le vecteur d'erreur de classification
    vect_err = np.absolute(original_data - classified_data).astype(bool)
    indexes = np.array(np.where(vect_err == True))[0]

    return indexes

C1 = np.loadtxt('test/C1.txt')
C2 = np.loadtxt('test/C2.txt')
C3 = np.loadtxt('test/C3.txt')

training_set = {
    "C1": C1,
    "C2": C2,
    "C3": C3
}

classifier = BayesianClassifier(training_set)

data = [C1, C2, C3]
data = np.array(data)
x, y, z = data.shape
data = data.reshape(x*y, z)
ndata = len(data)
class_labels = np.zeros(ndata)
class_labels[range(len(C1), 2*len(C1))] = 1
class_labels[range(2*len(C1), ndata)] = 2

classes_orig = np.array([classifier.fit(d, likelihood="gaussian", cost_matrix=np.array([[1,1,1],[1,1,1],[1,1,1]])) for d in data])
error_class = 6  #optionnel, assignation d'une classe différente à toutes les données en erreur, aide pour la visualisation
error_indexes = calc_erreur_classification(class_labels, classes_orig)
#error_indexes = np.expand_dims(error_indexes, 1)
classes_orig[error_indexes] = error_class
print(f'Taux de classification moyen sur l\'ensemble des classes: {100*(1-len(error_indexes)/ndata)}%')

xmin = -8
xmax = 9
ymin = -6
ymax = 17

npoints = 40
donnee_test = np.transpose(np.array([(xmax-xmin)*np.random.random(npoints)+xmin, \
    (ymax-ymin)*np.random.random(npoints)+ymin]))

classes = np.array([classifier.fit(d) for d in donnee_test])

cmap = cm.get_cmap('prism')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.scatter(data[:, 0], data[:, 1], s=5, color=cmap(classes_orig / error_class / .75))
ax2.scatter(donnee_test[:, 0], donnee_test[:, 1], s=5, color=cmap(classes / error_class / .75))
ax3.scatter(data[:, 0], data[:, 1], s=5, color=cmap(class_labels / error_class / .75))
ax1.set_title('Données d\'origine reclassées')
ax2.set_title('Classification des données aléatoires selon Bayes')
ax3.set_title('Données d\'origine')
ax1.set_xlim([xmin, xmax])
ax1.set_ylim([ymin, ymax])
ax2.set_xlim([xmin, xmax])
ax2.set_ylim([ymin, ymax])
ax3.set_xlim([xmin, xmax])
ax3.set_ylim([ymin, ymax])

plt.show()