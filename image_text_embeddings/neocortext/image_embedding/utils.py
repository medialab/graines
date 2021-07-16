
from matplotlib.offsetbox import OffsetImage
import matplotlib.pyplot as plt

def getImage(path):
    return OffsetImage(plt.imread(path))