import sys
import matplotlib
matplotlib.use('Qt5Agg')

from PySide2 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def getHistGraph():
    fig = Figure(figsize=(5, 4), dpi=100)
    '''
    fig = plt.hist([1,1,1,2,2,2,2,2,2,3,3,4,5,5,5],5)
    print(type(fig))
    print("len(fig) = ", len(fig))
    for i in fig:
        print(i)
    print(type(fig[2]))
    '''
    return fig.add_subplot(111).hist([1,1,1,2,2,2,2,2,2,3,3,4,5,5,5],5)

def getPlotGraph():
    fig = plt.plot([0,1,2,3,4], [10,1,20,3,40])
    print(type(fig))
    print("len(fig) = ", len(fig))
    for i in fig:
        print(i)
    return fig

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

    def plotHist(self, hist):
        return self.fig.add_subplot(111).hist([1,1,1,2,2,2,2,2,2,3,3,4,5,5,5],5)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        #sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        sc.axes.hist([1,1,1,2,2,2,2,2,2,3,3,4,5,5,5], 5)
        #sc.axes = sc.fig.add_subplot(111).hist([1,1,1,2,2,2,2,2,2,3,3,4,5,5,5],5)
        #sc.axes = sc.plotHist(getHistGraph())
        

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(sc, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec_()