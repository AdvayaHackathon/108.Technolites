from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class SeverityGraphCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 2))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_confidences(self, confidences, labels):
        self.ax.clear()
        self.ax.bar(labels, confidences, color='darkblue')
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel('Confidence')
        self.ax.set_title('Prediction Confidence by Severity')
        self.draw()
