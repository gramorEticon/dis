import sys
from PyQt5.QtWidgets import QApplication
from united_metric_of_recommender_systen.sliders.ui import SliderApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SliderApp()
    ex.show()
    sys.exit(app.exec_())
