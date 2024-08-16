import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtChart import QChart, QChartView, QBarSet, QBarSeries, QBarCategoryAxis, QValueAxis
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Пример столбчатой диаграммы")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout(self.central_widget)
        self.chart_view = QChartView()
        layout.addWidget(self.chart_view)

        self.create_bar_chart()

    def create_bar_chart(self):
        series = QBarSeries()

        set0 = QBarSet('2019')
        set0.append([1,2,3,4])



        series.append(set0)


        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Пример столбчатой диаграммы")
        chart.setAnimationOptions(QChart.SeriesAnimations)

        categories = ['Категория 1', 'Категория 2', 'Категория 3', 'Категория 4', 'Категория 5']
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setRange(0, 7)
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        self.chart_view.setChart(chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
