from PyQt5.QtChart import QChartView, QBarSeries, QBarSet, QChart, QBarCategoryAxis, QValueAxis
from PyQt5.QtGui import QPainter


from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QComboBox
from PyQt5.QtCore import Qt

from united_metric_of_recommender_systen.sliders.method import Methods, NormMethod


class SliderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.current_pos = [50,50,50,50]
        self.current_value = [1/len(self.current_pos)]*len(self.current_pos)
        self.current_method = 0
        self.methods = Methods
        self.initUI()

    def pool(self):
        self.onGlobalChange()
        self.initUI()


    def initUI(self):
        self.setWindowTitle('User Score')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.comboBox = QComboBox(self)
        self.comboBox.addItem('Без нормализации', 0)
        self.comboBox.addItem('Сигмоида', 1)
        self.comboBox.addItem('Корень (Насыщение)', 2)
        self.comboBox.addItem('Сигмоида (Насыщение)', 3)
        self.comboBox.currentIndexChanged.connect(self.onComboBoxChange)
        layout.addWidget(self.comboBox)

        self.label1 = QLabel('50', self)
        layout.addWidget(self.label1)
        self.slider1 = QSlider(Qt.Horizontal, self)
        self.slider1.setValue(50)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(100)
        self.slider1.valueChanged[int].connect(self.onSlider1Change)
        layout.addWidget(self.slider1)

        self.label2 = QLabel('50', self)
        layout.addWidget(self.label2)
        self.slider2 = QSlider(Qt.Horizontal, self)
        self.slider2.setMinimum(0)
        self.slider2.setValue(50)
        self.slider2.setMaximum(100)
        self.slider2.valueChanged[int].connect(self.onSlider2Change)
        layout.addWidget(self.slider2)

        self.label3 = QLabel('50', self)
        layout.addWidget(self.label3)
        self.slider3 = QSlider(Qt.Horizontal, self)
        self.slider3.setMinimum(0)
        self.slider3.setValue(50)
        self.slider3.setMaximum(100)
        self.slider3.valueChanged[int].connect(self.onSlider3Change)
        layout.addWidget(self.slider3)

        self.label4 = QLabel('50', self)
        layout.addWidget(self.label4)
        self.slider4 = QSlider(Qt.Horizontal, self)
        self.slider4.setMinimum(0)
        self.slider4.setValue(50)
        self.slider4.setMaximum(100)
        self.slider4.valueChanged[int].connect(self.onSlider4Change)
        layout.addWidget(self.slider4)


        self.val1 = QLabel(f"P1: {str(self.current_value[0])}  P2: {str(self.current_value[1])}  P3: {str(self.current_value[2])}  P4: {str(self.current_value[3])}  ",self)
        layout.addWidget(self.val1)

        self.chart_view = QChartView()
        layout.addWidget(self.chart_view)

        self.setLayout(layout)
        self.draw_bar_chart()

    def draw_bar_chart(self):
        series = QBarSeries()

        set0 = QBarSet("")
        set0.append(self.current_value)


        series.append(set0)

        chart = QChart()
        chart.setAnimationOptions(QChart.NoAnimation)
        chart.addSeries(series)
        # chart.setTitle("Пример столбчатой диаграммы")
        chart.setAnimationOptions(QChart.SeriesAnimations)

        categories = [f"P{i}" for i in range(0, len(self.current_pos))]
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setRange(0, 1)
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        self.chart_view.setChart(chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)

    def onComboBoxChange(self, index):
        self.current_method = index
        self.onGlobalChange()

    def onSlider1Change(self, value):
        self.label1.setText(str(value))
        self.current_pos[0] = value
        self.onGlobalChange()

    def onSlider2Change(self, value):
        self.label2.setText(str(value))
        self.current_pos[1] = value
        self.onGlobalChange()

    def onSlider3Change(self, value):
        self.label3.setText(str(value))
        self.current_pos[2] = value
        self.onGlobalChange()

    def onSlider4Change(self, value):
        self.label4.setText(str(value))
        self.current_pos[3] = value
        self.onGlobalChange()

    def onGlobalChange(self):
        self.calculation()
        self.labelUpdate()
        self.draw_bar_chart()

    def labelUpdate(self):
        self.val1.setText(f"P1: {str(round(self.current_value[0],2))}  P2: {str(round(self.current_value[1],2))}  P3: {str(round(self.current_value[2],2))}  P4: {str(round(self.current_value[3],2))}  ")

    def calculation(self):

        if self.current_method == NormMethod.SIGMOID.value:
            self.current_value = self.methods.sigmoid(self.current_pos)
        if self.current_method == NormMethod.SQRT_HS.value:
            self.current_value = self.methods.squirt_hs(self.current_pos)
        if self.current_method == NormMethod.SIGMOID_HS.value:
            self.current_value = self.methods.sigmoid(self.current_pos)
        if self.current_method == NormMethod.DEFAULT.value:
            self.current_value = self.methods.default(self.current_pos)

