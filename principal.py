from  Verificacion_Manual import *
from Grafica import *

V=manual()
V.split_data()
V.build_red()
V.compiler()
V.ajustar_modelo()


'''
m=modelado()
m.build_red()
m.compiler()
m.ajustar_modelo()
m.evaluar_modelo()
m.predicciones()
'''

