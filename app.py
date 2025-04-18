#importar librerias
import streamlit as st
import pickle
import pandas as pd
import numpy as np 

#Extrar los archivos pickle
with open('modelo_log_reg.pkl', 'rb') as archivo:
    model = pickle.load(archivo)

coef_dict = model.params.to_dict()

#funcion para clasificar las plantas 
def classify(num):
    if num <= 0.33:
        return '❌ Alto riesgo de incumplimiento'
    elif num <= 0.67:
        return '⚠️ Riesgo moderado de incumplimento'
    else:
        return '✅ Bajo riesgo de incumplimiento'

def calcular_probabilidad_fila(data: dict, coef_dict=coef_dict):

    z = float(coef_dict['const'])
    for var, coef in coef_dict.items():
        if var != 'const':
            z += float(coef) * float(data[var])

    # Probabilidad logística
    prob = 1 / (1 + np.exp(-z))

    return 1 - prob

def main():
    #titulo
    st.title('Aplicación - Medición de Incumplimiento de Pago')
    #titulo de sidebar
    st.write("Esta aplicación permite predecir la probabilidad de incumplimiento de pago usando un modelo de regresión logística.")
    st.markdown("**¿Cómo usar esta aplicación?**")
    st.markdown("""
    1. Ingrese sus datos personales en la barra lateral.
    2. Haga clic en el botón **RUN**.
    3. Visualice la probabilidad estimada de incumplimiento.
    """)

    
    #funcion para poner los parametrso en el sidebar
    def user_input_parameters():
        opciones_genero = ['Masculino','Femenino']
        genero = st.sidebar.selectbox('Género',opciones_genero)

        if genero == 'Masculino':
            sexo_m = 1
            sexo_f = 0
        else:
            sexo_m = 0
            sexo_f = 1

        # Macrorisk
        dict_macrorisk_num = {
            '1. Prospectable': 1,
            '2. Examinable': 2,
            '3. Peligroso': 3,
            '4. UCI': 4
        }

        opciones_macrorisk = ['1. Prospectable','2. Examinable', '3. Peligroso', '4. UCI']
        macrorisk_num = st.sidebar.selectbox('Riesgo Crediticio',opciones_macrorisk)

        # Situación laboral
        dict_bin_sit_laboral = {
            'Desempleado': 4,
            'Informal': 3,
            'Independiente': 2,
            'Dependiente': 1
        }

        opciones_sit_lab = ['Dependiente','Desempleado','Independiente','Informal']
        fact_sit_lab = st.sidebar.selectbox('Situación laboral', opciones_sit_lab)

        # Rango de Endeudamiento
        dict_rango_endeudamiento = {
            'Sin línea': 3,
            'De 0% a 25%': 3,
            'De 26% a 100%': 2,
            'Más de 100%': 1
        }

        opciones_rng_end = ['Sin línea','De 0% a 25%','De 26% a 100%','Más de 100%']
        fact_rng_end = st.sidebar.selectbox('Nivel de Endeudamiento', opciones_rng_end)

        data = {'Sexo_M': sexo_m,
                'Sexo_F': sexo_f,
                'macrorisk_num': dict_macrorisk_num[macrorisk_num],
                'Factor_1_RNG_END': dict_bin_sit_laboral[fact_sit_lab],
                'Factor_1_SIT_LAB': dict_rango_endeudamiento[fact_rng_end],
                'const': 1
                }

        #features = pd.DataFrame(data, index=[0])
        return data

    data = user_input_parameters()

    texto_integrantes = """
    <strong>Integrantes</strong>:
    <ul>
        <li>Alessandro Perales</li>
        <li>Antony Vargas</li>
        <li>Gabriel Ochoa</li>
        <li>Jordyn Urbina</li>
    </ul>
    """

    # Texto "anclado" al final visualmente
    st.sidebar.markdown(texto_integrantes,unsafe_allow_html=True)

    if st.button('RUN'):
        prob = calcular_probabilidad_fila(data)
        st.success(f'Probabilidad de incumplimento de pago: {(1 - prob):.4f}')
        st.success(classify(prob))

if __name__ == '__main__':
    main()
    
