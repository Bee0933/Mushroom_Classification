import streamlit as st
import PIL
from PIL import Image
import numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import joblib as jl


#encoded text features
capShape = {'Bell':'b','conical':'c','convex':'x','flat':'f','knobbed':'k','sunken':'s'}
capSurface = {'fibrous':'f','grooves':'g','scaly':'y','smooth':'s'}
capColor = {'brown':'n','buff':'b','cinnamon':'c','gray':'g','green':'r','pink':'p','purple':'u','red':'e','white':'w','yellow':'y'}
bruises_ = {'bruises':'t','no':'f'}
odor_ = {'almond': 'a', 'anise':'l', 'creosote':'c', 'fishy':'y','foul':'f','musty':'m','none':'n', 'pungent':'p','spicy':'s'}
gillSpacing = {'close':'c','crowded':'w'}
gillSize = {'broad':'b','narrow':'n'}
gillColor =  {'black':'k','brown':'n','buff':'b','chocolate':'h','gray':'g','green':'r','orange':'o','pink':'p','purple':'u','red':'e','white':' w', 'yellow':'y'}
stalkShape = {'enlarging':'e', 'tapering': 't'}
stalkRoot = {'bulbous':'b', 'club':'c', 'equal': 'e', 'rooted':'r','missing':'?'}
stalkSurfaceAboveRing = {'fibrous': 'f', 'scaly':'y','silky':'k','smooth' :'s'}
stalkSurfaceBelowRing = {'fibrous':'f', 'scaly':'y','silky':'k','smooth':'s'}
stalkColorAboveRing = {'brown':'n', 'buff':'b', 'cinnamon': 'c', 'gray':'g','orange': 'o','pink':'p','red':'e', 'white': 'w', 'yellow':'y'}
stalkColorBelowRing =  {'brown':'n', 'buff':'b','cinnamon':'c','gray':'g','orange':'o','pink':'p','red':'e', 'white':'w','yellow':'y'}
veilColor =  {'brown':'n', 'orange':'o', 'white':'w', 'yellow':'y'}
ringNumber = {'none':'n', 'one':'o','two':'t'}
ringType = {'evanescent':'e', 'flaring':'f','large':'l', 'none':'n','pendant':'p'}
sporePrintColor = {'black': 'k', 'brown' :'n', 'buff':'b', 'chocolate': 'h', 'green':'r', 'orange':'o','purple': 'u','white':'w', 'yellow':'y'}
population_ = {'abundant':'a', 'clustered': 'c', 'numerous': 'n', 'scattered':'s','several':'v', 'solitary': 'y'}
habitat_ = {'grasses':'g', 'leaves':'l','meadows':'m','paths':'p','urban':'u','waste':'w','woods':'d'}

#side bar
st.sidebar.header('Select your features')

cap_shape = capShape[st.sidebar.selectbox('cap_shape', capShape.keys())]
cap_surface = capSurface[st.sidebar.selectbox('cap_surace', capSurface.keys())]
cap_Color = capColor[st.sidebar.selectbox('cap_color', capColor.keys())]
bruises = bruises_[st.sidebar.selectbox('bruises', bruises_.keys())]
odor = odor_[st.sidebar.selectbox('odor', odor_.keys())]
gill_spacing = gillSpacing[st.sidebar.selectbox('gill_spacing', gillSpacing.keys())]
gill_size = gillSize[st.sidebar.selectbox('gill_size', gillSize.keys())]
gill_color = gillColor[st.sidebar.selectbox('gill_color', gillColor.keys())]
stalk_shape = stalkShape[st.sidebar.selectbox('gill_shape', stalkShape.keys())]
stalk_root = stalkRoot[st.sidebar.selectbox('stalk_root',stalkRoot.keys())]
stalk_surface_above_ring = stalkSurfaceAboveRing[st.sidebar.selectbox('stalk_surface_above_ring', stalkSurfaceAboveRing.keys())]
stalk_surface_below_ring = stalkSurfaceBelowRing[st.sidebar.selectbox('stalk_surface_below_ring', stalkSurfaceBelowRing.keys())]
stalk_color_above_ring = stalkColorAboveRing[st.sidebar.selectbox('stalk_color_above_ring', stalkColorAboveRing.keys())]
stalk_color_below_ring = stalkColorBelowRing[st.sidebar.selectbox('stalk_color_below_ring', stalkColorBelowRing.keys())]
veil_color = veilColor[st.sidebar.selectbox('veil_color',veilColor.keys())]
ring_number = ringNumber[st.sidebar.selectbox('ring_number',ringNumber.keys())]
ring_type = ringType[st.sidebar.selectbox('ring_type',ringType.keys())]
spore_print_color = sporePrintColor[st.sidebar.selectbox('spore_print_color',sporePrintColor.keys())]
population = population_[st.sidebar.selectbox('population',population_.keys())]
habitat = habitat_[st.sidebar.selectbox('habitat',habitat_.keys())]

#encodings
cap_shape_encode = {'b':0, 'c':1,'f':2,'k':3,'s':4,'x': 5}
cap_surface_encode = {'f':0, 'g':1, 's':2,'y':3}
cap_Color_encode = {'b':0,'c':1,'e':2,'g':3,'n':4 ,'p':5,'r':6,'u':7,'w':7,'y':9}
bruises_encode = {'f':0,'t':1}
odor_encode = {'a':0, 'c':1,'f':2,'l':3,'m':4,'n':5,'p':6,'s':7,'y':7}
gill_spacing_encode =  {'c':0,'w':1}
gill_size_encode = {'b':0, 'n':1}
gill_color_encode =  {'b':0,'e':1,'g':2,'h':3,'k':4,'n':5,'o':6,'p':7,'r':8,'u':9,'w':10,'y':11}
stalk_shape_encode = {'e':1,'t':1}
stalk_root_encode = {'?':0,'b':1, 'c':2,'e':3, 'r':4}
stalk_surface_above_ring_encode = {'f':0, 'k':1,'s':2,'y':3}
stalk_surface_below_ring_encode = {'f':0,'k':1,'s':2,'y':3}
stalk_color_above_ring_encode = {'b':0,'c':1,'e':2, 'g':3, 'n':4,'o':5,'p':6,'w':7,'y':8}
stalk_color_below_ring_encode = {'b':0,'c':1, 'e':2,'g':3, 'n':4, 'o':5, 'p':6, 'w':7,'y':8}
veil_color_encode = {'n':0,'o':1,'w':2,'y':3}
ring_number_encode = {'n':0,'o':1,'t':2}
ring_type_encode = {'e': 0,'f':1,'l':2,'n':3,'p':4}
spore_print_color_encode = {'b':0,'h':1,'k':2, 'n':3,'o':4,'r':5,'u':6,'w':7,'y':8}
population_encode = {'a':0,'c':1,'n':2,'s':3,'v':4,'y':5}
habitat_encode = {'d':0,'g':1, 'l':2,'m':3,'p':4,'u':5,'w':6}

image = Image.open('static/Group 1.png') #read Hreo form image
st.image(image=image,use_column_width=True)
st.title('MUSHROOM CLASSIFICATION')
st.write("""
This Machine Learning app predits if a mushroom sample is *** Edible *** or *** Poisonous *** based on the given input values from the input sidebar
##### Features Description
""")

def preditions():
    data_read = { 'cap shape': cap_shape_encode[cap_shape], 'cap surface': cap_surface_encode[cap_surface], 'cap color':cap_Color_encode[cap_Color],
                  'bruises': bruises_encode[bruises], 'odor': odor_encode[odor], 'gill spacing': gill_spacing_encode[gill_spacing],'gill size': gill_size_encode[gill_size],
                  'gill color': gill_color_encode[gill_color],'stalk shape,':stalk_shape_encode[stalk_shape],'stalk root' :stalk_root_encode[stalk_root],
                  'stalk surface above ring': stalk_surface_above_ring_encode[stalk_surface_above_ring],
                  'stalk surface below ring':stalk_surface_below_ring_encode[stalk_surface_below_ring],
                  'stalk color above ring': stalk_color_above_ring_encode[stalk_color_above_ring], 'stalk color below ring': stalk_color_below_ring_encode[stalk_color_below_ring],
                  'veil color': veil_color_encode[veil_color], 'ring number': ring_number_encode[ring_number],'ring type': ring_type_encode[ring_type],
                  'spore print color': spore_print_color_encode[spore_print_color], 'population':population_encode[population], 'habitat': habitat_encode[habitat]
                  }
    data_frame = pd.DataFrame(data_read, index=[0])
    # st.write(data_frame)
    model = jl.load('trainedMushroomClassifier.sav', 'r')
    if st.button('Predict Mushroom'):
        result = model.predict(data_frame)
        if result == 1:
            image_poison = Image.open('static/poisonous.png')  # read Hreo form image
            st.image(image=image_poison, use_column_width=True)
        else:
            image_edible = Image.open('static/Edible.png')  # read Hreo form image
            st.image(image=image_edible, use_column_width=True)
    else:
        st.write('click the predict button to process your mushroom input data and make prediction')


preditions()




