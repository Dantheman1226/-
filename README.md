# Capstone project for DAAN888
1. Title: Toxic Comment Classification</br>

2. Professor: Youakim Badr. Member: Shaojie Lei, Chen-I Huang</br>


3. Summary
Our project focus on identifying the negative online toxic comments. Then, classify these negative comments into more detailed categories such as toxic, severe toxic, obscene, threat, insult and identity hate. 
Discussing thing you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, 
leading many communities to limit or completely shut down user comments. Moreover, anti-discrimination and racial equality are getting more and more attention. 
For example, the BLM recently, we can use the system we develop to prevent the discrimination and keep the harmony online. We used the dataset from Kaggle that contain over 150,000 comment. 
After analyze the data and data cleaning, we use two deep learning (LSTM and BERT) to classify it. The result of our model is beyond our expectation. The accuracy of our baseline model (LSTM) and BERT can detect toxic comment successfully, baseline model is around 95% and BERT is 100%. 
After training the model, we face the problem of deploying it. In the beggining, we want to build a Google Chrome Extension to detect the comment on the website in real time. However, without the experience needed, we follow a tutorial and adapt it's code so we can build a Chrome Extension successfuly.
 Because most of the code is not from us, we also build a streamlit as a simple version to deploy our model. 
People can paste the text on it and use BERT to detect the probility of toxic. 

4. Code
Our code included two Jupyter notebook and one Pyscript for streamlit. The first Jupyter notebook is data visualization and the second Jupyter notebook is model training. To run the streamlit is easy, just type streamlit run app.py. Becareful of the streamlit, since it use BERT, it need GPU for prediction. 


5. Requirements</br>
h5py==2.10.0</br>
Keras==2.4.3</br>
matplotlib @ file:///C:/ci/matplotlib-base_1592837548929/work</br>
nltk @ file:///tmp/build/80754af9/nltk_1592496090529/work</br>
numpy==1.18.5</br>
pandas @ file:///C:/ci/pandas_1592833613419/work</br>
scikit-learn @ file:///C:/ci/scikit-learn_1592853510272/work</br>
sklearn==0.0</br>
streamlit==0.71.0</br>
tensorboard==2.3.0</br>
tensorboard-plugin-wit==1.7.0</br>
tensorflow==2.3.0</br>
tensorflow-estimator==2.3.0</br>
tensorflow-gpu==2.3.0</br>
tensorflow-gpu-estimator==2.3.0</br>
tensorflow-hub==0.9.0</br>
tensorflowjs==2.7.0</br>
tokenizers==0.8.1rc2</br>
torch==1.6.0</br>
torchvision==0.7.0</br>
transformers==3.3.1</br>
