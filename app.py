from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
import joblib
import datetime
# https://www.tutorialspoint.com/flask
import flask

app = Flask(__name__)

snap_CA_days= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
snap_WI_days= [2, 3, 5, 6, 8, 9, 11, 12, 14, 15]
snap_TX_days= [1, 3, 5, 6, 7, 9, 11, 12, 13, 15]
#loading the encoders
state_le= joblib.load('state_le.pkl')
store_le= joblib.load('store_le.pkl')
cat_le= joblib.load('cat_le.pkl')
dept_le= joblib.load('dept_le.pkl')
item_le= joblib.load('item_le.pkl')
id_le= joblib.load('id_le.pkl')

model_CA_1= joblib.load('model_CA_1')
model_CA_2= joblib.load('model_CA_2')
model_CA_3= joblib.load('model_CA_3')
model_CA_4= joblib.load('model_CA_4')
model_WI_1= joblib.load('model_WI_1')
model_WI_2= joblib.load('model_WI_2')
model_WI_3= joblib.load('model_WI_3')
model_TX_1= joblib.load('model_TX_1')
model_TX_2= joblib.load('model_TX_2')
model_TX_3= joblib.load('model_TX_3')

models= [model_CA_1, model_CA_2, model_CA_3, model_CA_4, model_WI_1, model_WI_2, model_WI_3, model_TX_1, model_TX_2, model_TX_3]

@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    data = list(data.values())

    state_id= data[0]
    store_id= data[1]
    cat_id= data[3]
    dept_id= data[4]
    item_id= data[2]
    ids= item_id +'_'+ store_id
    
    date= str(data[5])
    date= pd.to_datetime(date)
    
    dates=[] #list of dates from the days given
    for i in range(1, 29):
        start_date= pd.to_datetime(date) - datetime.timedelta(1)
        datess= start_date + datetime.timedelta(i)
        dates.append(datess)
    
    state_id= state_le.transform([state_id])[0]
    store_id= store_le.transform([store_id])[0]
    cat_id= cat_le.transform([cat_id])[0]
    dept_id= dept_le.transform([dept_id])[0]
    item_id= item_le.transform([item_id])[0]
    ids= id_le.transform([ids])[0] 
    
    model= models[store_id]
        
    #data= np.array([state_id, store_id, cat_id, dept_id, item_id, ids, wday, month, year, day, snap_CA, snap_TX, snap_WI]).reshape(1,-1)
    
    cols= ['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id', 'ids', 'wday', 'month', 'year', 'day',
       'snap_CA', 'snap_TX', 'snap_WI']
    
    df= pd.DataFrame(columns= cols)
    df['state_id']= [state_id] * 28
    df['store_id']= [store_id] * 28
    df['cat_id']= [cat_id] * 28
    df['dept_id']= [dept_id] * 28
    df['item_id']= [item_id] * 28
    df['ids']= [ids] * 28
    df['date']= dates
    df['wday']= df['date'].dt.weekday
    df['month']= df['date'].dt.month
    df['year']= df['date'].dt.year
    df['day']= df['date'].dt.day
    df['snap_CA']= df['day'].apply(lambda x: 1 if x in snap_CA_days else 0)
    df['snap_TX']= df['day'].apply(lambda x: 1 if x in snap_TX_days else 0)
    df['snap_WI']= df['day'].apply(lambda x: 1 if x in snap_WI_days else 0)
    df.drop('date', axis= 1, inplace= True)
    
    prediction= model.predict(df)
        
    for i in range(len(prediction)):
        prediction[i]= np.round(prediction[i],0).astype(int)
    
    # return  list(prediction)

    return jsonify({'prediction for the sales of next 28 days are': list(prediction)})
    # return render_template('index.html', prediction_text= 'prediction for the sales of next 28 days are {}'.format(prediction))


if __name__ == '__main__':
    app.run(debug= True)
