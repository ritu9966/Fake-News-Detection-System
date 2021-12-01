from flask import Flask,render_template,request,redirect
import pickle as pkl
import numpy as np



app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/fakenews',methods=['GET','POST'])
def predict():
    nws=str(request.args.get("text"))

    with open('model.pkl','rb') as f:
        model=pkl.load(f)

    with open('model1.pkl','rb') as f:
        model1=pkl.load(f)
 
    st=model1.transform([nws])
    st=model.predict(st)

    
    return render_template('after.html',data=st[0])
    
        






if __name__=="__main__":
    app.run(debug=True)

