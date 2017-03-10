from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello world!'

# Training API
@app.route('/train?basedata=<basedata>&dataname=<dataname>')
def train_data(basedata, dataname):
    return '%s, %s' % basedata, dataname