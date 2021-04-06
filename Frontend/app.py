from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('home_page.html')

@app.route('/home', methods = ['POST', 'GET'])
def home():
    return render_template('home_page.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/statistics/<disease>')
def statistics(disease):
    if disease == 'dengue':
        return render_template('disease_statistics.html', disease_name = disease.upper())
    elif disease == 'flu':
        return render_template('disease_statistics.html', disease_name = disease.upper())

if __name__ == '__main__':
    app.run(debug=True)