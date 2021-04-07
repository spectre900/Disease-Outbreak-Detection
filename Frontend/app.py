from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)

def retrieve_data(disease_name):
    filename = './static/data/' + disease_name + '.txt'

    # Using readlines()
    dataFile = open(filename, 'r')
    Lines = dataFile.readlines()
    
    data = []
    # Strips the newline character
    for line in Lines:
        data.append(line.strip())

    data1 = data[:5]
    data2 = data[5]
    data3 = {}

    for i in range(6, len(data), 2):
        key = data[i]
        value = data[i+1]
        data3[key] = value

    return data1, data2, data3

@app.route('/')
def root():
    return render_template('home_page.html')

@app.route('/home')
def home():
    return render_template('home_page.html')

@app.route('/statistics/<disease>')
def statistics(disease):
    
    about_disease, disease_transmission, disease_prevention  = retrieve_data(disease)
    if disease == 'dengue':
        retrieve_data(disease)
        return render_template('disease_statistics.html', disease_name = disease.capitalize(), about_disease = about_disease, disease_transmission = disease_transmission, disease_prevention = disease_prevention)

@app.route('/about')
def about():
    return render_template('about_us.html')

if __name__ == '__main__':
    app.run(debug=True)