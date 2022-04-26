from flask import Flask, render_template, request
import requests
from logregression import logistic_regression

app = Flask(__name__)


get = requests.get('https://api.npoint.io/c790b4d5cab58020d391')
blog_posts = get.json()


@app.route('/')
def home():

    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    sales = request.form['sales']
    profit = request.form['profit']
    state = request.form['state']
    subcat = request.form['subcat']

    results = logistic_regression(sales, profit, state, subcat)

    return render_template("index2.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)