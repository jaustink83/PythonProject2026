from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def chart_example():
    # Data often comes from a database, but we use lists here for simplicity
    labels = ["January", "February", "March", "April", "May"]
    data_values = [12, 19, 3, 5, 2]

    return render_template('index.html', labels=labels, values=data_values)

@app.route('/students')
def studentchart():
    # Data often comes from a database, but we use lists here for simplicity
    labels = ["John", "Irina", "Bob", "Jeff"]
    data_values = [90, 95, 70, 10]

    return render_template('index.html', labels=labels, values=data_values)


if __name__ == '__main__':
    app.run(debug=True)