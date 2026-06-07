from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "SAES 2.0 funcionando"

if __name__ == "__main__":
    app.run()