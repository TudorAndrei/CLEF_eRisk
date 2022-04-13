from json import loads

from faker import Faker
from flask import Flask, jsonify, request

fake = Faker()


nicks = [f"subject{x}" for x in range(1, 20)]
ids = [x for x in range(1, 20)]
number = 0


def generate_writing(number):
    writing = []
    for nick, id_ in zip(nicks, ids):
        writing.append(
            {
                "id": id_,
                "number": number,
                "nick": nick,
                "title": fake.sentence(),
                "content": fake.sentence(),
                "date": fake.date(),
            }
        )
    return writing


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "GET":
        global number
        writings = generate_writing(number)
        number += 1
        return jsonify(writings)
    if request.method == "POST":
        data = request.get_json()
        print(data)
        return jsonify({"message": "Success"})
