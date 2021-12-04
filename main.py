from flask import Flask, request
import json
import requests as axios
from typing import List, Optional
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

app = Flask(__name__)

ACCESS_TOKEN = ""
model: Optional[Pipeline] = None
dataset: Optional[DataFrame] = None

@app.before_first_request
def boot():
    global ACCESS_TOKEN
    global model
    global dataset

    vectorizer = TfidfVectorizer()
    balancer = RandomOverSampler()
    ml_layer = MultinomialNB()
    model = make_pipeline_imb(vectorizer, balancer, ml_layer)
    dataset = pd.read_csv("databases/dataset.csv")
    model.fit(dataset["content"], dataset["outcome"])

    with open("databases/access_token.json") as f:
        ACCESS_TOKEN = json.loads(f.read().replace("\n", ''))["access_token"]


def get_value_by_key(obj: object, key: str):
    try:
        value = obj[key]
        return value
    except:
        return None


class FacebookComment:
    identifier: str
    message: str
    rating: str

    def __init__(self, identifier: str, message: str):
        self.identifier = identifier
        self.message = message
        result = model.predict(np.array([message]))
        self.rating = result[0]

    def to_json_object(self) -> object:
        return {
            "identifier": self.identifier,
            "message": self.message,
            "rating": self.rating
        }

    def to_json_string(self) -> str:
        return json.dumps(self.to_json_object())


class FacebookPost:
    identifier: str
    content: str
    url: str
    comments: List[FacebookComment]

    def __init__(self, identifier: str, content: str, url: str, comments: List[FacebookComment]):
        self.identifier = identifier
        self.content = content
        self.url = url
        self.comments = comments

    def to_json_object(self) -> object:
        return {
            "identifier": self.identifier,
            "content": self.content,
            "url": self.url,
            "comments": [comment.to_json_object() for comment in self.comments]
        }

    def to_json_string(self) -> str:
        return json.dumps(self.to_json_object())


@app.route("/account/login", methods=["POST"])
def login_handler():
    username = request.json["username"]
    password = request.json["password"]
    with open('databases/users.json', 'r') as f:
        content = f.read().replace('\n', '')
        users = json.loads(content)['users']
        for user in users:
            if user["username"] == username and user["password"] == password:
                json_response = {
                    "status": "OK",
                    "page_ids": user["page_ids"]
                }
                return json.dumps(json_response)

    json_response = {
        "status": "error"
    }
    return json.dumps(json_response)


@app.route("/account/register", methods=["POST"])
def register_handler():
    username = request.json["username"]
    password = request.json["password"]
    with open('databases/users.json', 'r') as f:
        content = f.read().replace('\n', '')
        users = json.loads(content)['users']
        for user in users:
            if user["username"] == username:
                json_response = {
                    "status": "error"
                }
                return json.dumps(json_response)

    with open('databases/users.json', 'w') as f:
        new_user = {
            "username": username,
            "password": password,
            "page_ids": []
        }
        new_users = users + [new_user]
        new_database = {
            "users": new_users
        }
        f.write(json.dumps(new_database))
        json_response = {
            "status": "OK"
        }
        return json.dumps(json_response)


@app.route("/account/add_page_id", methods=["POST"])
def add_page_id_handler():
    username = request.json["username"]
    page_id = request.json["page_id"]
    with open('databases/users.json', 'r') as f:
        content = f.read().replace('\n', '')
        users = json.loads(content)['users']
        for user in (user for user in users if user["username"] == username):
            if not (any(True for pid in user["page_ids"] if pid == page_id)):
                user["page_ids"] = user["page_ids"] + [page_id]

    with open('databases/users.json', 'w') as f:
        new_database = {
            "users": users
        }
        f.write(json.dumps(new_database))

    json_response = {
        "status": "OK"
    }
    return json.dumps(json_response)


@app.route("/page/<string:page_id>/feeds", methods=["GET"])
def list_feeds_handler(page_id):
    def make_facebook_comment(obj) -> FacebookComment:
        print(obj["message"])
        return FacebookComment(
            identifier = obj["id"],
            message = obj["message"],
        )

    content = axios.get(f"https://graph.facebook.com/{page_id}/feed?access_token={ACCESS_TOKEN}").json()

    print(content)

    posts = [post for post in content["data"] if get_value_by_key(post, "message") is not None]

    fb_posts = [
        FacebookPost(
            identifier = post["id"],
            content = post["message"],
            url = get_value_by_key(post, "link"),
            comments = [] if get_value_by_key(post, "comments") is not None else [make_facebook_comment(comment_obj) for comment_obj in post["comments"]["data"]]
        )
        for post in posts
    ]

    return json.dumps([post.to_json_object() for post in fb_posts])

@app.route("/feedback", methods=["POST"])
def feedback_hanlder():
    global dataset
    content = request.json["content"]
    outcome = request.json["outcome"]
    new_column = pd.DataFrame([[content, outcome]], columns = ["content", "outcome"])
    dataset = dataset.append(new_column, ignore_index = True)
    dataset.to_csv("databases/dataset.csv", index = False)
    model.fit(dataset["content"], dataset["outcome"])
    return "success"