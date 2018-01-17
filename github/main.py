import requests

import json
import sqlite3
import pprint
import shutil
import os.path

from collections import namedtuple

import argparse

GITHUB = 'https://api.github.com'
AUTH = None
OFFLINE = True
PP = pprint.PrettyPrinter(indent=4, depth=2)

def auth():
    auths = {}
    with open('cred.key', 'r') as f:
        creds = json.load(f)
        auths["github"] = ( creds["github"]["username"], creds['github']["password"] )
    return auths

def load_cache():
    Cache = namedtuple("Cache", ["url", "text"])
    with open("repo_req.cache.json", 'r') as f:
        x = f.read()
    return Cache(url="cache", text=x)


def quick_save(name, query, query_url, urls, page):
    save = { 
        "name":name, 
        "query": query, 
        "query_url":query_url, 
        "urls":urls,  
        "page":page,
    }

    mode = "r+" if os.path.exists("urls.save.json") else "w+"
    with open("urls.save.json", mode) as f:
        _c = f.read()
        bk = json.loads(_c if len(_c) else "[]" ) 
        bk.append(save)
    with open(".urls.save.json", "w") as g:
        g.write(json.dumps(bk, sort_keys=True,
                          indent=4, separators=(',', ': ')))
    shutil.copy(".urls.save.json", "urls.save.json")


def query_github(end, payload):
    global AUTH
    if not AUTH:
        AUTH = auth()
    if OFFLINE:
        return load_cache()
    return requests.get(GITHUB + end, auth=AUTH["github"], params=payload)
    
def make_repository_search_request(terms, language, page):  
    end = "/search/repositories"
    filters = " language:{} fork:false".format(language)

    payload = {
        'q': " ".join(terms) + " language:{}".format(language),
        'order': 'desc',
        'per_page':100,
        'page': page

    }
    return end, payload
    
def extract_repo_data(repo_request_json):
    repos = [{}]
    for raw_repo in repo_request_json["items"]:
        r = {
            'url': raw_repo["clone_url"],
            'name': raw_repo["name"],
            'id': raw_repo['id'],
            'size': raw_repo['size'],
            'stars': raw_repo["stargazers_count"],
            'searched': False
        }
        repos.append(r)
    return repos

def search_repos(search_terms, language, pages):
    
    for p in range(pages):
        req = make_repository_search_request(search_terms, language, p)
        res = query_github(*req)

        res_json = json.loads(res.text)
        repos = extract_repo_data(res_json)
        
        quick_save("Test", "|".join(search_terms), res.url, repos, p)



def main():
    print("Hello World")
    search_terms = ["poker"]
    language = "python"

    search_repos(search_terms, language, 3)





if __name__ == "__main__":
    main()