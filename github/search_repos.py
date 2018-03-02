#!/usr/local/bin/python3.6
import requests

import json
import pprint
import shutil
import argparse
import os

from collections import namedtuple

GITHUB = 'https://api.github.com'
AUTH = None
OFFLINE = False
PP = pprint.PrettyPrinter(indent=4, depth=2) # for debugging

SAVE_FILE = "urls.save.json"
CACHE_FILE = "repo_req.cache.json"

def parse_args():
    '''
    Provide cmd line interface.
    '''
    parser = argparse.ArgumentParser(description='Search for repos on GitHub.')
    parser.add_argument('search_terms', type=str, nargs='+',
                    help='search terms in GitHub query')
    parser.add_argument('-l', '--language', default='python', dest='language',
                        nargs=1, action='store',
                        help='language of the repo (according to GH)')
    parser.add_argument('-p', '--pages', default=4, type=int,
                    help='how many pages of results (100 per page)')
    return vars(parser.parse_args())


def auth():
    '''
    Authorize from a credentials file
    '''
    auths = {}
    with open('cred.key', 'r') as f:
        creds = json.load(f)
        auths["github"] = ( creds["github"]["username"], creds['github']["password"] )
    return auths

def load_cache():
    '''
    Mock out the resuponse from a cached json file, for faster development
    '''
    Cache = namedtuple("Cache", ["url", "text"])
    with open(CACHE_FILE, 'r') as f:
        x = f.read()
    return Cache(url="cache", text=x)


def quick_save(name, query, query_url, urls, page):
    '''
    Introduce some persistance. Save everything to a json file. This
    can be replaced with a database later.
    '''
    save = {
        "name":name,
        "query": query,
        "query_url":query_url,
        "urls":urls,
        "page":page,
    }

    save_file = SAVE_FILE
    bk_file = "." + SAVE_FILE

    mode = "r+" if os.path.exists(save_file) else "w+"
    with open(save_file, mode) as f:
        content = f.read()
        js = json.loads(content if len(content) else "[]")
        js.append(save)

    with open(bk_file, "w") as g:
        g.write(json.dumps(js, sort_keys=True, indent=4, separators=(',', ': ')))

    shutil.copy(bk_file, SAVE_FILE)
    os.remove(bk_file)

def query_github(end, payload):
    '''
    Send an API Request to GitHUb
    '''
    global AUTH
    if not AUTH:
        AUTH = auth()
    if OFFLINE:
        return load_cache()
    return requests.get(GITHUB + end, auth=AUTH["github"], params=payload)

def make_code_search_request(terms, language, page):
    '''
    Create request and endpoint to send to GitHub
    '''
    end = "/search/code"
    filters = " language:{} fork:false".format(language)

    payload = {
        'q': " ".join(terms) + " language:{}".format(language),
        'order': 'desc',
        'per_page':100,
        'page': page

    }
    return end, payload

def extract_repo_data(repo_request_json):
    '''
    Parse through return GitHub json and extract important info
    '''
    repos = []
    for raw_repo in repo_request_json["items"]:
        # PP.pprint(raw_repo)  # debug
        r = {
            'url': raw_repo["repository"]["html_url"],
            'name': raw_repo["repository"]["full_name"],
            'id': raw_repo["repository"]['id'],
            'score': raw_repo['score'],
            'searched': False
        }
        repos.append(r)
    return repos

def search_repos(search_terms, language, pages):
    '''
    Conduct searches for repositories and save the data.
    '''
    for p in range(1, pages+1): # GH page from 1 not 0
        req = make_code_search_request(search_terms, language, p)
        res = query_github(*req)

        print("Request {}: {}: RC: {}".format(p, res.url, res.status_code))


        res_json = json.loads(res.text)
        repos = extract_repo_data(res_json)

        quick_save("Test", "|".join(search_terms), res.url, repos, p)


def main(search_terms, language, pages):
    print("Searching GH code for {} {} pages with: {} ".format(
        pages, language, " ".join(search_terms)))
    search_repos(search_terms, language, pages)

if __name__ == "__main__":
    main(**parse_args())
