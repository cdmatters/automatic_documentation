# import requests
import json

GITHUB = 'https://api.github.com'
AUTH = None



def make_repository_search(language, user):
    end = "/search/repositories"

    q = "language:{}".format(language)
    
    payload = {
        'q': q,
        'order': 'desc',
    }

    
    x = requests.get(GITHUB + end, auth=('user', 'condnsdmatters'), data=payload)
    print(x.text)


def main():
    print("Hello World")

def auth():
    with open('cred.key', 'r') as f:
        crendentials = json.load(f)
        gh_cred = crendentials["github"]
    return (gh_cred["username"], gh_cred["password"] )



if __name__ == "__main__":
    AUTH = auth()
    print(AUTH)
    
    # make_repository_search("python", USERNAME)