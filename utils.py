import appdirs
import hashlib
import io
import logging
import os
import requests

from requests.auth import HTTPBasicAuth
from urllib.parse import urlparse


def uri_to_url(uri, owner, repo):
    if uri.startswith('http'):
        return uri
    elif uri.startswith('repo://'):
        link_data = uri.split("repo://")[-1].split("/")
        commit, tree_path = link_data[0], "/".join(link_data[1:])
        return f"https://dagshub.com/api/v1/repos/{owner}/{repo}/raw/{commit}/{tree_path}"
    raise FileNotFoundError(f'Unkown URI {uri}')


def cache_path():
    cache = appdirs.user_cache_dir(appname='relaxml')
    os.makedirs(cache, exist_ok=True)
    return cache


def download_url(url, user, token):
    cache = cache_path()
    parsed_url = urlparse(url)
    url_filename = os.path.basename(parsed_url.path)
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    filepath = os.path.join(cache, url_hash + '__' + url_filename)
    if not os.path.exists(filepath):
        logging.info('Download {url} to {filepath}'.format(url=url, filepath=filepath))
        auth = HTTPBasicAuth(user, token)
        res = requests.get(url, stream=True, auth=auth)
        res.raise_for_status()
        with io.open(filepath, mode='wb') as f:
            f.write(res.content)
    return filepath